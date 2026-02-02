import argparse
import copy
import json
import logging
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from bittensor_network import BittensorNetwork
from bittensor_network.bittensor_config import BittensorConfig
from core.evaluations import verify_solution_quality
from validator.auth import ValidatorAuth
from validator.capacitorless_weight_manager import CapacitorlessWeightManager
from validator.capacitorless_sticky_weight_manager import (
    CapacitorlessStickyBurnSplitWeightManager,
)
from validator.metrics_logger import ValidatorMetricsLogger
from validator.relay_client import RelayClient
from validator.relay_poller import RelayPoller
from validator.submission_scheduler import SubmissionScheduler
from validator.weight_manager import WeightManager

try:
    from validator.contract_manager import ContractManager
except Exception:  # pragma: no cover
    ContractManager = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(argv=None):
    """
    Main function for the validator client.
    Initializes all necessary components and starts the background services.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        "-c",
        default="validator_config.yaml",
        help="Path to validator YAML config (default: validator_config.yaml)",
    )
    parser.add_argument(
        "--accept-test",
        action="store_true",
        help="Poll relay test-submission queue for logging only (UID0-only; no weights).",
    )
    args, remaining = parser.parse_known_args(argv)
    accept_test = bool(getattr(args, "accept_test", False))

    logging.info("=" * 60)
    logging.info("Starting Validator Node")
    logging.info("=" * 60)

    try:
        bt_argv = (
            [sys.argv[0], *remaining]
            if argv is None
            else ["validator_node", *remaining]
        )
        config = BittensorConfig.get_bittensor_config(args.config, bt_argv=bt_argv)
        logging.info(f"Loaded config for wallet: {config.wallet_name}/{config.wallet_hotkey}")
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return

    reward_mode = str(config.get("reward_mode", "capacitor")).strip().lower()
    if reward_mode not in {"capacitor", "capacitorless", "capacitorless_sticky"}:
        logging.warning("Unknown reward_mode '%s'; defaulting to 'capacitor'", reward_mode)
        reward_mode = "capacitor"
    logging.info("Reward mode: %s", reward_mode)
    is_capacitorless = reward_mode in {"capacitorless", "capacitorless_sticky"}
    cap_cfg = config.get("capacitorless", {}) if is_capacitorless else {}

    try:
        net = BittensorNetwork(config)
        wallet_to_use = net.wallet[0] if isinstance(net.wallet, list) else net.wallet
        my_hotkey = wallet_to_use.hotkey.ss58_address
        logging.info(f"Validator hotkey: {my_hotkey}")
    except Exception as e:
        logging.error(f"Failed to initialize Bittensor network: {e}")
        return

    contract_manager = None
    if not is_capacitorless:
        if ContractManager is None:
            logging.error("ContractManager import failed; cannot run in capacitor mode")
            return

        contract_config = config.get("contract", {})
        abi_file = contract_config.get("abi_file", "capacitor_abi.json")
        abi_path = Path(abi_file)

        try:
            contract_abi = json.loads(abi_path.read_text())
            logging.info(f"Loaded contract ABI from {abi_path}")
        except FileNotFoundError:
            logging.error(f"Contract ABI file not found: {abi_path}")
            return
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in ABI file: {e}")
            return

        try:
            contract_manager = ContractManager(
                rpc_url=contract_config.get("rpc_url", "https://test.chain.opentensor.ai"),
                contract_address=contract_config.get("address", "0xfakefake"),
                abi=contract_abi,
                bt_wallet=wallet_to_use,
                evm_key_path=config.get("evm_key_path", None),
            )
            logging.info(
                f"Initialized ContractManager for contract: {contract_config.get('address')}"
            )
        except Exception as e:
            logging.error(f"Failed to initialize ContractManager: {e}")
            return

    relay_config = config.get("relay", {})
    blacklist_config = config.get("blacklist", {})
    relay_url = relay_config.get("url")
    blacklist_cutoff = blacklist_config.get(
        "cutoff_percentage", 9999999999999999.0
    )

    if relay_url:
        try:
            relay_client = RelayClient(relay_url=relay_url, wallet=wallet_to_use)
            logging.info(f"Initialized RelayClient for {relay_url}")
            logging.info(f"Polling interval: {relay_config.get('poll_interval_seconds', 60)}s")
        except Exception as e:
            logging.error(f"Failed to initialize RelayClient: {e}")
            relay_client = None
    else:
        logging.warning("No relay URL configured - running without relay")
        relay_client = None

    if is_capacitorless and relay_client is None:
        logging.error("capacitorless mode requires relay.url configured and reachable")
        return

    try:
        if is_capacitorless:
            burn_hotkey = cap_cfg.get("burn_hotkey") or config.get("burn_hotkey")
            if not burn_hotkey:
                raise ValueError("capacitorless mode requires capacitorless.burn_hotkey")

            cap_mode = str(cap_cfg.get("mode", "sticky_burnsplit")).strip().lower()
            if reward_mode == "capacitorless_sticky":
                cap_mode = "sticky_burnsplit"

            if cap_mode in {"sticky", "sticky_burnsplit", "burnsplit"}:
                weight_manager = CapacitorlessStickyBurnSplitWeightManager(
                    net,
                    relay_client=relay_client,
                    burn_hotkey=burn_hotkey,
                    burn_share=float(cap_cfg.get("burn_share", 0.9)),
                    winner_share=cap_cfg.get("winner_share", None),
                    winner_source=str(cap_cfg.get("winner_source", "relay")),
                    min_winner_improvement=float(
                        cap_cfg.get("min_winner_improvement", 0.0)
                    ),
                    events_limit=int(cap_cfg.get("events_limit", 50)),
                    event_refresh_interval_s=int(
                        cap_cfg.get("event_refresh_interval_s", 60)
                    ),
                    metagraph_refresh_interval_s=int(
                        cap_cfg.get("metagraph_refresh_interval_s", 600)
                    ),
                    poll_interval_s=float(cap_cfg.get("poll_interval_s", 6.0)),
                    retry_interval_s=float(cap_cfg.get("retry_interval_s", 5.0)),
                )
                logging.info(
                    "Initialized CapacitorlessStickyBurnSplitWeightManager (burn=%.3f winner=%.3f)",
                    weight_manager.burn_share,
                    weight_manager.winner_share,
                )
            else:
                # Default alignment to the commit→reveal delay (360 blocks) so weight changes
                # only occur on the same cadence as reveals, unless explicitly configured.
                alignment_mod = int(cap_cfg.get("alignment_mod", 360))
                if int(getattr(config, "epoch_length", alignment_mod)) != alignment_mod:
                    logging.warning(
                        "capacitorless.alignment_mod differs from epoch_length; "
                        "on-chain weight rate limits may prevent exact boundary syncing"
                    )

                weight_manager = CapacitorlessWeightManager(
                    net,
                    relay_client=relay_client,
                    burn_hotkey=burn_hotkey,
                    alignment_mod=alignment_mod,
                    events_limit=int(cap_cfg.get("events_limit", 50)),
                    event_refresh_interval_s=int(
                        cap_cfg.get("event_refresh_interval_s", 60)
                    ),
                    metagraph_refresh_interval_s=int(
                        cap_cfg.get("metagraph_refresh_interval_s", 600)
                    ),
                    poll_interval_s=float(cap_cfg.get("poll_interval_s", 6.0)),
                )
                logging.info("Initialized CapacitorlessWeightManager")
        else:
            weight_manager = WeightManager(net)
            logging.info("Initialized WeightManager")
    except Exception as e:
        logging.error(f"Failed to initialize weight manager: {e}")
        return

    try:
        metrics_logger = ValidatorMetricsLogger("validator_metrics.log")
        metrics_logger.log_session_start()
        logging.info("Initialized MetricsLogger")
    except Exception as e:
        logging.warning(f"Failed to initialize MetricsLogger: {e}")
        metrics_logger = None

    submission_scheduler = SubmissionScheduler(config.get("submission_schedule", {}))

    submission_threshold_config = config.get("submission_threshold", {})
    threshold_mode = str(submission_threshold_config.get("mode", "sota_only")).lower()
    if threshold_mode not in {"sota_only", "local_best"}:
        logging.warning(
            "Unknown submission_threshold mode '%s'. Falling back to 'sota_only'.",
            threshold_mode,
        )
        threshold_mode = "sota_only"
    use_local_best_gate = threshold_mode == "local_best"
    if use_local_best_gate:
        logging.info("Local best submission gate enabled")

    local_best_state = {"score": None}
    local_best_lock = threading.Lock()

    pending_submission = {"result": None}
    pending_lock = threading.Lock()

    def _fallback_sota_threshold() -> float:
        try:
            return float(config.get("sota", {}).get("default_threshold", 0.0))
        except Exception:
            return 0.0

    def _store_pending_submission(candidate: dict, *, seen_block: int | None = None):
        candidate_copy = copy.deepcopy(candidate)
        candidate_copy["_pending_id"] = str(uuid.uuid4())
        candidate_copy["_cached_at"] = time.time()
        if seen_block is not None:
            candidate_copy["_seen_block"] = int(seen_block)

        miner_hotkey = candidate_copy.get("miner_hotkey", "")
        pending_score = candidate_copy.get("validator_score")

        with pending_lock:
            existing = pending_submission["result"]
            if existing:
                existing_score = existing.get("validator_score")
                if (
                    pending_score is not None
                    and existing_score is not None
                    and existing_score >= pending_score
                ):
                    logging.info(
                        "🕒 Pending submission already stored with score %.4f >= %.4f. Keeping existing candidate.",
                        existing_score,
                        pending_score,
                    )
                    return
                logging.info(
                    "🔁 Replacing pending submission %.4f → %.4f",
                    existing_score if existing_score is not None else 0.0,
                    pending_score if pending_score is not None else 0.0,
                )
            else:
                logging.info(
                    "💾 Caching pending submission for miner %s with score %.4f",
                    miner_hotkey[:8] if miner_hotkey else "unknown",
                    pending_score if pending_score is not None else 0.0,
                )
            pending_submission["result"] = candidate_copy

        _update_local_best(pending_score, reason="cached pending submission")

        next_allowed = submission_scheduler.get_next_allowed_time()
        if next_allowed:
            logging.info(
                "   Next allowed submission window: %sZ",
                next_allowed.isoformat(),
            )

    def _get_pending_submission():
        with pending_lock:
            if not pending_submission["result"]:
                return None
            return copy.deepcopy(pending_submission["result"])

    def _clear_pending_submission(candidate=None):
        identifier = None
        if isinstance(candidate, dict):
            identifier = candidate.get("_pending_id")
        with pending_lock:
            existing = pending_submission["result"]
            if not existing:
                return
            if identifier is None or existing.get("_pending_id") == identifier:
                pending_submission["result"] = None
                logging.info("🗑️  Cleared pending submission cache")

    def _get_local_best_score():
        if not use_local_best_gate:
            return None
        with local_best_lock:
            return local_best_state["score"]

    def _update_local_best(score, reason="update"):
        if not use_local_best_gate or score is None:
            return
        with local_best_lock:
            current_best = local_best_state["score"]
            if current_best is None or score > current_best:
                local_best_state["score"] = score
                if current_best is None:
                    logging.info("📈 Local best initialized at %.4f (%s)", score, reason)
                else:
                    logging.info(
                        "📈 Local best improved from %.4f to %.4f (%s)",
                        current_best,
                        score,
                        reason,
                    )

    def _submit_candidate(
        candidate: dict,
        sota_override=None,
        source="relay",
        skip_schedule_check=False,
        *,
        seen_block: int | None = None,
    ):
        miner_hotkey = candidate.get("miner_hotkey")
        validator_score = candidate.get("validator_score")
        if not miner_hotkey or validator_score is None:
            logging.warning("Pending submission missing required fields; discarding.")
            if source == "pending":
                _clear_pending_submission(candidate)
            return False

        if is_capacitorless:
            if relay_client is None:
                logging.error("No relay client configured; cannot submit SOTA vote")
                if source == "pending":
                    _clear_pending_submission(candidate)
                return False

            if seen_block is None:
                seen_block = candidate.get("_seen_block") or candidate.get("seen_block")
            try:
                seen_block_int = int(seen_block) if seen_block is not None else 0
            except Exception:
                seen_block_int = 0
            if seen_block_int <= 0:
                try:
                    with net.subtensor_lock:
                        seen_block_int = int(net.subtensor.get_current_block())
                except Exception:
                    seen_block_int = 0
            if seen_block_int <= 0:
                logging.error("Could not determine current block for SOTA vote")
                if source == "pending":
                    _clear_pending_submission(candidate)
                return False
        else:
            seen_block_int = 0

        reward_address = candidate.get("coldkey_address")
        if not is_capacitorless and not reward_address:
            logging.warning(
                "Candidate from miner %s lacks a coldkey address; skipping submission.",
                miner_hotkey[:8] if miner_hotkey else "unknown",
            )
            if source == "pending":
                _clear_pending_submission(candidate)
            return False

        try:
            if sota_override is not None:
                current_sota = float(sota_override)
            elif is_capacitorless:
                logging.info("Fetching relay SOTA threshold")
                current_sota = relay_client.get_sota_threshold()
                if current_sota is None:
                    current_sota = _fallback_sota_threshold()
            else:
                if contract_manager is None:
                    raise RuntimeError("Contract manager not configured")
                current_sota = contract_manager.get_current_sota_threshold(
                    force_refresh=(source == "pending")
                )
        except Exception as e:
            logging.error(f"❌ Could not get SOTA score before submission: {e}")
            return False

        if validator_score <= current_sota:
            logging.info(
                "⚠️  Candidate score %.4f not better than current SOTA %.4f. Skipping submission.",
                validator_score,
                current_sota,
            )
            if source == "pending":
                _clear_pending_submission(candidate)
            return False

        local_best_score = _get_local_best_score()
        if local_best_score is not None and validator_score < local_best_score:
            logging.info(
                "⛔  Candidate score %.4f below local best %.4f. Skipping submission.",
                validator_score,
                local_best_score,
            )
            if source == "pending":
                _clear_pending_submission(candidate)
            return False

        if not skip_schedule_check and not submission_scheduler.can_submit():
            logging.info(
                "⏸️  Submission schedule blocks vote for miner %s (score %.4f).",
                miner_hotkey[:8] if miner_hotkey else "unknown",
                validator_score,
            )
            if source != "pending":
                _store_pending_submission(candidate, seen_block=seen_block_int or None)
            return False

        try:
            result_id = candidate.get("id")

            if is_capacitorless:
                logging.info(
                    "📤 Submitting SOTA vote to relay for miner %s with score %.4f at block %d",
                    miner_hotkey[:8] + "...",
                    validator_score,
                    seen_block_int,
                )
                submission_start = time.time()
                resp = relay_client.submit_sota_vote(
                    miner_hotkey,
                    validator_score,
                    seen_block=seen_block_int,
                    result_id=result_id,
                )
                submission_time = time.time() - submission_start
                if not resp:
                    logging.error("❌ Relay SOTA vote failed (no response)")
                    return False

                logging.info(
                    "✅ Relay vote status: %s (%s/%s) in %.2fs",
                    resp.get("status"),
                    resp.get("votes_for_candidate"),
                    resp.get("votes_needed"),
                    submission_time,
                )
                finalized = resp.get("finalized_event")
                if finalized:
                    logging.info(
                        "🏁 Finalized event: start=%s end=%s",
                        finalized.get("start_block"),
                        finalized.get("end_block"),
                    )

                submission_scheduler.record_submission()
                if source == "pending":
                    _clear_pending_submission(candidate)
                _update_local_best(validator_score, reason="submitted to relay")

                if metrics_logger:
                    metrics_logger.log_miner_result(
                        miner_hotkey,
                        candidate.get("score", 0) or 0,
                        validator_score,
                        current_sota,
                        "passed",
                        pushed_sota=bool(finalized),
                    )

                return True

            if contract_manager is None:
                raise RuntimeError("Contract manager not configured; cannot submit vote")

            logging.info(
                "Using coldkey %s for miner %s payout.",
                reward_address[:8] + "...",
                miner_hotkey[:8] + "...",
            )
            recipient_bytes32 = contract_manager.ss58_to_bytes32(reward_address)
            scaled_score = int(validator_score * 10**18)
            if contract_manager._already_voted_for(recipient_bytes32, scaled_score):
                logging.info(
                    "ℹ️  Already voted for %s with score %.4f",
                    reward_address[:8],
                    validator_score,
                )
                if source == "pending":
                    _clear_pending_submission(candidate)
                return False

            if relay_client and result_id:
                try:
                    logging.info("Verifying relay result_id=%s", result_id)
                    if relay_client.verify_result(result_id):
                        logging.info(f"✓ Marked result {result_id} as verified on relay")
                except Exception as e:
                    logging.warning(f"Failed to verify result on relay: {e}")

            logging.info(
                "📤 Submitting vote to contract for coldkey %s (miner %s) with score %.4f",
                reward_address[:8] + "...",
                miner_hotkey[:8] + "...",
                validator_score,
            )
            submission_start = time.time()
            tx_hash = contract_manager.submit_contract_entry(
                recipient_ss58_address=reward_address,
                new_score=validator_score,
                verbose=False,
            )
            submission_time = time.time() - submission_start

            logging.info(f"✅ Vote submitted! TX: {tx_hash}")
            logging.info(f"   Miner: {miner_hotkey}")
            logging.info(f"   Coldkey: {reward_address}")
            logging.info(f"   Score: {validator_score:.4f}")
            logging.info(f"   Time: {submission_time:.2f}s")

            submission_scheduler.record_submission()
            if source == "pending":
                _clear_pending_submission(candidate)

            _update_local_best(validator_score, reason="submitted to contract")

            if metrics_logger:
                metrics_logger.log_contract_submission(
                    miner_hotkey, validator_score, tx_hash, submission_time
                )
                metrics_logger.log_miner_result(
                    miner_hotkey,
                    candidate.get("score", 0) or 0,
                    validator_score,
                    current_sota,
                    "passed",
                    pushed_sota=True,
                )

            return True

        except Exception as e:
            logging.error(f"❌ Failed to submit vote: {e}")
            traceback.print_exc()
            return False

    def _try_submit_pending(reason: str):
        pending = _get_pending_submission()
        if not pending:
            return False
        if not submission_scheduler.can_submit():
            return False
        logging.info(f"🚀 Attempting pending submission ({reason})")
        return _submit_candidate(
            pending,
            sota_override=None,
            source="pending",
            skip_schedule_check=True,
            seen_block=pending.get("_seen_block"),
        )

    def process_relay_results(results):
        """Callback function to process results from the relay."""
        _try_submit_pending("relay poll")
        if not results:
            logging.debug("No results from relay this round")
            return

        logging.info("=" * 60)
        logging.info(f"📥 Received {len(results)} results from relay")
        logging.info("=" * 60)
        evaluation_start_time = time.time()

        sota_score = None
        if relay_client:
            try:
                logging.info("Fetching relay SOTA threshold")
                sota_score = relay_client.get_sota_threshold()
                if sota_score is not None:
                    logging.info(f"Current SOTA threshold (from relay): {sota_score:.4f}")
            except Exception as e:
                logging.warning(f"Failed to get SOTA from relay: {e}")

        if sota_score is None and contract_manager is not None:
            try:
                sota_score = contract_manager.get_current_sota_threshold()
                logging.info(f"Current SOTA threshold (from contract): {sota_score:.4f}")
            except Exception as e:
                logging.warning(f"Failed to get SOTA from contract: {e}")

        if sota_score is None:
            sota_score = _fallback_sota_threshold()
            logging.warning(f"Using default SOTA threshold: {sota_score:.4f}")

        evaluated_results = []
        for result in results:
            miner_hotkey = result.get("miner_hotkey")
            miner_score = result.get("score")
            result_id = result.get("id")
            timestamp_message = result.get("timestamp_message")
            signature = result.get("signature")
            algorithm_result_str = result.get("algorithm_result")

            if not all(
                [
                    miner_hotkey,
                    miner_score,
                    result_id,
                    timestamp_message,
                    signature,
                    algorithm_result_str,
                ]
            ):
                logging.warning(f"Skipping invalid relay result (missing fields): {result}")
                continue

            if miner_hotkey not in net.metagraph.hotkeys:
                logging.warning(
                    f"❌ Miner {miner_hotkey[:8]} not registered on netuid {config.netuid}"
                )
                if metrics_logger:
                    metrics_logger.log_miner_result(
                        miner_hotkey, miner_score, 0, sota_score, "not_registered"
                    )
                continue

            if not ValidatorAuth.verify_miner_signature(
                miner_hotkey, timestamp_message, signature
            ):
                logging.warning(
                    f"❌ Signature verification failed for miner {miner_hotkey[:8]}"
                )
                if metrics_logger:
                    metrics_logger.log_miner_result(
                        miner_hotkey, miner_score, 0, sota_score, "failed_validation"
                    )
                continue

            try:
                result["algorithm_result"] = json.loads(algorithm_result_str)
            except json.JSONDecodeError:
                logging.warning(
                    f"Could not deserialize algorithm_result for miner {miner_hotkey[:8]}. Skipping."
                )
                continue

            eval_start = time.time()
            is_valid, validator_score = verify_solution_quality(
                result["algorithm_result"], sota_score
            )
            eval_duration_s = time.time() - eval_start
            logging.info(
                "Evaluation complete for result_id=%s miner=%s (%.3fs)",
                result_id,
                miner_hotkey[:8],
                eval_duration_s,
            )

            logging.info(
                f"Miner {miner_hotkey[:8]}: "
                f"Miner Score = {miner_score:.4f}, "
                f"Validator Score = {validator_score:.4f}, "
                f"SOTA = {sota_score:.4f}"
            )

            if not is_valid:
                logging.warning(
                    f"❌ Miner {miner_hotkey[:8]} score {validator_score:.4f} not above SOTA {sota_score:.4f}"
                )
                if metrics_logger:
                    metrics_logger.log_miner_result(
                        miner_hotkey,
                        miner_score,
                        validator_score,
                        sota_score,
                        "failed_sota",
                    )
                continue

            if abs(validator_score - miner_score) > blacklist_cutoff:
                logging.warning(
                    f"⚠️  Blacklisting miner {miner_hotkey[:8]} - score delta too large. "
                    f"Validator: {validator_score:.4f}, Miner claimed: {miner_score:.4f}"
                )
                if relay_client:
                    try:
                        logging.info(
                            "Submitting blacklist request to relay for miner %s",
                            miner_hotkey[:8],
                        )
                        relay_client.blacklist_miner(miner_hotkey)
                    except Exception as e:
                        logging.error(f"Failed to blacklist miner on relay: {e}")
                if metrics_logger:
                    metrics_logger.log_miner_result(
                        miner_hotkey,
                        miner_score,
                        validator_score,
                        sota_score,
                        "blacklisted",
                    )
                continue

            evaluated_results.append({"validator_score": validator_score, **result})
            logging.info(f"✓ Miner {miner_hotkey[:8]} passed validation")

            if metrics_logger:
                metrics_logger.log_miner_result(
                    miner_hotkey, miner_score, validator_score, sota_score, "passed"
                )

        if metrics_logger:
            metrics_logger.log_evaluation_batch(len(results), sota_score, evaluation_start_time)

        if not evaluated_results:
            logging.info("No results passed validation and SOTA checks.")
            return

        best_result = max(evaluated_results, key=lambda x: x["validator_score"])
        highest_validator_score = best_result["validator_score"]
        miner_hotkey = best_result.get("miner_hotkey")

        logging.info(
            f"🏆 Best submission: Miner {miner_hotkey[:8]} with score {highest_validator_score:.4f}"
        )

        if highest_validator_score <= sota_score:
            logging.info(
                f"⚠️  Best score {highest_validator_score:.4f} not better than SOTA {sota_score:.4f}. Not voting."
            )
            return

        _update_local_best(highest_validator_score, reason="best relay evaluation")

        # In capacitorless mode, optionally drive weights directly from local evaluation,
        # without waiting for relay SOTA finalization.
        if (
            is_capacitorless
            and miner_hotkey
            and str(cap_cfg.get("winner_source", "relay")).strip().lower() == "local"
            and hasattr(weight_manager, "update_local_winner")
        ):
            try:
                changed = weight_manager.update_local_winner(
                    miner_hotkey, highest_validator_score
                )
                if (
                    changed
                    and bool(cap_cfg.get("apply_weights_inline", True))
                    and hasattr(weight_manager, "apply_once")
                ):
                    weight_manager.apply_once(force=True)
            except Exception as e:
                logging.warning("Failed to update local winner in weight manager: %s", e)

        seen_block = None
        if is_capacitorless:
            try:
                with net.subtensor_lock:
                    seen_block = int(net.subtensor.get_current_block())
            except Exception:
                seen_block = None

        # If configured, skip posting SOTA votes to the relay entirely (local-only mode).
        if is_capacitorless and not bool(cap_cfg.get("submit_sota_votes", True)):
            logging.info("Skipping relay SOTA vote (capacitorless.submit_sota_votes=false)")
            return

        if not _submit_candidate(
            best_result,
            sota_override=sota_score,
            source="relay",
            seen_block=seen_block,
        ):
            logging.info("Submission deferred or failed; result may remain cached for later.")

    def process_test_submissions(submissions):
        """Evaluate relay test submissions for logging only (no votes/weights)."""
        if not submissions:
            return

        logging.info("=" * 60)
        logging.info(f"🧪 Received {len(submissions)} TEST submissions from relay")
        logging.info("=" * 60)

        sota_score = None
        if relay_client:
            try:
                sota_score = relay_client.get_sota_threshold()
            except Exception:
                sota_score = None

        if sota_score is None and contract_manager is not None:
            try:
                sota_score = contract_manager.get_current_sota_threshold()
            except Exception:
                sota_score = None

        if sota_score is None:
            sota_score = _fallback_sota_threshold()

        for sub in submissions:
            submission_id = sub.get("id") or sub.get("test_submission_id") or "unknown"
            task_id = sub.get("task_id") or "unknown"
            submitter_hotkey = sub.get("submitter_hotkey")
            claimed_score = sub.get("score")
            algorithm_result_str = sub.get("algorithm_result")

            if not algorithm_result_str:
                logging.warning(
                    "🧪 Skipping TEST submission %s (missing algorithm_result)",
                    submission_id,
                )
                continue

            try:
                algorithm_result = (
                    algorithm_result_str
                    if isinstance(algorithm_result_str, dict)
                    else json.loads(algorithm_result_str)
                )
            except Exception as e:
                logging.warning(
                    "🧪 Skipping TEST submission %s (invalid algorithm_result JSON): %s",
                    submission_id,
                    e,
                )
                continue

            eval_start = time.time()
            is_valid, validator_score = verify_solution_quality(algorithm_result, sota_score)
            eval_duration_s = time.time() - eval_start

            logging.info(
                "🧪 TEST submission id=%s task=%s valid=%s score=%.4f claimed=%s sota=%.4f (%.3fs)",
                str(submission_id)[:16],
                task_id,
                is_valid,
                float(validator_score),
                claimed_score,
                float(sota_score),
                eval_duration_s,
            )

            if metrics_logger:
                try:
                    metrics_logger.log_test_submission(
                        str(submission_id),
                        str(task_id),
                        float(claimed_score) if claimed_score is not None else None,
                        float(validator_score),
                        float(sota_score),
                        bool(is_valid),
                        submitter_hotkey=str(submitter_hotkey) if submitter_hotkey else None,
                    )
                except Exception:
                    pass

    if relay_client:
        try:
            relay_poller = RelayPoller(
                relay_client=relay_client,
                interval=relay_config.get("poll_interval_seconds", 60),
                on_new_results=process_relay_results,
            )
            logging.info("Initialized RelayPoller")
        except Exception as e:
            logging.error(f"Failed to initialize RelayPoller: {e}")
            relay_poller = None
    else:
        relay_poller = None

    test_poller = None
    if accept_test:
        if relay_client is None:
            logging.warning(
                "--accept-test enabled but relay.url is not configured; test polling disabled"
            )
        else:
            try:
                test_poller = RelayPoller(
                    relay_client=relay_client,
                    interval=int(
                        relay_config.get(
                            "test_poll_interval_seconds",
                            relay_config.get("poll_interval_seconds", 60),
                        )
                    ),
                    on_new_results=process_test_submissions,
                    fetch_fn=lambda: relay_client.get_test_submissions(
                        limit=int(relay_config.get("test_poll_limit", 50))
                    ),
                )
                logging.info("Initialized TestSubmission poller")
            except Exception as e:
                logging.error(f"Failed to initialize TestSubmission poller: {e}")
                test_poller = None

    logging.info("=" * 60)
    logging.info("Starting background workers...")
    logging.info("=" * 60)

    try:
        weight_manager.start_background_worker()
        logging.info("✓ WeightManager worker started")
    except Exception as e:
        logging.error(f"Failed to start WeightManager: {e}")

    if relay_poller:
        try:
            relay_poller.start()
            logging.info("✓ RelayPoller started")
        except Exception as e:
            logging.error(f"Failed to start RelayPoller: {e}")
            relay_poller = None

    if test_poller:
        try:
            test_poller.start()
            logging.info("✓ TestSubmission poller started")
        except Exception as e:
            logging.error(f"Failed to start TestSubmission poller: {e}")
            test_poller = None

    logging.info("=" * 60)
    logging.info("✅ Validator node is running")
    logging.info("   Press Ctrl+C to exit")
    logging.info("=" * 60)

    try:
        cycle_count = 0
        while True:
            time.sleep(60)
            cycle_count += 1
            _try_submit_pending("periodic tick")

            if cycle_count % 10 == 0:
                if metrics_logger:
                    metrics_logger.log_periodic_summary()
                logging.info(f"⏱️  Uptime: {cycle_count} minutes")

            if (
                weight_manager.background_thread
                and not weight_manager.background_thread.is_alive()
            ):
                logging.warning("⚠️  Weight manager thread died. Restarting...")
                try:
                    weight_manager.start_background_worker()
                    logging.info("✓ Weight manager restarted")
                except Exception as e:
                    logging.error(f"Failed to restart weight manager: {e}")

            if (
                relay_poller
                and relay_poller.background_thread
                and not relay_poller.background_thread.is_alive()
            ):
                logging.warning("⚠️  Relay poller thread died. Restarting...")
                try:
                    relay_poller.start()
                    logging.info("✓ Relay poller restarted")
                except Exception as e:
                    logging.error(f"Failed to restart relay poller: {e}")

            if (
                test_poller
                and test_poller.background_thread
                and not test_poller.background_thread.is_alive()
            ):
                logging.warning("⚠️  TestSubmission poller thread died. Restarting...")
                try:
                    test_poller.start()
                    logging.info("✓ TestSubmission poller restarted")
                except Exception as e:
                    logging.error(f"Failed to restart test submission poller: {e}")

    except KeyboardInterrupt:
        logging.info("\n" + "=" * 60)
        logging.info("Shutting down validator client...")
        logging.info("=" * 60)
        if metrics_logger:
            metrics_logger.log_session_end()
        if relay_poller:
            relay_poller.stop()
        if test_poller:
            test_poller.stop()
        logging.info("✓ Shutdown complete")


if __name__ == "__main__":
    main()

import time
import logging
import json
import copy
import threading
import uuid
from pathlib import Path
import traceback
from bittensor_network import BittensorNetwork
from bittensor_network.bittensor_config import BittensorConfig
from validator.contract_manager import ContractManager
from validator.weight_manager import WeightManager
from validator.relay_client import RelayClient
from validator.relay_poller import RelayPoller
from validator.auth import ValidatorAuth
from core.evaluations import verify_solution_quality
from validator.metrics_logger import ValidatorMetricsLogger
from validator.submission_scheduler import SubmissionScheduler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main function for the validator client.
    Initializes all necessary components and starts the background services.
    """
    logging.info("="*60)
    logging.info("Starting Validator Node")
    logging.info("="*60)

    try:
        config = BittensorConfig.get_bittensor_config()
        logging.info(f"Loaded config for wallet: {config.wallet_name}/{config.wallet_hotkey}")
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return

    try:
        net = BittensorNetwork(config)
        wallet_to_use = net.wallet[0] if isinstance(net.wallet, list) else net.wallet
        my_hotkey = wallet_to_use.hotkey.ss58_address
        logging.info(f"Validator hotkey: {my_hotkey}")
    except Exception as e:
        logging.error(f"Failed to initialize Bittensor network: {e}")
        return

    # Load contract ABI from file
    contract_config = config.get("contract", {})
    abi_file = contract_config.get("abi_file", "capacitor_abi.json")
    abi_path = Path(abi_file)

    try:
        with open(abi_path, "r") as f:
            contract_abi = json.load(f)
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
            evm_key_path=config.get("evm_key_path", None)
        )
        logging.info(f"Initialized ContractManager for contract: {contract_config.get('address')}")
    except Exception as e:
        logging.error(f"Failed to initialize ContractManager: {e}")
        return

    # Setup WeightManager
    try:
        weight_manager = WeightManager(net)
        logging.info("Initialized WeightManager")
    except Exception as e:
        logging.error(f"Failed to initialize WeightManager: {e}")
        return

    # Setup Relay Poller
    relay_config = config.get("relay", {})
    blacklist_config = config.get("blacklist", {})
    relay_url = relay_config.get("url")
    blacklist_cutoff = blacklist_config.get("cutoff_percentage", 9999999999999999.0)

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

    # Initialize metrics logger
    try:
        metrics_logger = ValidatorMetricsLogger("validator_metrics.log")
        metrics_logger.log_session_start()
        logging.info("Initialized MetricsLogger")
    except Exception as e:
        logging.warning(f"Failed to initialize MetricsLogger: {e}")
        metrics_logger = None

    submission_scheduler = SubmissionScheduler(config.get("submission_schedule", {}))

    submission_threshold_config = config.get("submission_threshold", {})
    threshold_mode = str(
        submission_threshold_config.get("mode", "sota_only")
    ).lower()
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

    def _store_pending_submission(candidate: dict):
        candidate_copy = copy.deepcopy(candidate)
        candidate_copy["_pending_id"] = str(uuid.uuid4())
        candidate_copy["_cached_at"] = time.time()
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

    def _submit_candidate(candidate: dict, sota_override=None, source="relay", skip_schedule_check=False):
        miner_hotkey = candidate.get("miner_hotkey")
        validator_score = candidate.get("validator_score")
        if not miner_hotkey or validator_score is None:
            logging.warning("Pending submission missing required fields; discarding.")
            if source == "pending":
                _clear_pending_submission(candidate)
            return False

        reward_address = candidate.get("coldkey_address")
        if not reward_address:
            logging.warning(
                "Candidate from miner %s lacks a coldkey address; skipping submission.",
                miner_hotkey[:8] if miner_hotkey else "unknown",
            )
            if source == "pending":
                _clear_pending_submission(candidate)
            return False

        try:
            current_sota = (
                sota_override
                if sota_override is not None
                else contract_manager.get_current_sota_threshold(
                    force_refresh=(source == "pending")
                )
            )
        except Exception as e:
            logging.error(f"❌ Could not get SOTA score before submission: {e}")
            return False

        if validator_score <= current_sota:
            logging.info(
                f"⚠️  Candidate score {validator_score:.4f} not better than current SOTA {current_sota:.4f}. Skipping submission."
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
                _store_pending_submission(candidate)
            return False

        try:
            logging.info(
                "Using coldkey %s for miner %s payout.",
                reward_address[:8] + "...",
                miner_hotkey[:8] + "...",
            )
            recipient_bytes32 = contract_manager.ss58_to_bytes32(reward_address)
            scaled_score = int(validator_score * 10**18)
            if contract_manager._already_voted_for(recipient_bytes32, scaled_score):
                logging.info(
                    f"ℹ️  Already voted for {reward_address[:8]} with score {validator_score:.4f}"
                )
                if source == "pending":
                    _clear_pending_submission(candidate)
                return False

            result_id = candidate.get("id")
            if relay_client and result_id:
                try:
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
                    candidate.get("score", 0),
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
            pending, sota_override=None, source="pending", skip_schedule_check=True
        )

    def process_relay_results(results):
        """Callback function to process results from the relay."""
        _try_submit_pending("relay poll")
        if not results:
            logging.debug("No results from relay this round")
            return

        logging.info("="*60)
        logging.info(f"📥 Received {len(results)} results from relay")
        logging.info("="*60)
        evaluation_start_time = time.time()

        sota_score = None
        if relay_client:
            try:
                sota_score = relay_client.get_sota_threshold()
                if sota_score is not None:
                    logging.info(f"Current SOTA threshold (from relay): {sota_score:.4f}")
            except Exception as e:
                logging.warning(f"Failed to get SOTA from relay: {e}")

        if sota_score is None:
            try:
                sota_score = contract_manager.get_current_sota_threshold()
                logging.info(f"Current SOTA threshold (from contract): {sota_score:.4f}")
            except Exception as e:
                logging.error(f"❌ Could not get SOTA score: {e}")
                return

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
                logging.warning(
                    f"Skipping invalid relay result (missing fields): {result}"
                )
                continue

            # Verify miner signature using the correct timestamp message
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

            # Deserialize algorithm_result and update the result dict
            try:
                result["algorithm_result"] = json.loads(algorithm_result_str)
            except json.JSONDecodeError:
                logging.warning(
                    f"Could not deserialize algorithm_result for miner {miner_hotkey[:8]}. Skipping."
                )
                continue

            # Re-evaluate the solution to get a trusted validator score
            is_valid, validator_score = verify_solution_quality(
                result["algorithm_result"], sota_score
            )

            logging.info(
                f"Miner {miner_hotkey[:8]}: "
                f"Miner Score = {miner_score:.4f}, "
                f"Validator Score = {validator_score:.4f}, "
                f"SOTA = {sota_score:.4f}"
            )

            # Check if the validator's score is above SOTA
            if not is_valid:
                logging.warning(
                    f"❌ Miner {miner_hotkey[:8]} score {validator_score:.4f} not above SOTA {sota_score:.4f}"
                )
                if metrics_logger:
                    metrics_logger.log_miner_result(
                        miner_hotkey, miner_score, validator_score, sota_score, "failed_sota"
                    )
                continue

            # Check for blacklisting based on score delta
            if abs(validator_score - miner_score) > blacklist_cutoff:
                logging.warning(
                    f"⚠️  Blacklisting miner {miner_hotkey[:8]} - score delta too large. "
                    f"Validator: {validator_score:.4f}, Miner claimed: {miner_score:.4f}"
                )
                if relay_client:
                    try:
                        relay_client.blacklist_miner(miner_hotkey)
                    except Exception as e:
                        logging.error(f"Failed to blacklist miner on relay: {e}")
                if metrics_logger:
                    metrics_logger.log_miner_result(
                        miner_hotkey, miner_score, validator_score, sota_score, "blacklisted"
                    )
                continue

            # Add the validated result to our list for final selection
            evaluated_results.append({"validator_score": validator_score, **result})
            logging.info(f"✓ Miner {miner_hotkey[:8]} passed validation")

            # Log successful validation
            if metrics_logger:
                metrics_logger.log_miner_result(
                    miner_hotkey, miner_score, validator_score, sota_score, "passed"
                )

        # Log evaluation batch metrics
        if metrics_logger:
            metrics_logger.log_evaluation_batch(len(results), sota_score, evaluation_start_time)
        
        if not evaluated_results:
            logging.info("No results passed validation and SOTA checks.")
            return

        # Get the best result based on the validator's evaluation
        best_result = max(evaluated_results, key=lambda x: x["validator_score"])
        highest_validator_score = best_result["validator_score"]
        miner_hotkey = best_result.get("miner_hotkey")

        logging.info(f"🏆 Best submission: Miner {miner_hotkey[:8]} with score {highest_validator_score:.4f}")

        # Final check: ensure our evaluated best score is still above SOTA
        if highest_validator_score <= sota_score:
            logging.info(
                f"⚠️  Best score {highest_validator_score:.4f} not better than SOTA {sota_score:.4f}. Not voting."
            )
            return

        _update_local_best(highest_validator_score, reason="best relay evaluation")

        # Process the single best result
        if not _submit_candidate(best_result, sota_override=sota_score, source="relay"):
            logging.info("Submission deferred or failed; result may remain cached for later.")

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

    logging.info("="*60)
    logging.info("Starting background workers...")
    logging.info("="*60)

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

    # Keep the main thread alive
    logging.info("="*60)
    logging.info("✅ Validator node is running")
    logging.info("   Press Ctrl+C to exit")
    logging.info("="*60)

    try:
        cycle_count = 0
        while True:
            time.sleep(60)
            cycle_count += 1
            _try_submit_pending("periodic tick")

            # Log periodic summary every 10 minutes
            if cycle_count % 10 == 0:
                if metrics_logger:
                    metrics_logger.log_periodic_summary()
                logging.info(f"⏱️  Uptime: {cycle_count} minutes")

            # Periodically check the status of background threads
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

    except KeyboardInterrupt:
        logging.info("\n" + "="*60)
        logging.info("Shutting down validator client...")
        logging.info("="*60)
        if metrics_logger:
            metrics_logger.log_session_end()
        if relay_poller:
            relay_poller.stop()
        logging.info("✓ Shutdown complete")


if __name__ == "__main__":
    main()

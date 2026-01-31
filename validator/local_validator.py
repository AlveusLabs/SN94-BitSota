from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import bittensor as bt

from core.evaluations import verify_solution_quality
from validator.auth import ValidatorAuth
from validator.metrics_logger import ValidatorMetricsLogger
from validator.relay_client import RelayClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalValidatorSettings:
    relay_url: str
    wallet_name: str
    wallet_hotkey: str
    wallet_path: Optional[str] = None
    poll_interval_seconds: float = 15.0
    results_limit: int = 256
    task_id: Optional[str] = None
    blacklist_cutoff: float = 0.1
    seen_block: int = 1
    seen_block_mode: str = "time"
    log_level: str = "INFO"
    relay_client_log_level: str = "WARNING"
    stats_interval_seconds: float = 30.0
    metrics_log: str = "local_validator_metrics.log"


@dataclass
class LocalValidatorStats:
    polls: int = 0
    results_fetched: int = 0
    results_new: int = 0
    results_duplicate: int = 0
    results_invalid: int = 0
    signature_failed: int = 0
    algorithm_parse_failed: int = 0
    eval_errors: int = 0
    failed_sota: int = 0
    blacklisted: int = 0
    passed: int = 0
    votes_attempted: int = 0
    votes_submitted: int = 0
    votes_finalized: int = 0
    last_sota_threshold: Optional[float] = None


class LocalRelayValidator:
    """
    Minimal validator runner intended for local simulation.

    - Polls relay `/results`
    - Verifies miner signatures
    - Re-evaluates submissions via `core.evaluations.verify_solution_quality`
    - Submits capacitorless relay votes via `/sota/vote`
    """

    def __init__(
        self,
        relay_client: RelayClient,
        *,
        blacklist_cutoff: float = 0.1,
        results_limit: int = 256,
        task_id: Optional[str] = None,
        seen_block_fn: Optional[Callable[[], int]] = None,
        verify_signature_fn: Callable[[str, str, str], bool] = ValidatorAuth.verify_miner_signature,
        verify_solution_fn: Callable[[Dict[str, Any], float], Tuple[bool, float]] = verify_solution_quality,
        metrics_logger: Optional[ValidatorMetricsLogger] = None,
    ) -> None:
        self.relay_client = relay_client
        self.blacklist_cutoff = float(blacklist_cutoff)
        self.results_limit = max(1, int(results_limit))
        self.task_id = task_id
        self.verify_signature_fn = verify_signature_fn
        self.verify_solution_fn = verify_solution_fn
        self.seen_block_fn = seen_block_fn or (lambda: max(1, int(time.time())))
        self._seen: set[Tuple[str, str]] = set()
        self.stats = LocalValidatorStats()
        self.metrics_logger = metrics_logger

    def stats_summary(self) -> str:
        unique_seen = len(self._seen)
        parts = [
            f"polls={self.stats.polls}",
            f"fetched={self.stats.results_fetched}",
            f"new={self.stats.results_new}",
            f"dupe={self.stats.results_duplicate}",
            f"invalid={self.stats.results_invalid}",
            f"sig_fail={self.stats.signature_failed}",
            f"parse_fail={self.stats.algorithm_parse_failed}",
            f"eval_err={self.stats.eval_errors}",
            f"failed_sota={self.stats.failed_sota}",
            f"blacklisted={self.stats.blacklisted}",
            f"passed={self.stats.passed}",
            f"votes={self.stats.votes_submitted}/{self.stats.votes_attempted}",
            f"finalized={self.stats.votes_finalized}",
            f"unique_seen={unique_seen}",
        ]
        if self.stats.last_sota_threshold is not None:
            parts.append(f"sota={self.stats.last_sota_threshold:.6f}")
        return " ".join(parts)

    def _result_key(self, result: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        miner_hotkey = (result.get("miner_hotkey") or "").strip()
        timestamp_message = (result.get("timestamp_message") or "").strip()
        if not miner_hotkey or not timestamp_message:
            return None
        return miner_hotkey, timestamp_message

    def _parse_algorithm_result(self, raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                return None
            return data if isinstance(data, dict) else None
        return None

    def poll_once(self) -> None:
        results = self.relay_client.get_all_results(
            limit=self.results_limit, task_id=self.task_id
        )
        self.stats.polls += 1
        if not results:
            return
        if not isinstance(results, list):
            logger.warning("Unexpected relay /results payload type=%s", type(results).__name__)
            return
        self.stats.results_fetched += len(results)
        self.process_results(results)

    def process_results(self, results: Sequence[Dict[str, Any]]) -> None:
        evaluation_start_time = time.time()
        try:
            sota_score = self.relay_client.get_sota_threshold()
            sota_threshold = float(sota_score) if sota_score is not None else 0.0
        except Exception:
            sota_threshold = 0.0
        self.stats.last_sota_threshold = float(sota_threshold)

        evaluated: list[Dict[str, Any]] = []
        for result in results:
            key = self._result_key(result)
            if key is None:
                self.stats.results_invalid += 1
                continue
            if key in self._seen:
                self.stats.results_duplicate += 1
                continue

            miner_hotkey = key[0]
            signature = result.get("signature")
            timestamp_message = key[1]
            miner_score = result.get("score")
            algorithm_raw = result.get("algorithm_result")

            if signature is None or miner_score is None or algorithm_raw is None:
                self.stats.results_invalid += 1
                continue

            if not self.verify_signature_fn(miner_hotkey, timestamp_message, signature):
                self._seen.add(key)
                self.stats.signature_failed += 1
                if self.metrics_logger:
                    try:
                        miner_score_f = float(miner_score)
                    except Exception:
                        miner_score_f = 0.0
                    self.metrics_logger.log_miner_result(
                        miner_hotkey,
                        miner_score_f,
                        0.0,
                        float(sota_threshold),
                        "failed_validation",
                    )
                continue

            algorithm_result = self._parse_algorithm_result(algorithm_raw)
            if algorithm_result is None:
                self._seen.add(key)
                self.stats.algorithm_parse_failed += 1
                if self.metrics_logger:
                    try:
                        miner_score_f = float(miner_score)
                    except Exception:
                        miner_score_f = 0.0
                    self.metrics_logger.log_miner_result(
                        miner_hotkey,
                        miner_score_f,
                        0.0,
                        float(sota_threshold),
                        "failed_validation",
                    )
                continue

            try:
                is_valid, validator_score = self.verify_solution_fn(
                    algorithm_result, sota_threshold
                )
            except Exception:
                logger.exception("Validator evaluation crashed miner=%s", miner_hotkey[:8])
                self.stats.eval_errors += 1
                continue
            self._seen.add(key)
            self.stats.results_new += 1

            try:
                miner_score_f = float(miner_score)
            except Exception:
                self.stats.results_invalid += 1
                continue

            if not is_valid:
                self.stats.failed_sota += 1
                if self.metrics_logger:
                    self.metrics_logger.log_miner_result(
                        miner_hotkey,
                        miner_score_f,
                        float(validator_score),
                        float(sota_threshold),
                        "failed_sota",
                    )
                continue

            if abs(float(validator_score) - miner_score_f) > self.blacklist_cutoff:
                self.stats.blacklisted += 1
                try:
                    self.relay_client.blacklist_miner(miner_hotkey)
                except Exception:
                    logger.exception("Failed to post blacklist vote miner=%s", miner_hotkey[:8])
                if self.metrics_logger:
                    self.metrics_logger.log_miner_result(
                        miner_hotkey,
                        miner_score_f,
                        float(validator_score),
                        float(sota_threshold),
                        "blacklisted",
                    )
                continue

            enriched = dict(result)
            enriched["algorithm_result"] = algorithm_result
            enriched["validator_score"] = float(validator_score)
            evaluated.append(enriched)
            self.stats.passed += 1
            if self.metrics_logger:
                self.metrics_logger.log_miner_result(
                    miner_hotkey,
                    miner_score_f,
                    float(validator_score),
                    float(sota_threshold),
                    "passed",
                )

        if not evaluated:
            if self.metrics_logger:
                self.metrics_logger.log_evaluation_batch(
                    num_results=len(results),
                    sota_threshold=float(sota_threshold),
                    evaluation_start_time=evaluation_start_time,
                )
            return

        best = max(evaluated, key=lambda r: float(r.get("validator_score", float("-inf"))))
        best_score = float(best.get("validator_score", float("-inf")))
        if best_score <= sota_threshold:
            if self.metrics_logger:
                self.metrics_logger.log_evaluation_batch(
                    num_results=len(results),
                    sota_threshold=float(sota_threshold),
                    evaluation_start_time=evaluation_start_time,
                )
            return

        miner_hotkey = best.get("miner_hotkey")
        if not miner_hotkey:
            if self.metrics_logger:
                self.metrics_logger.log_evaluation_batch(
                    num_results=len(results),
                    sota_threshold=float(sota_threshold),
                    evaluation_start_time=evaluation_start_time,
                )
            return

        seen_block = int(self.seen_block_fn())
        if seen_block <= 0:
            seen_block = 1

        self.stats.votes_attempted += 1
        try:
            resp = self.relay_client.submit_sota_vote(
                miner_hotkey=str(miner_hotkey),
                score=float(best_score),
                seen_block=seen_block,
                result_id=best.get("id"),
            )
            if resp is not None:
                self.stats.votes_submitted += 1
                is_finalized = bool(resp.get("finalized_event")) or (
                    str(resp.get("status", "")).strip().lower() == "finalized"
                )
                if is_finalized:
                    self.stats.votes_finalized += 1
                    if self.metrics_logger:
                        self.metrics_logger.log_sota_update(
                            old_sota=float(sota_threshold),
                            new_sota=float(best_score),
                            miner_hotkey=str(miner_hotkey),
                        )
        except Exception:
            logger.exception("Failed to submit relay SOTA vote miner=%s", str(miner_hotkey)[:8])
        finally:
            if self.metrics_logger:
                self.metrics_logger.log_evaluation_batch(
                    num_results=len(results),
                    sota_threshold=float(sota_threshold),
                    evaluation_start_time=evaluation_start_time,
                )


def _build_wallet(settings: LocalValidatorSettings) -> "bt.wallet":
    kwargs: Dict[str, Any] = {"name": settings.wallet_name, "hotkey": settings.wallet_hotkey}
    if settings.wallet_path:
        kwargs["path"] = settings.wallet_path
    return bt.wallet(**kwargs)


def _configure_logging(*, log_level: str, relay_client_log_level: str) -> None:
    resolved_level = getattr(logging, str(log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    resolved_relay_level = getattr(
        logging, str(relay_client_log_level).upper(), logging.WARNING
    )
    logging.getLogger("validator.relay_client").setLevel(resolved_relay_level)


def _parse_args(argv: Optional[Sequence[str]] = None) -> LocalValidatorSettings:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--relay-url", default="http://127.0.0.1:8002")
    parser.add_argument("--wallet-name", required=True)
    parser.add_argument("--wallet-hotkey", required=True)
    parser.add_argument("--wallet-path", default=None)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--blacklist-cutoff", type=float, default=0.1)
    parser.add_argument(
        "--seen-block-mode",
        choices=("time", "fixed"),
        default="time",
        help="Value to send as `seen_block` when voting (relay requires >0).",
    )
    parser.add_argument(
        "--seen-block",
        type=int,
        default=1,
        help="Used when --seen-block-mode=fixed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Root log level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--relay-client-log-level",
        default="WARNING",
        help="Log level for validator.relay_client (reduces noisy HTTP polling logs).",
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=30.0,
        help="Seconds between periodic stats summaries (0 disables).",
    )
    parser.add_argument(
        "--metrics-log",
        default="local_validator_metrics.log",
        help="Write JSONL metrics here (empty disables).",
    )
    args = parser.parse_args(argv)
    return LocalValidatorSettings(
        relay_url=str(args.relay_url),
        wallet_name=str(args.wallet_name),
        wallet_hotkey=str(args.wallet_hotkey),
        wallet_path=str(args.wallet_path) if args.wallet_path else None,
        poll_interval_seconds=float(args.poll_interval),
        results_limit=int(args.limit),
        task_id=str(args.task_id) if args.task_id else None,
        blacklist_cutoff=float(args.blacklist_cutoff),
        seen_block=max(1, int(args.seen_block)),
        seen_block_mode=str(args.seen_block_mode),
        log_level=str(args.log_level),
        relay_client_log_level=str(args.relay_client_log_level),
        stats_interval_seconds=float(args.stats_interval),
        metrics_log=str(args.metrics_log),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    settings = _parse_args(argv)
    _configure_logging(
        log_level=settings.log_level,
        relay_client_log_level=settings.relay_client_log_level,
    )

    wallet = _build_wallet(settings)
    relay_client = RelayClient(relay_url=settings.relay_url, wallet=wallet)

    if settings.seen_block_mode == "fixed":
        seen_block_fn = lambda: max(1, int(settings.seen_block))
    else:
        seen_block_fn = lambda: max(1, int(time.time()))

    metrics_logger: Optional[ValidatorMetricsLogger] = None
    if settings.metrics_log.strip():
        metrics_logger = ValidatorMetricsLogger(settings.metrics_log.strip())
        metrics_logger.log_session_start()

    runner = LocalRelayValidator(
        relay_client,
        blacklist_cutoff=settings.blacklist_cutoff,
        results_limit=settings.results_limit,
        task_id=settings.task_id,
        seen_block_fn=seen_block_fn,
        metrics_logger=metrics_logger,
    )

    logger.info(
        "Local validator started relay=%s hotkey=%s poll=%.1fs",
        settings.relay_url,
        wallet.hotkey.ss58_address,
        float(settings.poll_interval_seconds),
    )

    try:
        stats_interval_s = max(0.0, float(settings.stats_interval_seconds))
        last_stats_ts = 0.0
        while True:
            runner.poll_once()
            now = time.time()
            if stats_interval_s and (now - last_stats_ts) >= stats_interval_s:
                logger.info(runner.stats_summary())
                last_stats_ts = now
            time.sleep(max(0.5, float(settings.poll_interval_seconds)))
    except KeyboardInterrupt:
        logger.info("Local validator stopped (%s)", runner.stats_summary())


if __name__ == "__main__":  # pragma: no cover
    main()

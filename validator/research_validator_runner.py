from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

from miner.wallet_inputs import EphemeralWallet, load_wallet
from validator.public_replay import PublicReplayEngine
from validator.research_validator_client import (
    AutoresearchValidatorClient,
    DEFAULT_RESEARCH_COORDINATOR_URL,
    DEFAULT_VALIDATOR_WORKLIST_PATH,
)


@dataclass(frozen=True, slots=True)
class PublicValidatorRunnerConfig:
    coordinator_url: str = DEFAULT_RESEARCH_COORDINATOR_URL
    workspace_root: Path = Path(".bitsota_public_validator_workspaces")
    task_id: str | None = None
    task_slug: str | None = None
    claim_path: str | None = DEFAULT_VALIDATOR_WORKLIST_PATH
    interval_seconds: float = 30.0
    cycles: int = 0
    timeout_s: float = 30.0
    allow_unsafe_host_replay: bool = False
    allow_local_artifacts: bool = False
    max_replay_log_chars: int = 128_000
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class PublicValidatorRunOutcome:
    job_id: str | None
    submission_id: str | None
    status: str
    observed_metrics: dict[str, Any]
    verification: dict[str, Any] | None
    dry_run: bool = False


class PublicValidatorRunner:
    def __init__(
        self,
        *,
        client: AutoresearchValidatorClient,
        engine: PublicReplayEngine,
        config: PublicValidatorRunnerConfig,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.engine = engine
        self.config = config
        self.log = log or (lambda _message: None)

    def run_once(self) -> PublicValidatorRunOutcome:
        if hasattr(self.client, "claim_replay_jobs"):
            jobs = self.client.claim_replay_jobs(
                task_id=self.config.task_id,
                task_slug=self.config.task_slug,
                claim_path=self.config.claim_path,
            )
        else:
            job = self.client.claim_replay_job(
                task_id=self.config.task_id,
                task_slug=self.config.task_slug,
                claim_path=self.config.claim_path,
            )
            jobs = [job] if job is not None else []
        if not jobs:
            self.log("[public-validator] idle: no pending replay jobs")
            return PublicValidatorRunOutcome(
                job_id=None,
                submission_id=None,
                status="idle",
                observed_metrics={},
                verification=None,
                dry_run=self.config.dry_run,
            )
        self.log(f"[public-validator] received {len(jobs)} replay job(s)")
        latest_outcome: PublicValidatorRunOutcome | None = None
        for job in jobs:
            self.log(f"[public-validator] checking submission={job.submission_id} source={job.source}")
            replay = self.engine.run(job)
            self.log(
                "[public-validator] replay "
                f"submission={replay.submission_id} status={replay.status} metrics={replay.observed_metrics}"
            )
            verification = None
            if self.config.dry_run:
                self.log("[public-validator] dry-run: verification result was not submitted")
            else:
                verification = self.client.submit_verification(
                    submission_id=replay.submission_id,
                    status=replay.status,
                    observed_metrics=replay.observed_metrics,
                    notes=replay.notes,
                    replay_log=replay.replay_log,
                    job_id=job.job_id,
                )
                self.log(
                    "[public-validator] submitted verification "
                    f"job={job.job_id or 'legacy'} submission={replay.submission_id} status={verification.get('status')}"
                )
            latest_outcome = PublicValidatorRunOutcome(
                job_id=job.job_id,
                submission_id=replay.submission_id,
                status=replay.status,
                observed_metrics=dict(replay.observed_metrics),
                verification=verification,
                dry_run=self.config.dry_run,
            )
        if latest_outcome is None:
            raise RuntimeError("validator worklist produced no outcomes")
        return latest_outcome

    def run_loop(self) -> list[PublicValidatorRunOutcome]:
        outcomes: list[PublicValidatorRunOutcome] = []
        completed = 0
        while int(self.config.cycles) == 0 or completed < int(self.config.cycles):
            try:
                outcomes.append(self.run_once())
            except Exception as exc:
                self.log(f"[public-validator] cycle failed: {exc}")
            completed += 1
            if int(self.config.cycles) != 0 and completed >= int(self.config.cycles):
                break
            time.sleep(max(1.0, float(self.config.interval_seconds)))
        return outcomes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SN94 public autoresearch validator replay client."
    )
    parser.add_argument(
        "--coordinator-url",
        default=DEFAULT_RESEARCH_COORDINATOR_URL,
        help=f"Autoresearch coordinator API base URL (default: {DEFAULT_RESEARCH_COORDINATOR_URL}).",
    )
    parser.add_argument("--task-id", default="", help="Optional task ID filter.")
    parser.add_argument("--task-slug", default="", help="Optional task slug filter.")
    parser.add_argument(
        "--claim-path",
        default=DEFAULT_VALIDATOR_WORKLIST_PATH,
        help=(
            "Signed backend worklist path. Defaults to the public validator scan endpoint."
        ),
    )
    parser.add_argument(
        "--pending-submissions-fallback",
        action="store_true",
        help="Use the legacy pending-submissions scan instead of the backend validator worklist endpoint.",
    )
    parser.add_argument(
        "--workspace-root",
        default=".bitsota_public_validator_workspaces",
        help="Directory for disposable replay checkouts.",
    )
    parser.add_argument("--once", action="store_true", help="Process at most one replay job and exit.")
    parser.add_argument("--cycles", type=int, default=0, help="Number of loop cycles. 0 means forever.")
    parser.add_argument("--interval-seconds", type=float, default=30.0, help="Delay between loop cycles.")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="Coordinator request timeout.")
    parser.add_argument(
        "--allow-unsafe-host-replay",
        action="store_true",
        help="Allow local host execution of submitted setup and benchmark commands.",
    )
    parser.add_argument(
        "--allow-local-artifacts",
        action="store_true",
        help="Allow file:// and relative local artifact URIs during replay.",
    )
    parser.add_argument(
        "--max-replay-log-chars",
        type=int,
        default=128_000,
        help="Maximum replay log characters to submit to the coordinator.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Replay but do not submit the validator result.")
    parser.add_argument("--hotkey-mnemonic", default="")
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--wallet-path", default="~/.bittensor/wallets/")
    parser.add_argument("--wallet-file", default="")
    return parser


def _load_wallet(args: argparse.Namespace) -> EphemeralWallet | Any:
    return load_wallet(
        hotkey_mnemonic=str(getattr(args, "hotkey_mnemonic", "") or ""),
        wallet_file=str(getattr(args, "wallet_file", "") or ""),
        wallet_name=str(getattr(args, "wallet_name", "default")),
        wallet_hotkey=str(getattr(args, "wallet_hotkey", "default")),
        wallet_path=str(getattr(args, "wallet_path", "~/.bittensor/wallets/")),
    )


def _config_from_args(args: argparse.Namespace) -> PublicValidatorRunnerConfig:
    cycles = 1 if bool(getattr(args, "once", False)) else int(getattr(args, "cycles", 0))
    return PublicValidatorRunnerConfig(
        coordinator_url=str(args.coordinator_url),
        workspace_root=Path(str(args.workspace_root)).expanduser().resolve(),
        task_id=str(getattr(args, "task_id", "") or "").strip() or None,
        task_slug=str(getattr(args, "task_slug", "") or "").strip() or None,
        claim_path=(
            None
            if bool(getattr(args, "pending_submissions_fallback", False))
            else str(getattr(args, "claim_path", "") or DEFAULT_VALIDATOR_JOB_CLAIM_PATH).strip()
        ),
        interval_seconds=float(args.interval_seconds),
        cycles=cycles,
        timeout_s=float(args.timeout_s),
        allow_unsafe_host_replay=bool(args.allow_unsafe_host_replay),
        allow_local_artifacts=bool(args.allow_local_artifacts),
        max_replay_log_chars=int(args.max_replay_log_chars),
        dry_run=bool(args.dry_run),
    )


def build_runner(
    *,
    wallet: Any,
    config: PublicValidatorRunnerConfig,
    log: Callable[[str], None] | None = None,
) -> PublicValidatorRunner:
    client = AutoresearchValidatorClient(
        base_url=config.coordinator_url,
        wallet=wallet,
        timeout_s=config.timeout_s,
    )
    engine = PublicReplayEngine(
        workspace_root=config.workspace_root,
        allow_unsafe_host_replay=config.allow_unsafe_host_replay,
        allow_local_artifacts=config.allow_local_artifacts,
        max_replay_log_chars=config.max_replay_log_chars,
    )
    return PublicValidatorRunner(client=client, engine=engine, config=config, log=log)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)
    if not config.allow_unsafe_host_replay:
        raise SystemExit(
            "host replay is disabled by default; pass --allow-unsafe-host-replay only on an isolated validator host"
        )
    wallet = _load_wallet(args)
    runner = build_runner(wallet=wallet, config=config, log=lambda message: print(message))  # noqa: T201
    if int(config.cycles) == 1:
        outcome = runner.run_once()
        print(json.dumps(asdict(outcome), indent=2, default=str))  # noqa: T201
        return 0
    outcomes = runner.run_loop()
    print(json.dumps([asdict(outcome) for outcome in outcomes], indent=2, default=str))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

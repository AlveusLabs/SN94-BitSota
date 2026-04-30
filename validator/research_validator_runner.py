from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

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
        "--config",
        default=None,
        help="Path to a YAML, .config, or JSON validator runner config file.",
    )
    parser.add_argument(
        "--coordinator-url",
        default=None,
        help=f"Autoresearch coordinator API base URL (default: {DEFAULT_RESEARCH_COORDINATOR_URL}).",
    )
    parser.add_argument("--task-id", default=None, help="Optional task ID filter.")
    parser.add_argument("--task-slug", default=None, help="Optional task slug filter.")
    parser.add_argument(
        "--claim-path",
        default=None,
        help=(
            "Signed backend worklist path. Defaults to the public validator scan endpoint."
        ),
    )
    parser.add_argument(
        "--pending-submissions-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the legacy pending-submissions scan instead of the backend validator worklist endpoint.",
    )
    parser.add_argument(
        "--workspace-root",
        default=None,
        help="Directory for disposable replay checkouts.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=None,
        help="Process at most one replay cycle and exit.",
    )
    parser.add_argument("--cycles", type=int, default=None, help="Number of loop cycles. 0 means forever.")
    parser.add_argument("--interval-seconds", type=float, default=None, help="Delay between loop cycles.")
    parser.add_argument("--timeout-s", type=float, default=None, help="Coordinator request timeout.")
    parser.add_argument(
        "--allow-unsafe-host-replay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow local host execution of submitted setup and benchmark commands.",
    )
    parser.add_argument(
        "--allow-local-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow file:// and relative local artifact URIs during replay.",
    )
    parser.add_argument(
        "--max-replay-log-chars",
        type=int,
        default=None,
        help="Maximum replay log characters to submit to the coordinator.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replay but do not submit the validator result.",
    )
    parser.add_argument("--hotkey-mnemonic", default=None)
    parser.add_argument("--wallet-name", default=None)
    parser.add_argument("--wallet-hotkey", default=None)
    parser.add_argument("--wallet-path", default=None)
    parser.add_argument("--wallet-file", default=None)
    return parser


def _load_runner_config_file(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(str(config_path)).expanduser()
    if not path.exists():
        raise SystemExit(f"validator config file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    name = path.name.lower()
    try:
        if suffix == ".json" or name.endswith(".json.example"):
            parsed = json.loads(raw)
        elif suffix in {".yaml", ".yml", ".config", ""} or name.endswith(
            (".yaml.example", ".yml.example", ".config.example")
        ):
            try:
                import yaml
            except ModuleNotFoundError as exc:
                raise SystemExit("PyYAML is required to read validator YAML config files") from exc
            try:
                parsed = yaml.safe_load(raw) or {}
            except yaml.YAMLError as exc:
                raise SystemExit(f"invalid validator YAML config {path}: {exc}") from exc
        else:
            raise SystemExit(
                "validator config must be a .yaml, .yml, .config, or .json file"
            )
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid validator JSON config {path}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise SystemExit("validator config must be a mapping/object")
    return _normalise_config_data(parsed)


def _normalise_config_data(config: Mapping[str, Any]) -> dict[str, Any]:
    normalised = {str(key): value for key, value in dict(config).items()}
    for section_name in ("validator", "runner"):
        section = config.get(section_name)
        if isinstance(section, Mapping):
            for key, value in section.items():
                normalised.setdefault(str(key), value)
    wallet = config.get("wallet")
    if isinstance(wallet, Mapping):
        wallet_aliases = {
            "name": "wallet_name",
            "hotkey": "wallet_hotkey",
            "path": "wallet_path",
            "file": "wallet_file",
            "hotkey_mnemonic": "hotkey_mnemonic",
        }
        for source_key, target_key in wallet_aliases.items():
            if source_key in wallet:
                normalised.setdefault(target_key, wallet[source_key])
    if "path" in normalised:
        normalised.setdefault("wallet_path", normalised["path"])
    return normalised


def _value_from_args_or_config(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    arg_value = getattr(args, key, None)
    if arg_value is not None:
        return arg_value
    if key in config and config[key] is not None:
        return config[key]
    return default


def _bool_value_from_args_or_config(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    key: str,
    default: bool,
) -> bool:
    value = _value_from_args_or_config(args, config, key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    normalised = str(value).strip().lower()
    if normalised in {"1", "true", "yes", "y", "on"}:
        return True
    if normalised in {"0", "false", "no", "n", "off", ""}:
        return False
    raise SystemExit(f"validator config field {key!r} must be a boolean")


def _wallet_kwargs_from_args(
    args: argparse.Namespace,
    config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    config = config or {}
    return {
        "hotkey_mnemonic": str(
            _value_from_args_or_config(args, config, "hotkey_mnemonic", "") or ""
        ),
        "wallet_file": str(_value_from_args_or_config(args, config, "wallet_file", "") or ""),
        "wallet_name": str(
            _value_from_args_or_config(args, config, "wallet_name", "default")
            or "default"
        ),
        "wallet_hotkey": str(
            _value_from_args_or_config(args, config, "wallet_hotkey", "default")
            or "default"
        ),
        "wallet_path": str(
            _value_from_args_or_config(
                args,
                config,
                "wallet_path",
                "~/.bittensor/wallets/",
            )
            or "~/.bittensor/wallets/"
        ),
    }


def _load_wallet(
    args: argparse.Namespace,
    config: Mapping[str, Any] | None = None,
) -> EphemeralWallet | Any:
    return load_wallet(**_wallet_kwargs_from_args(args, config))


def _config_from_args(
    args: argparse.Namespace,
    config_data: Mapping[str, Any] | None = None,
) -> PublicValidatorRunnerConfig:
    config_data = config_data or {}
    if bool(getattr(args, "once", False)):
        cycles = 1
    elif getattr(args, "cycles", None) is not None:
        cycles = int(args.cycles)
    elif _bool_value_from_args_or_config(args, config_data, "once", False):
        cycles = 1
    else:
        cycles = int(_value_from_args_or_config(args, config_data, "cycles", 0))
    pending_fallback = _bool_value_from_args_or_config(
        args,
        config_data,
        "pending_submissions_fallback",
        False,
    )
    claim_path_value = _value_from_args_or_config(
        args,
        config_data,
        "claim_path",
        DEFAULT_VALIDATOR_WORKLIST_PATH,
    )
    return PublicValidatorRunnerConfig(
        coordinator_url=str(
            _value_from_args_or_config(
                args,
                config_data,
                "coordinator_url",
                DEFAULT_RESEARCH_COORDINATOR_URL,
            )
        ).rstrip("/"),
        workspace_root=Path(
            str(
                _value_from_args_or_config(
                    args,
                    config_data,
                    "workspace_root",
                    ".bitsota_public_validator_workspaces",
                )
            )
        )
        .expanduser()
        .resolve(),
        task_id=str(
            _value_from_args_or_config(args, config_data, "task_id", "") or ""
        ).strip()
        or None,
        task_slug=str(
            _value_from_args_or_config(args, config_data, "task_slug", "") or ""
        ).strip()
        or None,
        claim_path=(
            None
            if pending_fallback and getattr(args, "claim_path", None) is None
            else str(claim_path_value or DEFAULT_VALIDATOR_WORKLIST_PATH).strip()
        ),
        interval_seconds=float(
            _value_from_args_or_config(args, config_data, "interval_seconds", 30.0)
        ),
        cycles=cycles,
        timeout_s=float(
            _value_from_args_or_config(args, config_data, "timeout_s", 30.0)
        ),
        allow_unsafe_host_replay=_bool_value_from_args_or_config(
            args,
            config_data,
            "allow_unsafe_host_replay",
            False,
        ),
        allow_local_artifacts=_bool_value_from_args_or_config(
            args,
            config_data,
            "allow_local_artifacts",
            False,
        ),
        max_replay_log_chars=int(
            _value_from_args_or_config(
                args,
                config_data,
                "max_replay_log_chars",
                128_000,
            )
        ),
        dry_run=_bool_value_from_args_or_config(args, config_data, "dry_run", False),
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
    config_data = _load_runner_config_file(getattr(args, "config", None))
    config = _config_from_args(args, config_data)
    if not config.allow_unsafe_host_replay:
        raise SystemExit(
            "host replay is disabled by default; set allow_unsafe_host_replay: true in the config "
            "or pass --allow-unsafe-host-replay only on an isolated validator host"
        )
    wallet = _load_wallet(args, config_data)
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

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import requests


DEFAULT_AUTORESEARCH_COORDINATOR_URL = "https://autoresearch.bitsota.com"
DEFAULT_REWARD_SNAPSHOT_PATH = "/api/v1/reward-snapshot"


@dataclass(frozen=True, slots=True)
class BackendWeightSetOutcome:
    mode: str
    status: str
    scores: dict[str, float]
    dry_run: bool = False
    message: str = ""


def _join_url(base_url: str, path: str) -> str:
    return f"{str(base_url).rstrip('/')}/{str(path).lstrip('/')}"


def _policy_mode(policy: Mapping[str, Any]) -> str:
    return str(policy.get("mode") or "local").strip().lower()


def fetch_reward_snapshot(
    *,
    coordinator_url: str = DEFAULT_AUTORESEARCH_COORDINATOR_URL,
    snapshot_path: str = DEFAULT_REWARD_SNAPSHOT_PATH,
    timeout_s: float = 30.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    client = session or requests.Session()
    response = client.get(
        _join_url(coordinator_url, snapshot_path),
        timeout=max(1.0, float(timeout_s)),
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"GET {snapshot_path} failed: HTTP {response.status_code} ({response.text})"
        )
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise RuntimeError("reward snapshot must be a JSON object")
    return dict(payload)


def extract_validator_weight_policy(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    reward_policy = snapshot.get("reward_policy")
    if not isinstance(reward_policy, Mapping):
        raise RuntimeError("reward snapshot is missing reward_policy")
    validator_weights = reward_policy.get("validator_weights") or {"mode": "local"}
    if not isinstance(validator_weights, Mapping):
        raise RuntimeError("reward_policy.validator_weights must be an object")
    return dict(validator_weights)


def _hotkey_for_uid(metagraph_hotkeys: Sequence[str], uid: int) -> str:
    if uid < 0 or uid >= len(metagraph_hotkeys):
        raise RuntimeError(
            f"backend validator weight target uid={uid} is outside metagraph size {len(metagraph_hotkeys)}"
        )
    hotkey = str(metagraph_hotkeys[uid] or "").strip()
    if not hotkey:
        raise RuntimeError(f"metagraph uid={uid} has an empty hotkey")
    return hotkey


def _normalise_scores(scores: Mapping[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in scores.values())
    if total <= 0.0:
        raise RuntimeError("backend validator weights must sum to a positive value")
    return {
        hotkey: float(weight) / total
        for hotkey, weight in sorted(scores.items())
        if float(weight) > 0.0
    }


def resolve_validator_weight_scores(
    policy: Mapping[str, Any],
    *,
    metagraph_hotkeys: Sequence[str],
) -> dict[str, float]:
    mode = _policy_mode(policy)
    hotkeys = [str(hotkey) for hotkey in metagraph_hotkeys]

    if mode == "local":
        return {}
    if mode == "burn_uid0":
        return {_hotkey_for_uid(hotkeys, 0): 1.0}
    if mode != "targets":
        raise RuntimeError(f"unsupported backend validator weight mode: {mode}")

    raw_targets = policy.get("targets")
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)):
        raise RuntimeError("validator_weights.targets must be a list")
    if not raw_targets:
        raise RuntimeError("validator_weights.targets must not be empty")

    scores: dict[str, float] = {}
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, Mapping):
            raise RuntimeError(f"validator_weights.targets[{index}] must be an object")

        try:
            weight = float(raw_target.get("weight"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"validator_weights.targets[{index}].weight must be numeric"
            ) from exc
        if weight <= 0.0:
            raise RuntimeError(
                f"validator_weights.targets[{index}].weight must be positive"
            )

        has_uid = raw_target.get("uid") is not None
        raw_hotkey = str(raw_target.get("hotkey") or "").strip()
        has_hotkey = bool(raw_hotkey)
        if has_uid == has_hotkey:
            raise RuntimeError(
                f"validator_weights.targets[{index}] must define exactly one of uid or hotkey"
            )

        if has_uid:
            try:
                hotkey = _hotkey_for_uid(hotkeys, int(raw_target["uid"]))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"validator_weights.targets[{index}].uid must be an integer"
                ) from exc
        else:
            if raw_hotkey not in hotkeys:
                raise RuntimeError(
                    f"backend validator weight target hotkey is not in metagraph: {raw_hotkey}"
                )
            hotkey = raw_hotkey

        scores[hotkey] = scores.get(hotkey, 0.0) + weight

    return _normalise_scores(scores)


def apply_backend_weight_policy(
    *,
    policy: Mapping[str, Any],
    network: Any,
    dry_run: bool = False,
    respect_rate_limit: bool = True,
    log: Callable[[str], None] | None = None,
) -> BackendWeightSetOutcome:
    logger = log or (lambda _message: None)
    mode = _policy_mode(policy)
    if hasattr(network, "resync_metagraph"):
        network.resync_metagraph(lite=True)

    metagraph = getattr(network, "metagraph", None)
    metagraph_hotkeys = list(getattr(metagraph, "hotkeys", []) or [])
    scores = resolve_validator_weight_scores(policy, metagraph_hotkeys=metagraph_hotkeys)

    if not scores:
        return BackendWeightSetOutcome(
            mode=mode,
            status="skipped_local",
            scores={},
            dry_run=bool(dry_run),
            message="backend policy mode is local; no legacy relay/local fallback is run",
        )

    for hotkey, weight in scores.items():
        try:
            uid = metagraph_hotkeys.index(hotkey)
            uid_label = f"uid={uid}"
        except ValueError:
            uid_label = "uid=?"
        logger(
            "[backend-weights] "
            f"target {uid_label} hotkey={hotkey} weight={weight:.6f} ({weight * 100:.3f}%)"
        )

    if respect_rate_limit and hasattr(network, "should_set_weights"):
        if not bool(network.should_set_weights()):
            return BackendWeightSetOutcome(
                mode=mode,
                status="skipped_rate_limit",
                scores=dict(scores),
                dry_run=bool(dry_run),
                message="chain weight rate limit is not ready",
            )

    if dry_run:
        return BackendWeightSetOutcome(
            mode=mode,
            status="dry_run",
            scores=dict(scores),
            dry_run=True,
            message="resolved backend weights without submitting set_weights",
        )

    ok = bool(network.set_weights(dict(scores)))
    return BackendWeightSetOutcome(
        mode=mode,
        status="submitted" if ok else "failed",
        scores=dict(scores),
        dry_run=False,
        message="set_weights returned ok" if ok else "set_weights returned false",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set SN94 validator weights from the autoresearch backend reward snapshot."
    )
    parser.add_argument(
        "--config",
        "-c",
        default="validator_config.weights.yaml",
        help="Path to validator weight config with wallet and subtensor settings.",
    )
    parser.add_argument(
        "--coordinator-url",
        default=DEFAULT_AUTORESEARCH_COORDINATOR_URL,
        help=f"Autoresearch backend base URL (default: {DEFAULT_AUTORESEARCH_COORDINATOR_URL}).",
    )
    parser.add_argument(
        "--snapshot-path",
        default=DEFAULT_REWARD_SNAPSHOT_PATH,
        help=f"Reward snapshot path (default: {DEFAULT_REWARD_SNAPSHOT_PATH}).",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep polling the backend and setting weights when the chain rate limit allows it.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=300.0,
        help="Delay between loop iterations when --loop is set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and resolve backend targets without submitting set_weights.",
    )
    parser.add_argument(
        "--ignore-rate-limit",
        action="store_true",
        help="Attempt set_weights even when the local weight-rate-limit gate says to wait.",
    )
    parser.add_argument("--wallet-name", default=None)
    parser.add_argument("--wallet-hotkey", default=None)
    parser.add_argument("--wallet-path", default=None)
    parser.add_argument("--network", default=None)
    parser.add_argument("--netuid", type=int, default=None)
    parser.add_argument("--subtensor-chain-endpoint", default=None)
    return parser


def _apply_config_overrides(config: Any, args: argparse.Namespace) -> None:
    overrides = {
        "wallet_name": args.wallet_name,
        "wallet_hotkey": args.wallet_hotkey,
        "path": args.wallet_path,
        "network": args.network,
        "netuid": args.netuid,
        "subtensor_chain_endpoint": args.subtensor_chain_endpoint,
    }
    for key, value in overrides.items():
        if value is None:
            continue
        setattr(config, key, value)
        if hasattr(config, "yaml_data") and isinstance(config.yaml_data, dict):
            config.yaml_data[key] = value

    if args.wallet_name is not None:
        setattr(config.wallet, "name", args.wallet_name)
    if args.wallet_hotkey is not None:
        setattr(config.wallet, "hotkey", args.wallet_hotkey)
    if args.wallet_path is not None:
        setattr(config.wallet, "path", args.wallet_path)
    if args.network is not None:
        setattr(config.subtensor, "network", args.network)
    if args.subtensor_chain_endpoint is not None:
        setattr(config.subtensor, "chain_endpoint", args.subtensor_chain_endpoint)


def _run_once(
    *,
    args: argparse.Namespace,
    network: Any,
    log: Callable[[str], None],
) -> BackendWeightSetOutcome:
    snapshot = fetch_reward_snapshot(
        coordinator_url=args.coordinator_url,
        snapshot_path=args.snapshot_path,
        timeout_s=args.timeout_s,
    )
    policy = extract_validator_weight_policy(snapshot)
    log(f"[backend-weights] fetched policy mode={_policy_mode(policy)}")
    return apply_backend_weight_policy(
        policy=policy,
        network=network,
        dry_run=bool(args.dry_run),
        respect_rate_limit=not bool(args.ignore_rate_limit),
        log=log,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)

    from bittensor_network import BittensorNetwork
    from bittensor_network.bittensor_config import BittensorConfig

    config = BittensorConfig.get_bittensor_config(
        args.config,
        bt_argv=["backend_weight_setter", *remaining],
    )
    _apply_config_overrides(config, args)
    network = BittensorNetwork(config, force_reinit=True)

    def log(message: str) -> None:
        print(message, flush=True)

    try:
        while True:
            try:
                outcome = _run_once(args=args, network=network, log=log)
                log(
                    "[backend-weights] "
                    f"status={outcome.status} mode={outcome.mode} dry_run={outcome.dry_run} "
                    f"message={outcome.message}"
                )
            except Exception as exc:
                log(f"[backend-weights] failed: {exc}")
                if not args.loop:
                    return 1

            if not args.loop:
                return 0 if outcome.status != "failed" else 1
            time.sleep(max(1.0, float(args.interval_seconds)))
    finally:
        close = getattr(network, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

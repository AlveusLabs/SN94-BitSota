#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.engine_defaults import apply_cpp_defaults_to_engine_params
from miner.local_benchmark import run_local_benchmark_worker


class _StdoutQueue:
    def __init__(self, *, jsonl: bool = False):
        self._jsonl = bool(jsonl)

    def put(self, msg: Any) -> None:
        if not self._jsonl:
            if isinstance(msg, dict):
                if msg.get("type") == "error":
                    tb = msg.get("traceback") or msg
                    print(f"[error] {tb}", file=sys.stderr, flush=True)
                    return
                line = msg.get("log")
                if line:
                    print(str(line), flush=True)
                    return
            return

        try:
            payload = json.dumps(msg, separators=(",", ":"), sort_keys=False)
        except Exception:
            payload = json.dumps({"type": "unserializable", "repr": repr(msg)})
        print(payload, flush=True)


def _load_engine_params(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    if raw.endswith(".json") and Path(raw).exists():
        try:
            return json.loads(Path(raw).read_text())
        except Exception:
            return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Offline AutoML search benchmark (no relay/wallet HTTP).")
    parser.add_argument("--task-type", default="cifar10_binary", help="Task type (default: cifar10_binary)")
    parser.add_argument("--engine-type", default="baseline", help="Engine type: baseline|archive (default: baseline)")
    parser.add_argument("--checkpoint-generations", type=int, default=10, help="Emit stats every N generations")
    parser.add_argument(
        "--validate-every-n-generations",
        type=int,
        default=None,
        help="Validator-suite verification cadence (default: env MINER_VALIDATE_EVERY_N_GENERATIONS or 1)",
    )
    parser.add_argument(
        "--validator-task-count",
        type=int,
        default=None,
        help="Number of tasks for verification (default: core.evaluations default)",
    )
    parser.add_argument(
        "--miner-task-count",
        type=int,
        default=None,
        help="Miner eval suite size (default: env MINER_TASK_COUNT or engine default)",
    )
    parser.add_argument(
        "--engine-params",
        default=None,
        help="JSON string or path to JSON file of engine params (pop_size, phase_max_sizes, etc)",
    )
    parser.add_argument(
        "--sota-threshold",
        type=float,
        default=None,
        help="Optional bar used for distance_to_sota + is_sota_breaker (default: unset)",
    )
    parser.add_argument("--public-address", default="local", help="State namespace (default: local)")
    parser.add_argument("--state-dir", default=None, help="Override MINER_STATE_DIR / default state dir")
    parser.add_argument("--no-persist", action="store_true", help="Disable state persistence")
    parser.add_argument("--persist-every", type=int, default=None, help="Persist engine every N generations")
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed (worker_id is added)")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker id (default: 0)")
    parser.add_argument("--jsonl", action="store_true", help="Emit structured JSONL instead of log lines")

    args = parser.parse_args(argv)

    engine_params = _load_engine_params(args.engine_params)
    engine_params = apply_cpp_defaults_to_engine_params(
        str(args.task_type),
        engine_params if isinstance(engine_params, dict) else None,
        explicit_engine_params=engine_params if isinstance(engine_params, dict) else None,
    )

    cfg: Dict[str, Any] = {
        "task_type": str(args.task_type),
        "engine_type": str(args.engine_type),
        "checkpoint_generations": int(args.checkpoint_generations),
        "validate_every_n_generations": int(args.validate_every_n_generations)
        if args.validate_every_n_generations is not None
        else None,
        "validator_task_count": int(args.validator_task_count) if args.validator_task_count is not None else None,
        "miner_task_count": int(args.miner_task_count) if args.miner_task_count is not None else None,
        "engine_params": engine_params,
        "sota_threshold": float(args.sota_threshold) if args.sota_threshold is not None else None,
        "public_address": str(args.public_address),
        "state_dir": str(args.state_dir) if args.state_dir else None,
        "persist_state": not bool(args.no_persist),
        "persist_every_n_generations": int(args.persist_every) if args.persist_every is not None else None,
        "seed": int(args.seed) if args.seed is not None else None,
        "migration_generations": 0,
        # Explicitly disable noisy engine-side logging; the runner emits its own sparse stats.
        "engine_verbose": False,
    }

    stop_event = threading.Event()
    out_queue = _StdoutQueue(jsonl=bool(args.jsonl))
    try:
        run_local_benchmark_worker(cfg, int(args.worker_id), out_queue, None, stop_event)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""GUI-side wrapper to run AutoML-Zero C++ backend through the sidecar.

Modes:
- direct: emits candidate events (GUI direct miner flow)
- lease: processes sidecar lease/evaluate/evolve jobs (GUI pool-lease flow)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_automl_zero_root() -> str:
    return os.getenv(
        "AUTOML_ZERO_CPP_ROOT",
        "/home/mekaneeky/repos/automl_zero_original/automl_zero",
    )


def _default_bridge_path(automl_root: str) -> str:
    env_path = os.getenv("AUTOML_ZERO_CPP_BRIDGE")
    if env_path:
        return str(Path(env_path).expanduser())
    return str(Path(automl_root).expanduser() / "tools" / "baseline_sidecar_bridge.py")


def _default_baseline_cmd(automl_root: str) -> str:
    env_cmd = os.getenv("AUTOML_ZERO_CPP_BASELINE_CMD")
    if env_cmd:
        return str(Path(env_cmd).expanduser())
    return str(Path(automl_root).expanduser() / "run_baseline.sh")


def _load_seed_from_config(config_path: Optional[str]) -> Optional[int]:
    if not config_path:
        return None
    try:
        p = Path(str(config_path)).expanduser().resolve()
    except Exception:
        return None
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    args = payload.get("args")
    if isinstance(args, dict):
        try:
            if args.get("seed") is not None:
                return int(args.get("seed"))
        except Exception:
            return None
    try:
        if payload.get("seed") is not None:
            return int(payload.get("seed"))
    except Exception:
        return None
    return None


def _parse_extra_baseline_args() -> List[str]:
    raw = str(os.getenv("AUTOML_ZERO_CPP_EXTRA_ARGS", "")).strip()
    if not raw:
        return []
    try:
        return list(shlex.split(raw))
    except Exception:
        return []


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _infer_population_size_from_baseline_cmd(baseline_cmd: str) -> int:
    fallback = 100
    try:
        text = Path(str(baseline_cmd)).expanduser().read_text(errors="ignore")
    except Exception:
        return fallback
    match = re.search(r"population_size:\\s*(\\d+)", text)
    if not match:
        return fallback
    try:
        return max(1, int(match.group(1)))
    except Exception:
        return fallback


def _lease_iters_from_target_indivs(target_indivs: int, population_size: int) -> int:
    pop = max(1, int(population_size))
    target = max(1, int(target_indivs))
    # run_search_experiment counts total individuals including the initial population.
    remaining = max(0, target - pop)
    iters = (remaining + pop - 1) // pop
    return max(1, int(iters))


def _build_direct_worker_cmd(
    *,
    python: str,
    bridge_path: str,
    sidecar_url: str,
    run_id: str,
    worker_id: int,
    task_type: str,
    input_dim: int,
    baseline_cmd: str,
    iterations: int,
    random_seed: Optional[int],
    echo_stdout: bool,
) -> List[str]:
    cmd = [
        python,
        bridge_path,
        "direct",
        "--sidecar-url",
        sidecar_url,
        "--run-id",
        run_id,
        "--worker-id",
        str(worker_id),
        "--task-type",
        task_type,
        "--input-dim",
        str(int(input_dim)),
        "--baseline-cmd",
        baseline_cmd,
    ]
    if echo_stdout:
        cmd.append("--echo-stdout")
    cmd.append("--")
    if iterations > 0:
        cmd.append(f"--iters={int(iterations)}")
    else:
        cmd.append("--iters=-1")
    # GUI miner should keep running like direct miner; do not stop at 1 experiment.
    cmd.append("--max-experiments=0")
    if random_seed is not None:
        cmd.append(f"--random_seed={int(random_seed)}")
    cmd.extend(_parse_extra_baseline_args())
    return cmd


def _build_lease_worker_cmd(
    *,
    python: str,
    bridge_path: str,
    sidecar_url: str,
    run_id: str,
    worker_id: int,
    baseline_cmd: str,
    lease_iters: int,
    eval_mode: str,
    bitsota_root: str,
    echo_stdout: bool,
) -> List[str]:
    cmd = [
        python,
        bridge_path,
        "lease",
        "--sidecar-url",
        sidecar_url,
        "--run-id",
        run_id,
        "--worker-id",
        str(worker_id),
        "--baseline-cmd",
        baseline_cmd,
        "--lease-iters",
        str(max(1, int(lease_iters))),
        "--eval-mode",
        str(eval_mode),
        "--bitsota-root",
        str(bitsota_root),
    ]
    if echo_stdout:
        cmd.append("--echo-stdout")
    if _env_flag("BITSOTA_CPP_TRACE_ACTIONS", False):
        cmd.append("--trace-actions")
    cmd.append("--")
    cmd.extend(_parse_extra_baseline_args())
    return cmd


def _terminate_children(children: List[subprocess.Popen]) -> None:
    for proc in children:
        if proc.poll() is not None:
            continue
        try:
            if os.name == "posix":
                os.killpg(int(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            pass

    deadline = time.time() + 5.0
    for proc in children:
        if proc.poll() is not None:
            continue
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except Exception:
            try:
                if os.name == "posix":
                    os.killpg(int(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C++ AutoML-Zero backend through sidecar.")
    parser.add_argument("--cpp-mode", choices=["direct", "lease"], default="direct")
    parser.add_argument("--sidecar-url", default=os.getenv("BITSOTA_SIDECAR_URL", "http://127.0.0.1:8123"))
    parser.add_argument("--run-id", default=os.getenv("BITSOTA_SIDECAR_RUN_ID", "cpp_run"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-type", default="cifar10_binary")
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", choices=["real", "mock"], default="real")
    parser.add_argument("--lease-evolve-generations", type=int, default=160)
    parser.add_argument(
        "--lease-evolve-indivs",
        "--lease-indivs",
        dest="lease_evolve_indivs",
        type=int,
        default=0,
        help=(
            "Target total individuals (includes initial population). "
            "If > 0, overrides --lease-evolve-generations after converting via population size."
        ),
    )
    parser.add_argument("--echo-stdout", action="store_true")
    parser.add_argument("--sota-threshold", type=float, default=0.0)  # accepted for GUI compatibility
    parser.add_argument("--population-snapshot-every", type=int, default=0)  # accepted for GUI compatibility
    parser.add_argument("--initial-population-path", type=str, default=None)  # accepted for GUI compatibility
    parser.add_argument("--automl-root", type=str, default=_default_automl_zero_root())
    parser.add_argument("--bitsota-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    automl_root = str(Path(args.automl_root).expanduser())
    bridge_path = _default_bridge_path(automl_root)
    baseline_cmd = _default_baseline_cmd(automl_root)
    python = sys.executable
    workers = max(1, int(args.workers))

    seed0 = _load_seed_from_config(args.config)
    if seed0 is None:
        seed_env = os.getenv("BITSOTA_CPP_RANDOM_SEED", "").strip()
        if seed_env:
            try:
                seed0 = int(seed_env)
            except Exception:
                seed0 = None

    if str(args.cpp_mode) == "direct":
        task_type = str(args.task_type or "cifar10_binary")
        if task_type in {"pool", "pool_lease"}:
            task_type = "cifar10_binary"
        worker_cmds = [
            _build_direct_worker_cmd(
                python=python,
                bridge_path=bridge_path,
                sidecar_url=str(args.sidecar_url),
                run_id=str(args.run_id),
                worker_id=wid,
                task_type=task_type,
                input_dim=int(args.feature_dim),
                baseline_cmd=baseline_cmd,
                iterations=int(args.iterations),
                random_seed=(seed0 + wid) if seed0 is not None else None,
                echo_stdout=bool(args.echo_stdout),
            )
            for wid in range(workers)
        ]
    else:
        eval_mode = str(args.mode).lower()
        if eval_mode not in {"real", "mock"}:
            eval_mode = "real"
        lease_iters = int(args.lease_evolve_generations)
        if int(args.lease_evolve_indivs) > 0:
            pop_size = _infer_population_size_from_baseline_cmd(baseline_cmd)
            lease_iters = _lease_iters_from_target_indivs(
                target_indivs=int(args.lease_evolve_indivs),
                population_size=pop_size,
            )
            print(
                "[miner-cpp-sidecar] "
                f"lease_indivs={int(args.lease_evolve_indivs)} pop_size={int(pop_size)} "
                f"-> lease_iters={int(lease_iters)}"
            )
        worker_cmds = [
            _build_lease_worker_cmd(
                python=python,
                bridge_path=bridge_path,
                sidecar_url=str(args.sidecar_url),
                run_id=str(args.run_id),
                worker_id=wid,
                baseline_cmd=baseline_cmd,
                lease_iters=int(lease_iters),
                eval_mode=eval_mode,
                bitsota_root=str(args.bitsota_root),
                echo_stdout=bool(args.echo_stdout),
            )
            for wid in range(workers)
        ]

    children: List[subprocess.Popen] = []
    try:
        for cmd in worker_cmds:
            proc = subprocess.Popen(
                cmd,
                cwd=automl_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
                start_new_session=True,
            )
            children.append(proc)

        while True:
            alive = [p for p in children if p.poll() is None]
            if not alive:
                # All workers exited.
                exit_codes = [int(p.returncode or 0) for p in children]
                return 0 if all(code == 0 for code in exit_codes) else 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        _terminate_children(children)


if __name__ == "__main__":
    raise SystemExit(main())

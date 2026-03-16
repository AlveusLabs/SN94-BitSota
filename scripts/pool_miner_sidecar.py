#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import requests


if __package__ in {None, ""}:
    # Allow running as `python scripts/pool_miner_sidecar.py` in addition to
    # module mode (`python -m scripts.pool_miner_sidecar`).
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)


def _default_sidecar_url() -> str:
    host = os.getenv("BITSOTA_SIDECAR_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.getenv("BITSOTA_SIDECAR_PORT", "8123").strip() or "8123"
    return f"http://{host}:{port}"


def _now_s() -> float:
    return float(time.time())


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _hash_to_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    n = int.from_bytes(digest[:8], byteorder="big")
    return n / float(2**64 - 1)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _as_epoch_s(value: Any) -> Optional[float]:
    direct = _as_float(value)
    if direct is not None:
        return direct
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def _effective_submit_buffer_s(
    configured_buffer_s: float,
    *,
    lease_timeout_at_s: Optional[float],
    now_s: Optional[float] = None,
) -> float:
    """Clamp submit buffer so lease mode still has time to evolve."""
    base = max(0.0, float(configured_buffer_s))
    if lease_timeout_at_s is None:
        return base
    now = _now_s() if now_s is None else float(now_s)
    remaining_s = max(0.0, float(lease_timeout_at_s) - now)
    if remaining_s <= 0.0:
        return 0.0
    # Keep enough runway for submission retries, but do not starve evolution.
    dynamic_cap_s = max(10.0, min(60.0, remaining_s * 0.33))
    return min(base, dynamic_cap_s)


def _mock_score(algorithm_id: int, algorithm_dsl: str, *, salt: str) -> float:
    base = _hash_to_unit_interval(f"{salt}:{algorithm_id}:{algorithm_dsl}")
    # Favor mid-range scores to avoid instantly saturating any leaderboards.
    return _clamp01(0.1 + 0.8 * float(base))


def _mock_evolve(parents: List[Dict[str, Any]], *, input_dim: int, rng: random.Random) -> str:
    vector_dim = int(input_dim) + 10
    templates = [
        f"""# meta: scalar_count=20 vector_count=10 matrix_count=5 vector_dim={vector_dim} setup_max_ops=30 predict_max_ops=30 learn_max_ops=30

# predict:
s0 = dot(v0, v1)
""",
        f"""# meta: scalar_count=20 vector_count=10 matrix_count=5 vector_dim={vector_dim} setup_max_ops=30 predict_max_ops=30 learn_max_ops=30

# predict:
s1 = norm(v0)
s0 = s1 * 0.5
""",
        f"""# meta: scalar_count=20 vector_count=10 matrix_count=5 vector_dim={vector_dim} setup_max_ops=30 predict_max_ops=30 learn_max_ops=30

# setup:
s1 = 0.1

# predict:
s2 = dot(v0, v1)
s0 = s1 + s2
""",
        f"""# meta: scalar_count=20 vector_count=10 matrix_count=5 vector_dim={vector_dim} setup_max_ops=30 predict_max_ops=30 learn_max_ops=30

# predict:
s1 = mean(v0)
s0 = abs(s1)
""",
    ]

    parent = None
    if parents:
        parent = rng.choice(parents) if len(parents) > 1 else parents[0]
    parent_id = None
    try:
        parent_id = int(parent.get("id")) if isinstance(parent, dict) else None
    except Exception:
        parent_id = None

    mutated = rng.choice(templates).strip()
    if parent_id is not None:
        mutated = f"# parent_id: {parent_id}\n{mutated}"
    return mutated.strip() + "\n"


class _SidecarClient:
    def __init__(self, base_url: str, *, run_id: str, worker_id: str, timeout_s: float) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.run_id = str(run_id)
        self.worker_id = str(worker_id)
        self.timeout_s = max(0.1, float(timeout_s))
        self._session = requests.Session()

    def log(self, message: str) -> None:
        try:
            self._session.post(
                f"{self.base_url}/ingest_batch",
                json={"run_id": self.run_id, "events": [{"type": "log", "message": str(message)}]},
                timeout=self.timeout_s,
            )
        except Exception:
            return

    def progress(self, iteration: int, *, rate: Optional[float] = None) -> None:
        ev: Dict[str, Any] = {"type": "progress", "worker_id": self.worker_id, "iteration": int(iteration)}
        if rate is not None:
            ev["rate"] = float(rate)
        try:
            self._session.post(
                f"{self.base_url}/ingest_batch",
                json={"run_id": self.run_id, "events": [ev]},
                timeout=self.timeout_s,
            )
        except Exception:
            return

    def lease_job(self, *, lease_seconds: float) -> Optional[Dict[str, Any]]:
        try:
            r = self._session.get(
                f"{self.base_url}/jobs/next",
                params={
                    "run_id": self.run_id,
                    "worker_id": self.worker_id,
                    "lease_seconds": float(lease_seconds),
                },
                timeout=self.timeout_s,
            )
        except Exception:
            return None

        if r.status_code == 204:
            return None
        if r.status_code != 200:
            return None
        try:
            payload = r.json() or {}
        except Exception:
            return None
        job = payload.get("job")
        return job if isinstance(job, dict) else None

    def submit_result(self, job_id: str, *, ok: bool, result: Dict[str, Any], error: Optional[str]) -> None:
        body: Dict[str, Any] = {
            "run_id": self.run_id,
            "status": "ok" if ok else "error",
            "result": dict(result or {}),
        }
        if error:
            body["error"] = str(error)
        try:
            self._session.post(
                f"{self.base_url}/jobs/{job_id}/result",
                json=body,
                timeout=self.timeout_s,
            )
        except Exception:
            return


def _run_worker(
    *,
    sidecar_url: str,
    run_id: str,
    worker_id: str,
    poll_interval_s: float,
    lease_seconds: float,
    lease_submit_buffer_s: float,
    lease_evolve_reserve_s: float,
    mode: str,
    evolve_generations: int,
    lease_evolve_generations: int,
    seed: Optional[int],
    stop: threading.Event,
) -> None:
    client = _SidecarClient(sidecar_url, run_id=run_id, worker_id=worker_id, timeout_s=2.0)

    rng = random.Random()
    if seed is None:
        rng.seed(int(_hash_to_unit_interval(worker_id) * (2**31 - 1)))
    else:
        rng.seed(int(seed))

    salt = f"{run_id}:{worker_id}"
    completed = 0
    last_completed_s = _now_s()

    client.log(f"[pool-miner] worker={worker_id} mode={mode}")

    while not stop.is_set():
        job = client.lease_job(lease_seconds=lease_seconds)
        if job is None:
            time.sleep(poll_interval_s)
            continue

        job_id = str(job.get("job_id") or "").strip()
        kind = str(job.get("kind") or "").strip()
        payload = job.get("payload") or {}
        if not job_id or not kind or not isinstance(payload, dict):
            client.log(f"[pool-miner] invalid job payload: {job}")
            time.sleep(poll_interval_s)
            continue

        try:
            if kind == "evaluate":
                algorithms = payload.get("algorithms") or []
                evaluations: List[Dict[str, Any]] = []
                for algo in algorithms:
                    if not isinstance(algo, dict):
                        continue
                    try:
                        algo_id = int(algo.get("id"))
                    except Exception:
                        continue
                    dsl = str(algo.get("algorithm_dsl") or "")

                    if mode == "real":
                        from core.evaluations import score_algorithm_on_eval_suite

                        input_dim = int(algo.get("input_dim") or 16)
                        score = float(score_algorithm_on_eval_suite(dsl, input_dim=input_dim))
                    else:
                        score = float(_mock_score(algo_id, dsl, salt=salt))

                    evaluations.append({"algorithm_id": algo_id, "score": score})

                result = {"evaluations": evaluations}
                client.submit_result(job_id, ok=True, result=result, error=None)
                client.log(f"[pool-miner] evaluate job={job_id} n={len(evaluations)}")

            elif kind == "evolve":
                algorithms = payload.get("algorithms") or []
                try:
                    input_dim = int(payload.get("input_dim") or 16)
                except Exception:
                    input_dim = 16

                if mode == "real":
                    from core.dsl_parser import DSLParser
                    from core.tasks.cifar10 import CIFAR10BinaryTask
                    from miner.engines.archive_engine import ArchiveAwareBaselineEvolution

                    task = CIFAR10BinaryTask()
                    task.load_data(task_id=0)
                    engine = ArchiveAwareBaselineEvolution(task=task, pop_size=5, verbose=False)
                    for parent in algorithms:
                        if not isinstance(parent, dict):
                            continue
                        dsl = str(parent.get("algorithm_dsl") or "")
                        if not dsl.strip():
                            continue
                        try:
                            algo = DSLParser.from_dsl(dsl, input_dim)
                        except Exception:
                            continue
                        for _ in range(2):
                            try:
                                engine._random_mutate(algo)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                    best_algo, _ = engine.evolve(generations=max(1, int(evolve_generations)))
                    evolved_dsl = DSLParser.to_dsl(best_algo) if best_algo is not None else _mock_evolve(algorithms, input_dim=input_dim, rng=rng)
                else:
                    evolved_dsl = _mock_evolve(algorithms, input_dim=input_dim, rng=rng)

                result = {"evolved_function": str(evolved_dsl)}
                client.submit_result(job_id, ok=True, result=result, error=None)
                client.log(f"[pool-miner] evolve job={job_id} dsl_len={len(evolved_dsl)}")

            elif kind == "lease":
                eval_algorithms = payload.get("evaluate_algorithms") or payload.get("algorithms") or []
                seed_algorithms = payload.get("seed_algorithms") or []
                lease_timeout_at_s = _as_epoch_s(payload.get("lease_timeout_at_s"))
                submit_deadline_s: Optional[float] = None
                effective_submit_buffer_s = max(0.0, float(lease_submit_buffer_s))
                if lease_timeout_at_s is not None:
                    now_for_deadline = _now_s()
                    effective_submit_buffer_s = _effective_submit_buffer_s(
                        lease_submit_buffer_s,
                        lease_timeout_at_s=lease_timeout_at_s,
                        now_s=now_for_deadline,
                    )
                    submit_deadline_s = lease_timeout_at_s - effective_submit_buffer_s
                    if effective_submit_buffer_s + 1e-6 < max(0.0, float(lease_submit_buffer_s)):
                        remaining_s = max(0.0, float(lease_timeout_at_s) - now_for_deadline)
                        client.log(
                            f"[pool-miner] lease submit buffer clamped "
                            f"{float(lease_submit_buffer_s):.1f}s->{effective_submit_buffer_s:.1f}s "
                            f"(remaining={remaining_s:.1f}s)"
                        )

                def _deadline_reached() -> bool:
                    return submit_deadline_s is not None and _now_s() >= float(submit_deadline_s)

                try:
                    evolve_budget = int(payload.get("evolve_budget") or 0)
                except Exception:
                    evolve_budget = 0
                effective_evolve_reserve_s = max(0.0, float(lease_evolve_reserve_s))
                if (
                    evolve_budget > 0
                    and submit_deadline_s is not None
                    and effective_evolve_reserve_s > 0.0
                ):
                    now_for_reserve = _now_s()
                    remaining_until_deadline_s = max(
                        0.0, float(submit_deadline_s) - now_for_reserve
                    )
                    reserve_cap_s = max(0.0, remaining_until_deadline_s * 0.8)
                    capped_reserve_s = min(effective_evolve_reserve_s, reserve_cap_s)
                    if capped_reserve_s + 1e-6 < effective_evolve_reserve_s:
                        client.log(
                            f"[pool-miner] lease evolve reserve clamped "
                            f"{effective_evolve_reserve_s:.1f}s->{capped_reserve_s:.1f}s "
                            f"(remaining_to_deadline={remaining_until_deadline_s:.1f}s)"
                        )
                    effective_evolve_reserve_s = capped_reserve_s

                def _eval_window_exhausted() -> bool:
                    if evolve_budget <= 0 or submit_deadline_s is None:
                        return False
                    if effective_evolve_reserve_s <= 0.0:
                        return False
                    return _now_s() >= float(submit_deadline_s) - float(effective_evolve_reserve_s)

                evaluations: List[Dict[str, Any]] = []
                lease_iterations = 0
                for algo in eval_algorithms:
                    if _eval_window_exhausted():
                        client.log(
                            f"[pool-miner] lease evaluation stopped early job={job_id}; "
                            f"reserving {effective_evolve_reserve_s:.1f}s for evolution "
                            f"(eval_n={len(evaluations)})"
                        )
                        break
                    if _deadline_reached():
                        client.log(
                            f"[pool-miner] lease deadline reached during evaluation job={job_id}; "
                            f"submitting partial evals n={len(evaluations)}"
                        )
                        break
                    if not isinstance(algo, dict):
                        continue
                    try:
                        algo_id = int(algo.get("id"))
                    except Exception:
                        continue
                    dsl = str(algo.get("algorithm_dsl") or "")
                    if not dsl.strip():
                        continue

                    if mode == "real":
                        from core.evaluations import score_algorithm_on_eval_suite

                        input_dim = int(algo.get("input_dim") or 16)
                        score = float(score_algorithm_on_eval_suite(dsl, input_dim=input_dim))
                    else:
                        score = float(_mock_score(algo_id, dsl, salt=salt))

                    evaluations.append({"algorithm_id": algo_id, "score": score})
                    lease_iterations += 1

                # Prefer evolving from the same batch that was just evaluated so
                # child-vs-batch-best comparisons are apples-to-apples.
                eval_algo_by_id: Dict[int, Dict[str, Any]] = {}
                for algo in eval_algorithms:
                    if not isinstance(algo, dict):
                        continue
                    try:
                        aid = int(algo.get("id"))
                    except Exception:
                        continue
                    eval_algo_by_id[aid] = algo
                ranked_eval_parents: List[Dict[str, Any]] = []
                for ev in sorted(evaluations, key=lambda row: float(row.get("score", 0.0)), reverse=True):
                    try:
                        aid = int(ev.get("algorithm_id"))
                    except Exception:
                        continue
                    algo = eval_algo_by_id.get(aid)
                    if algo is not None:
                        ranked_eval_parents.append(algo)
                parent_pool = ranked_eval_parents[: max(2, min(8, len(ranked_eval_parents)))]
                if not parent_pool:
                    parent_pool = [a for a in eval_algorithms if isinstance(a, dict)]
                if not parent_pool:
                    parent_pool = [a for a in seed_algorithms if isinstance(a, dict)]

                evolutions: List[Dict[str, Any]] = []
                if evolve_budget > 0:
                    if _deadline_reached():
                        client.log(
                            f"[pool-miner] lease deadline reached before evolution job={job_id}; skipping evolution"
                        )
                        evolve_budget = 0
                if evolve_budget > 0:
                    target_evolutions = max(0, int(evolve_budget))
                    parents = parent_pool
                    parent_ids: List[int] = []
                    for p in parents:
                        if not isinstance(p, dict) or p.get("id") is None:
                            continue
                        try:
                            parent_ids.append(int(p.get("id")))
                        except Exception:
                            continue
                    if parent_ids and target_evolutions > 0:
                        try:
                            input_dim = int(payload.get("input_dim") or 16)
                        except Exception:
                            input_dim = 16
                        if not input_dim:
                            try:
                                input_dim = int((parents[0] or {}).get("input_dim") or 16) if parents else 16
                            except Exception:
                                input_dim = 16

                        if mode == "real":
                            from core.dsl_parser import DSLParser
                            from core.tasks.cifar10 import CIFAR10BinaryTask
                            from miner.engines.archive_engine import ArchiveAwareBaselineEvolution

                            task = CIFAR10BinaryTask()
                            task.load_data(task_id=0)
                            engine = ArchiveAwareBaselineEvolution(task=task, pop_size=5, verbose=False)
                            seed_population: List[Any] = []
                            for parent in parents:
                                if not isinstance(parent, dict):
                                    continue
                                dsl = str(parent.get("algorithm_dsl") or "")
                                if not dsl.strip():
                                    continue
                                try:
                                    algo = DSLParser.from_dsl(dsl, input_dim)
                                except Exception:
                                    continue
                                seed_population.append(algo)
                                for _ in range(2):
                                    try:
                                        algo = engine._random_mutate(algo)  # type: ignore[attr-defined]
                                        seed_population.append(algo)
                                    except Exception:
                                        pass
                            if seed_population:
                                rng.shuffle(seed_population)
                                engine.population = list(
                                    seed_population[: max(1, int(getattr(engine, "pop_size", 5)))]
                                )
                            gen_limit = max(1, int(lease_evolve_generations))
                            gens_run = 0
                            seen_dsl: set[str] = set()
                            checkpoint = max(1, gen_limit // max(1, target_evolutions))
                            if submit_deadline_s is not None:
                                while (not stop.is_set()) and gens_run < gen_limit and (not _deadline_reached()):
                                    engine.evolve_generation()
                                    gens_run += 1
                                    should_snapshot = (
                                        gens_run == 1
                                        or (gens_run % checkpoint) == 0
                                        or gens_run >= gen_limit
                                    )
                                    if not should_snapshot:
                                        continue
                                    best_algo = engine.best_algo
                                    if best_algo is None:
                                        continue
                                    try:
                                        evolved_dsl = DSLParser.to_dsl(best_algo)
                                    except Exception:
                                        continue
                                    if not evolved_dsl.strip() or evolved_dsl in seen_dsl:
                                        continue
                                    evolutions.append(
                                        {
                                            "parent_algorithm_ids": parent_ids,
                                            "algorithm_dsl": str(evolved_dsl),
                                        }
                                    )
                                    seen_dsl.add(evolved_dsl)
                                    if len(evolutions) >= target_evolutions:
                                        break
                                if gens_run < gen_limit:
                                    client.log(
                                        f"[pool-miner] lease evolve truncated job={job_id} "
                                        f"gens={gens_run}/{gen_limit} "
                                        f"(submit buffer={effective_submit_buffer_s:.1f}s)"
                                    )
                            else:
                                while (not stop.is_set()) and gens_run < gen_limit:
                                    engine.evolve_generation()
                                    gens_run += 1
                                    should_snapshot = (
                                        gens_run == 1
                                        or (gens_run % checkpoint) == 0
                                        or gens_run >= gen_limit
                                    )
                                    if not should_snapshot:
                                        continue
                                    best_algo = engine.best_algo
                                    if best_algo is None:
                                        continue
                                    try:
                                        evolved_dsl = DSLParser.to_dsl(best_algo)
                                    except Exception:
                                        continue
                                    if not evolved_dsl.strip() or evolved_dsl in seen_dsl:
                                        continue
                                    evolutions.append(
                                        {
                                            "parent_algorithm_ids": parent_ids,
                                            "algorithm_dsl": str(evolved_dsl),
                                        }
                                    )
                                    seen_dsl.add(evolved_dsl)
                                    if len(evolutions) >= target_evolutions:
                                        break

                            # If best snapshots did not produce enough unique DSLs,
                            # mutate evaluated parents directly to fill remaining budget.
                            attempts = 0
                            max_attempts = max(20, target_evolutions * 20)
                            while (
                                len(evolutions) < target_evolutions
                                and (not _deadline_reached())
                                and attempts < max_attempts
                            ):
                                attempts += 1
                                parent = rng.choice(parents) if parents else None
                                if not isinstance(parent, dict):
                                    break
                                parent_dsl = str(parent.get("algorithm_dsl") or "")
                                if not parent_dsl.strip():
                                    continue
                                try:
                                    algo = DSLParser.from_dsl(parent_dsl, input_dim)
                                except Exception:
                                    continue
                                for _ in range(3):
                                    try:
                                        algo = engine._random_mutate(algo)  # type: ignore[attr-defined]
                                    except Exception:
                                        pass
                                try:
                                    evolved_dsl = DSLParser.to_dsl(algo)
                                except Exception:
                                    continue
                                if not evolved_dsl.strip() or evolved_dsl in seen_dsl:
                                    continue
                                evolutions.append(
                                    {
                                        "parent_algorithm_ids": parent_ids,
                                        "algorithm_dsl": str(evolved_dsl),
                                    }
                                )
                                seen_dsl.add(evolved_dsl)

                            lease_iterations += max(0, int(gens_run))
                        else:
                            gens_run = max(1, int(lease_evolve_generations))
                            lease_iterations += max(0, int(gens_run))
                            for _ in range(target_evolutions):
                                if _deadline_reached():
                                    break
                                evolved_dsl = _mock_evolve(parents, input_dim=input_dim, rng=rng)
                                evolutions.append(
                                    {
                                        "parent_algorithm_ids": parent_ids,
                                        "algorithm_dsl": str(evolved_dsl),
                                    }
                                )

                result = {
                    "evaluations": evaluations,
                    "evolutions": evolutions,
                    "iterations": int(lease_iterations),
                }
                client.submit_result(job_id, ok=True, result=result, error=None)
                client.log(
                    f"[pool-miner] lease job={job_id} eval_n={len(evaluations)} "
                    f"evo_n={len(evolutions)} iter_n={int(lease_iterations)}"
                )

            else:
                client.submit_result(job_id, ok=False, result={}, error=f"unknown kind: {kind}")
                client.log(f"[pool-miner] unknown kind for job={job_id}: {kind}")

        except Exception as e:
            client.submit_result(job_id, ok=False, result={}, error=str(e))
            client.log(f"[pool-miner] job={job_id} error={e}")

        completed += 1
        now = _now_s()
        elapsed = max(1e-6, now - last_completed_s)
        rate = 1.0 / elapsed
        last_completed_s = now
        client.progress(completed, rate=rate)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pool compute worker that communicates via the local sidecar job queue.")
    parser.add_argument("--sidecar-url", default=os.getenv("BITSOTA_SIDECAR_URL", _default_sidecar_url()))
    parser.add_argument("--run-id", default=os.getenv("BITSOTA_SIDECAR_RUN_ID", "pool_run"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("BITSOTA_POOL_MINER_WORKERS", "1")))
    parser.add_argument("--poll-interval-s", type=float, default=0.25)
    parser.add_argument("--lease-seconds", type=float, default=120.0)
    parser.add_argument(
        "--lease-submit-buffer-s",
        type=float,
        default=float(os.getenv("BITSOTA_POOL_LEASE_SUBMIT_BUFFER_S", "45")),
    )
    parser.add_argument(
        "--lease-evolve-reserve-s",
        type=float,
        default=float(os.getenv("BITSOTA_POOL_LEASE_EVOLVE_RESERVE_S", "90")),
    )
    env_mode = str(os.getenv("BITSOTA_POOL_MINER_MODE", "real") or "").strip().lower()
    if env_mode not in {"mock", "real"}:
        env_mode = "real"
    parser.add_argument("--mode", choices=["mock", "real"], default=env_mode)
    parser.add_argument("--evolve-generations", type=int, default=5)
    parser.add_argument("--lease-evolve-generations", type=int, default=160)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sidecar_url = str(args.sidecar_url).rstrip("/")
    run_id = str(args.run_id).strip()
    if not run_id:
        raise SystemExit("run-id must be non-empty")

    workers = max(1, int(args.workers))
    poll_interval_s = max(0.01, float(args.poll_interval_s))
    lease_seconds = max(1.0, float(args.lease_seconds))
    lease_submit_buffer_s = max(0.0, float(args.lease_submit_buffer_s))
    lease_evolve_reserve_s = max(0.0, float(args.lease_evolve_reserve_s))
    mode = str(args.mode)
    evolve_generations = max(1, int(args.evolve_generations))
    lease_evolve_generations = max(1, int(args.lease_evolve_generations))

    stop = threading.Event()
    threads: List[threading.Thread] = []
    for i in range(workers):
        wid = str(i)
        t = threading.Thread(
            target=_run_worker,
            kwargs={
                "sidecar_url": sidecar_url,
                "run_id": run_id,
                "worker_id": wid,
                "poll_interval_s": poll_interval_s,
                "lease_seconds": lease_seconds,
                "lease_submit_buffer_s": lease_submit_buffer_s,
                "lease_evolve_reserve_s": lease_evolve_reserve_s,
                "mode": mode,
                "evolve_generations": evolve_generations,
                "lease_evolve_generations": lease_evolve_generations,
                "seed": None if args.seed is None else int(args.seed) + i,
                "stop": stop,
            },
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

import requests


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
                try:
                    evolve_budget = int(payload.get("evolve_budget") or 0)
                except Exception:
                    evolve_budget = 0

                evaluations: List[Dict[str, Any]] = []
                for algo in eval_algorithms:
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

                evolutions: List[Dict[str, Any]] = []
                if evolve_budget > 0:
                    parents = seed_algorithms if seed_algorithms else eval_algorithms
                    parent_ids: List[int] = []
                    for p in parents:
                        if not isinstance(p, dict) or p.get("id") is None:
                            continue
                        try:
                            parent_ids.append(int(p.get("id")))
                        except Exception:
                            continue
                    if parent_ids:
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
                                for _ in range(2):
                                    try:
                                        engine._random_mutate(algo)  # type: ignore[attr-defined]
                                    except Exception:
                                        pass
                            best_algo, _ = engine.evolve(generations=max(1, int(lease_evolve_generations)))
                            evolved_dsl = (
                                DSLParser.to_dsl(best_algo)
                                if best_algo is not None
                                else _mock_evolve(parents, input_dim=input_dim, rng=rng)
                            )
                        else:
                            evolved_dsl = _mock_evolve(parents, input_dim=input_dim, rng=rng)

                        evolutions.append(
                            {
                                "parent_algorithm_ids": parent_ids,
                                "algorithm_dsl": str(evolved_dsl),
                            }
                        )

                result = {"evaluations": evaluations, "evolutions": evolutions}
                client.submit_result(job_id, ok=True, result=result, error=None)
                client.log(f"[pool-miner] lease job={job_id} eval_n={len(evaluations)} evo_n={len(evolutions)}")

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
    env_mode = str(os.getenv("BITSOTA_POOL_MINER_MODE", "real") or "").strip().lower()
    if env_mode not in {"mock", "real"}:
        env_mode = "real"
    parser.add_argument("--mode", choices=["mock", "real"], default=env_mode)
    parser.add_argument("--evolve-generations", type=int, default=5)
    parser.add_argument("--lease-evolve-generations", type=int, default=1000)
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

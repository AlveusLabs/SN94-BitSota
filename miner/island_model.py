from __future__ import annotations

import os
import queue as queue_module
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.dsl_parser import DSLParser


_BLAS_THREAD_ENV_VARS: Tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def set_blas_thread_env(workers: int, *, threads: int = 1) -> None:
    """Limit BLAS/OpenMP thread pools when using multiple worker processes."""

    if int(workers) <= 1:
        return
    threads = max(1, int(threads))
    for key in _BLAS_THREAD_ENV_VARS:
        if os.environ.get(key) is None:
            os.environ[key] = str(threads)


def seed_worker_rng(base_seed: Optional[int], worker_id: int) -> None:
    """Seed python and numpy global RNGs for a worker process."""

    if base_seed is None:
        return
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _sample_indices(pop_size: int, count: int) -> List[int]:
    pop_size = int(pop_size)
    count = int(count)
    if pop_size <= 0 or count <= 0:
        return []
    if count >= pop_size:
        return list(range(pop_size))
    indices = np.random.choice(pop_size, size=count, replace=False)
    return [int(i) for i in np.atleast_1d(indices)]


def migration_size_for_engine(engine: object) -> int:
    pop_size = int(getattr(engine, "pop_size", 0) or 0)
    return max(0, pop_size // 2)


def serialize_migrant(engine: object, algo: object) -> Dict[str, object]:
    """Serialize an individual for cross-process migration (prefer DSL)."""

    input_dim = int(getattr(algo, "input_dim", getattr(engine, "task", None) and getattr(engine.task, "input_dim", 0) or 0) or 0)  # type: ignore[attr-defined]
    dsl = DSLParser.to_dsl(algo)  # type: ignore[arg-type]
    return {"format": "dsl", "dsl": dsl, "input_dim": input_dim}


def deserialize_migrant(engine: object, payload: object) -> object:
    """Reconstruct a migrated individual inside the receiving process."""

    if isinstance(payload, dict):
        fmt = payload.get("format")
        if fmt == "dsl":
            dsl = str(payload.get("dsl") or "")
            input_dim = payload.get("input_dim")
            if input_dim is None:
                task = getattr(engine, "task", None)
                input_dim = getattr(task, "input_dim", None) if task is not None else None
            if input_dim is None:
                raise ValueError("Cannot deserialize migrant DSL without input_dim")
            return DSLParser.from_dsl(dsl, int(input_dim))
        return payload.get("algo")

    if isinstance(payload, str):
        task = getattr(engine, "task", None)
        input_dim = getattr(task, "input_dim", None) if task is not None else None
        if input_dim is None:
            raise ValueError("Cannot deserialize migrant DSL without input_dim")
        return DSLParser.from_dsl(payload, int(input_dim))

    return payload


def export_migrants(engine: object, migrate_count: int) -> Tuple[List[int], List[object]]:
    migrate_count = int(migrate_count)
    if migrate_count <= 0:
        return [], []

    if hasattr(engine, "_population_queue") and getattr(engine, "_population_queue", None) is not None:
        queue_list = list(getattr(engine, "_population_queue"))
        indices = _sample_indices(len(queue_list), migrate_count)
        migrants = [
            serialize_migrant(engine, queue_list[idx]["algo"])
            for idx in indices
            if 0 <= idx < len(queue_list)
        ]
        return indices, migrants

    population = list(getattr(engine, "population", []) or [])
    indices = _sample_indices(len(population), migrate_count)
    migrants = [
        serialize_migrant(engine, population[idx])
        for idx in indices
        if 0 <= idx < len(population)
    ]
    return indices, migrants


def apply_migrants(engine: object, replace_indices: Sequence[int], incoming: Sequence[object]) -> None:
    if not replace_indices or not incoming:
        return

    if hasattr(engine, "_population_queue") and getattr(engine, "_population_queue", None) is not None:
        queue_list = list(getattr(engine, "_population_queue"))
        for idx, payload in zip(replace_indices, incoming):
            if idx < 0 or idx >= len(queue_list):
                continue
            algo = deserialize_migrant(engine, payload)
            try:
                fitness = engine._evaluate_on_miner_tasks(algo)  # type: ignore[attr-defined]
            except Exception:
                fitness = -np.inf
            queue_list[idx] = {"algo": algo, "fitness": fitness}

        pop_size = int(getattr(engine, "pop_size", len(queue_list)) or len(queue_list))
        setattr(engine, "_population_queue", deque(queue_list, maxlen=pop_size))
        setattr(engine, "population", [entry["algo"] for entry in getattr(engine, "_population_queue")])

        try:
            best_entry = max(queue_list, key=lambda entry: float(entry.get("fitness", -np.inf)))
        except Exception:
            best_entry = None
        if best_entry is not None:
            try:
                best_fit = float(best_entry.get("fitness", -np.inf))
            except Exception:
                best_fit = -np.inf
            if getattr(engine, "best_algo", None) is None or best_fit > float(
                getattr(engine, "best_fitness", -np.inf)
            ):
                setattr(engine, "best_fitness", best_fit)
                setattr(engine, "best_algo", best_entry.get("algo"))
        return

    population = list(getattr(engine, "population", []) or [])
    if not population:
        return
    for idx, payload in zip(replace_indices, incoming):
        if idx < 0 or idx >= len(population):
            continue
        population[idx] = deserialize_migrant(engine, payload)
    setattr(engine, "population", population)


@dataclass
class MigrationRequest:
    worker_id: int
    iteration: int
    migrants: List[object]


class MigrationCoordinator:
    """Collect per-worker migrants and mix them once all workers reach a migration point."""

    def __init__(self, *, workers: int, seed: Optional[int] = None):
        self.workers = max(1, int(workers))
        self._rng = random.Random(seed)
        self._pending_iteration: Optional[int] = None
        self._pending: Dict[int, List[object]] = {}
        self._pending_counts: Dict[int, int] = {}

    def add_request(
        self, *, worker_id: int, iteration: int, migrants: List[object]
    ) -> Optional[Dict[int, List[object]]]:
        worker_id = int(worker_id)
        iteration = int(iteration)
        migrants = list(migrants or [])

        if self._pending_iteration is None:
            self._pending_iteration = iteration

        if iteration != int(self._pending_iteration):
            raise ValueError(
                f"migration desync: got iter {iteration} expected {int(self._pending_iteration)}"
            )

        self._pending[worker_id] = migrants
        self._pending_counts[worker_id] = len(migrants)

        if len(self._pending) < self.workers:
            return None

        pool: List[object] = []
        for wid in range(self.workers):
            pool.extend(self._pending.get(wid, []))
        self._rng.shuffle(pool)

        out: Dict[int, List[object]] = {}
        cursor = 0
        for wid in range(self.workers):
            k = int(self._pending_counts.get(wid, 0) or 0)
            out[wid] = pool[cursor : cursor + k]
            cursor += k

        self._pending_iteration = None
        self._pending.clear()
        self._pending_counts.clear()
        return out


class IslandEngineWrapper:
    """Wrap an evolution engine to perform periodic island-model migration."""

    def __init__(
        self,
        engine: object,
        *,
        worker_id: int,
        migration_generations: int,
        out_queue: object,
        in_queue: object,
        stop_event: object,
    ):
        self._engine = engine
        self._worker_id = int(worker_id)
        self._migration_generations = max(0, int(migration_generations))
        self._out_queue = out_queue
        self._in_queue = in_queue
        self._stop_event = stop_event
        self._iteration = 0

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    def evolve_generation(self, *args, **kwargs):
        result = self._engine.evolve_generation(*args, **kwargs)
        self._iteration += 1

        if (
            self._migration_generations > 0
            and self._iteration % self._migration_generations == 0
            and not bool(getattr(self._stop_event, "is_set", lambda: False)())
        ):
            migrate_count = migration_size_for_engine(self._engine)
            replace_indices, migrants = export_migrants(self._engine, migrate_count)
            if migrants:
                self._out_queue.put(
                    {
                        "type": "migration_request",
                        "worker_id": self._worker_id,
                        "iteration": int(self._iteration),
                        "migrants": migrants,
                    }
                )
                while not bool(getattr(self._stop_event, "is_set", lambda: False)()):
                    try:
                        resp = self._in_queue.get(timeout=0.5)
                    except queue_module.Empty:
                        continue
                    if (
                        isinstance(resp, dict)
                        and resp.get("type") == "migration_response"
                        and int(resp.get("iteration", -1)) == int(self._iteration)
                    ):
                        incoming = resp.get("incoming") or []
                        apply_migrants(self._engine, replace_indices, list(incoming))
                        break

        return result


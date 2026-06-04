from __future__ import annotations

import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.dsl_parser import DSLParser
from core.evaluations import TASK_REGISTRY as TASKS_BY_TYPE
from core.evaluations import verify_solution_quality
from miner.engines.archive_engine import ArchiveAwareBaselineEvolution
from miner.engines.base_engine import BaseEvolutionEngine, DEFAULT_MINER_TASK_COUNT
from miner.engines.ga_engine import BaselineEvolutionEngine
from miner.island_model import IslandEngineWrapper, seed_worker_rng
from miner.state_store import MinerStateStore

logger = logging.getLogger("miner.local_benchmark")


def _resolve_state_dir(raw: object) -> Optional[Path]:
    if raw is None:
        return None
    try:
        return Path(str(raw)).expanduser().resolve()
    except Exception:
        return None


def _load_best_verified_from_state(store: MinerStateStore, *, task_type: str) -> Optional[float]:
    payload = store.load_client_state()
    if not payload:
        return None
    lbs = payload.get("local_best_verified_score")
    if not isinstance(lbs, dict):
        return None
    raw = lbs.get(str(task_type))
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _persist_best_verified(
    store: MinerStateStore,
    *,
    task_type: str,
    verified_score: float,
    total_submissions: int = 0,
    total_sota_breaks: int = 0,
) -> None:
    existing = store.load_client_state() or {}
    lbs = existing.get("local_best_verified_score")
    if not isinstance(lbs, dict):
        lbs = {}
    lbs[str(task_type)] = float(verified_score)
    try:
        store.save_client_state(
            {
                "local_best_verified_score": dict(lbs),
                "total_submissions": int(existing.get("total_submissions", total_submissions) or 0),
                "total_sota_breaks": int(existing.get("total_sota_breaks", total_sota_breaks) or 0),
            }
        )
    except Exception:
        return


def _create_engine(
    *,
    task_type: str,
    engine_type: str,
    engine_params: Optional[Dict[str, Any]],
    miner_task_count: Optional[int],
    fec_cache_size: Optional[int] = None,
    fec_train_examples: Optional[int] = None,
    fec_valid_examples: Optional[int] = None,
    fec_forget_every: Optional[int] = None,
    engine_verbose: bool = False,
    store: Optional[MinerStateStore] = None,
    worker_id: int = 0,
) -> BaseEvolutionEngine:
    task_cls = TASKS_BY_TYPE.get(str(task_type))
    if not task_cls:
        raise ValueError(f"Unknown task type: {task_type}")

    task = task_cls()
    task.load_data()

    params = dict(engine_params) if isinstance(engine_params, dict) else {}

    engine_kwargs: Dict[str, Any] = {}
    pop_size = params.get("pop_size")
    if pop_size is not None:
        try:
            engine_kwargs["pop_size"] = max(1, int(pop_size))
        except Exception:
            pass

    if str(engine_type) == "baseline":
        for key in ("tournament_size", "mutation_prob"):
            if key not in params:
                continue
            try:
                if key == "tournament_size":
                    engine_kwargs[key] = max(1, int(params[key]))
                else:
                    engine_kwargs[key] = float(params[key])
            except Exception:
                continue

    if str(engine_type) == "archive":
        archive_size = params.get("archive_size")
        if archive_size is not None:
            try:
                engine_kwargs["archive_size"] = max(1, int(archive_size))
            except Exception:
                pass

    phase_max_sizes = params.get("phase_max_sizes")
    if isinstance(phase_max_sizes, dict):
        cleaned_phase_sizes: Dict[str, int] = {}
        for phase, size in phase_max_sizes.items():
            if not isinstance(phase, str):
                continue
            try:
                cleaned_phase_sizes[str(phase)] = max(1, int(size))
            except Exception:
                continue
        if cleaned_phase_sizes:
            engine_kwargs["phase_max_sizes"] = cleaned_phase_sizes

    for key_name in ("scalar_count", "vector_count", "matrix_count", "vector_dim", "cifar_seed"):
        value = params.get(key_name)
        if value is None:
            continue
        try:
            engine_kwargs[key_name] = int(value)
        except Exception:
            continue

    norm_task_count = max(1, int(miner_task_count or DEFAULT_MINER_TASK_COUNT))

    if str(engine_type) == "archive":
        engine: BaseEvolutionEngine = ArchiveAwareBaselineEvolution(
            task,
            **engine_kwargs,
            verbose=bool(engine_verbose),
            miner_task_count=norm_task_count,
            fec_cache_size=fec_cache_size,
            fec_train_examples=fec_train_examples,
            fec_valid_examples=fec_valid_examples,
            fec_forget_every=fec_forget_every,
        )
    elif str(engine_type) == "baseline":
        engine = BaselineEvolutionEngine(
            task,
            **engine_kwargs,
            verbose=bool(engine_verbose),
            miner_task_count=norm_task_count,
            fec_cache_size=fec_cache_size,
            fec_train_examples=fec_train_examples,
            fec_valid_examples=fec_valid_examples,
            fec_forget_every=fec_forget_every,
        )
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")

    if store is not None:
        try:
            restored = store.load_engine_state(
                task_type=str(task_type),
                engine_type=str(engine_type),
                engine=engine,
            )
            if restored:
                logger.info(
                    "Resumed local-benchmark engine state (worker_id=%d task_type=%s engine_type=%s)",
                    int(worker_id),
                    str(task_type),
                    str(engine_type),
                )
        except Exception:
            pass

    return engine


def run_local_benchmark_worker(
    worker_config: Dict[str, Any],
    worker_id: int,
    out_queue: Any,
    in_queue: Any,
    stop_event: Any,
) -> None:
    """
    Offline AutoML search worker used by the GUI and benchmark scripts.

    - No relay/wallet HTTP calls
    - Emits sparse queue messages:
        - {type: "stats", ...} at checkpoint intervals
        - {type: "best_verified", ...} when a new best verified score is found
        - {type: "done"} on exit
        - {type: "error"} on crash
    """

    worker_id = int(worker_id)
    try:
        env_overrides = worker_config.get("env_overrides")
        if isinstance(env_overrides, dict):
            for key, value in env_overrides.items():
                name = str(key).strip()
                if not name:
                    continue
                os.environ[name] = str(value)

        seed_worker_rng(worker_config.get("seed"), worker_id)

        task_type = str(worker_config.get("task_type") or "cifar10_binary")
        engine_type = str(worker_config.get("engine_type") or "baseline")

        checkpoint_generations = max(1, int(worker_config.get("checkpoint_generations") or 10))
        migration_generations = max(0, int(worker_config.get("migration_generations") or 0))

        miner_task_count = worker_config.get("miner_task_count")
        validator_task_count = worker_config.get("validator_task_count")
        try:
            validator_task_count_int = int(validator_task_count) if validator_task_count is not None else None
        except Exception:
            validator_task_count_int = None

        engine_params = worker_config.get("engine_params")
        engine_verbose = bool(worker_config.get("engine_verbose", False))

        fec_cache_size = worker_config.get("fec_cache_size")
        fec_train_examples = worker_config.get("fec_train_examples")
        fec_valid_examples = worker_config.get("fec_valid_examples")
        fec_forget_every = worker_config.get("fec_forget_every")

        public_address = str(worker_config.get("public_address") or "local")
        persist_state = worker_config.get("persist_state")
        if persist_state is None:
            raw = str(os.getenv("MINER_PERSIST_STATE", "1")).strip().lower()
            persist_state = raw not in {"0", "false", "no", "off"}
        persist_state = bool(persist_state)

        try:
            persist_every = int(
                worker_config.get("persist_every_n_generations")
                or os.getenv("MINER_PERSIST_EVERY_N_GENERATIONS", "5000")
            )
        except Exception:
            persist_every = 5000
        persist_every = max(0, int(persist_every))

        state_dir = _resolve_state_dir(worker_config.get("state_dir"))
        store = MinerStateStore(public_address=public_address, worker_id=worker_id, state_dir=state_dir)
        store_for_engine = store if persist_state else None

        base_engine = _create_engine(
            task_type=task_type,
            engine_type=engine_type,
            engine_params=engine_params if isinstance(engine_params, dict) else None,
            miner_task_count=miner_task_count,
            fec_cache_size=fec_cache_size,
            fec_train_examples=fec_train_examples,
            fec_valid_examples=fec_valid_examples,
            fec_forget_every=fec_forget_every,
            engine_verbose=engine_verbose,
            store=store_for_engine,
            worker_id=worker_id,
        )

        engine: Any = base_engine
        if migration_generations > 0:
            engine = IslandEngineWrapper(
                base_engine,
                worker_id=worker_id,
                migration_generations=migration_generations,
                out_queue=out_queue,
                in_queue=in_queue,
                stop_event=stop_event,
            )

        sota_threshold_raw = worker_config.get("sota_threshold")
        sota_threshold: Optional[float]
        if sota_threshold_raw is None:
            sota_threshold = None
        else:
            try:
                sota_threshold = float(sota_threshold_raw)
            except Exception:
                sota_threshold = None

        validate_every_raw = worker_config.get("validate_every_n_generations")
        if validate_every_raw is None:
            validate_every_raw = os.getenv("MINER_VALIDATE_EVERY_N_GENERATIONS", "1")
        try:
            validate_every = max(1, int(validate_every_raw))
        except Exception:
            validate_every = 1

        best_verified_score = -np.inf
        if persist_state:
            restored = _load_best_verified_from_state(store, task_type=task_type)
            if restored is not None:
                best_verified_score = float(restored)

        last_validation_generation = int(getattr(base_engine, "generation", 0) or 0)
        last_validated_mining_score = -np.inf

        best_ever_score = float(getattr(base_engine, "best_fitness", -np.inf) or -np.inf)
        generations_since_improvement = 0

        start_time = time.time()
        last_stats_time = start_time
        last_stats_generation = int(getattr(base_engine, "generation", 0) or 0)

        # One-time init snapshot so aggregators can baseline generation counters even when resuming.
        try:
            out_queue.put(
                {
                    "type": "init",
                    "worker_id": int(worker_id),
                    "task_type": str(task_type),
                    "engine_type": str(engine_type),
                    "generation": int(last_stats_generation),
                    "best_verified_score": float(best_verified_score) if np.isfinite(best_verified_score) else None,
                    "sota_threshold": float(sota_threshold) if sota_threshold is not None else None,
                }
            )
        except Exception:
            pass

        while not bool(getattr(stop_event, "is_set", lambda: False)()):
            best_algo, best_score, population, scores = engine.evolve_generation()

            generation = int(getattr(base_engine, "generation", 0) or 0)
            mining_score = float(best_score)

            if mining_score > best_ever_score:
                best_ever_score = float(mining_score)
                generations_since_improvement = 0
            else:
                generations_since_improvement += 1

            if (
                persist_state
                and persist_every > 0
                and generation > 0
                and (generation % persist_every) == 0
            ):
                try:
                    store.save_engine_state(
                        task_type=str(task_type),
                        engine_type=str(engine_type),
                        engine=base_engine,
                    )
                except Exception:
                    pass

            # --- Validation / new best verified detection -----------------
            should_validate = False
            if mining_score > float(last_validated_mining_score):
                if generation - last_validation_generation >= validate_every:
                    should_validate = True
                elif sota_threshold is not None and mining_score > float(sota_threshold):
                    # Fast-path: validate immediately when a miner score clears the current bar.
                    should_validate = True

            if should_validate and best_algo is not None:
                last_validation_generation = generation
                last_validated_mining_score = float(mining_score)
                try:
                    algorithm_dsl = DSLParser.to_dsl(best_algo)
                    task = getattr(base_engine, "task", None)
                    input_dim = int(getattr(task, "input_dim", 0) or 0)

                    threshold = float(sota_threshold) if sota_threshold is not None else -np.inf
                    solution_data = {
                        "task_type": str(task_type),
                        "algorithm_dsl": algorithm_dsl,
                        "eval_score": float(mining_score),
                        "input_dim": input_dim,
                    }
                    passed, verified_score = verify_solution_quality(
                        solution_data,
                        threshold,
                        task_count=validator_task_count_int,
                    )
                    verified_score_f = float(verified_score)
                    if verified_score_f > float(best_verified_score):
                        best_verified_score = float(verified_score_f)
                        if persist_state:
                            _persist_best_verified(
                                store,
                                task_type=str(task_type),
                                verified_score=float(best_verified_score),
                            )
                        out_queue.put(
                            {
                                "type": "best_verified",
                                "worker_id": int(worker_id),
                                "task_type": str(task_type),
                                "engine_type": str(engine_type),
                                "generation": int(generation),
                                "mining_score": float(mining_score),
                                "verified_score": float(best_verified_score),
                                "sota_threshold": float(sota_threshold) if sota_threshold is not None else None,
                                "is_sota_breaker": bool(passed) if sota_threshold is not None else None,
                                "algorithm_dsl": algorithm_dsl,
                                "input_dim": int(input_dim),
                                "log": (
                                    f"Score: {float(best_verified_score):.4f} (verified) "
                                    f"Mining Score: {float(mining_score):.4f} Generation: {int(generation)}"
                                ),
                            }
                        )
                except Exception:
                    # Validation should never crash the search loop.
                    pass

            # --- Periodic stats ------------------------------------------
            if generation > 0 and (generation % checkpoint_generations) == 0:
                now = time.time()
                dt = float(now - last_stats_time)
                dg = int(generation - last_stats_generation)
                iters_per_sec = float(dg / dt) if dt > 0 else 0.0
                last_stats_time = now
                last_stats_generation = generation

                finite_scores = [float(s) for s in scores if np.isfinite(s) and float(s) != -np.inf]
                pop_mean = float(np.mean(finite_scores)) if finite_scores else -np.inf
                pop_max = float(np.max(finite_scores)) if finite_scores else -np.inf
                distance_to_sota = None
                if sota_threshold is not None and finite_scores:
                    distance_to_sota = float(sota_threshold) - float(pop_max)

                log_line = (
                    f"Gen {generation}: best_ever={best_ever_score:.4f}, "
                    f"current_best={mining_score:.4f}, pop_mean={pop_mean:.4f}, "
                    f"distance_to_sota={distance_to_sota:.4f}, "
                    f"stagnation={generations_since_improvement}, "
                    f"iters_per_sec={iters_per_sec:.2f}"
                    if distance_to_sota is not None
                    else (
                        f"Gen {generation}: best_ever={best_ever_score:.4f}, "
                        f"current_best={mining_score:.4f}, pop_mean={pop_mean:.4f}, "
                        f"stagnation={generations_since_improvement}, "
                        f"iters_per_sec={iters_per_sec:.2f}"
                    )
                )

                out_queue.put(
                    {
                        "type": "stats",
                        "worker_id": int(worker_id),
                        "task_type": str(task_type),
                        "engine_type": str(engine_type),
                        "generation": int(generation),
                        "iters_per_sec": float(iters_per_sec),
                        "best_ever_score": float(best_ever_score),
                        "current_best_score": float(mining_score),
                        "population_mean": float(pop_mean),
                        "population_best": float(pop_max),
                        "distance_to_sota": float(distance_to_sota) if distance_to_sota is not None else None,
                        "stagnation": int(generations_since_improvement),
                        "best_verified_score": float(best_verified_score) if np.isfinite(best_verified_score) else None,
                        "sota_threshold": float(sota_threshold) if sota_threshold is not None else None,
                        "log": log_line,
                    }
                )

        out_queue.put({"type": "done", "worker_id": int(worker_id), "result": {"status": "stopped"}})
    except KeyboardInterrupt:
        try:
            out_queue.put({"type": "done", "worker_id": int(worker_id), "result": {"status": "stopped"}})
        except Exception:
            pass
    except Exception:
        try:
            out_queue.put(
                {
                    "type": "error",
                    "worker_id": int(worker_id),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        try:
            if hasattr(stop_event, "set"):
                stop_event.set()
        except Exception:
            pass

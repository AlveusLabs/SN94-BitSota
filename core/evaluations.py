"""Evaluation utilities shared between miners and validators."""

import json
import logging
import os
from typing import Dict, Any, Optional, Tuple

import numpy as np

from core.dsl_parser import DSLParser
from core.tasks.cifar10 import CIFAR10BinaryTask

logger = logging.getLogger(__name__)

DEFAULT_TASK_TYPE = "cifar10_binary"
DEFAULT_CIFAR_INPUT_DIM = 16
VALIDATOR_TASK_COUNT = int(os.getenv("VALIDATOR_TASK_COUNT", "128"))
VALIDATOR_TASK_SEED = int(os.getenv("VALIDATOR_TASK_SEED", "1337"))

_VALIDATOR_TASK_CACHE: Dict[Tuple[int, int], list] = {}

# Task registry mapping task names to their classes. We now fix validation to CIFAR-10.
TASK_REGISTRY = {
    DEFAULT_TASK_TYPE: CIFAR10BinaryTask,
}


def _load_cifar_task(
    requested_dim: Optional[int] = None,
    *,
    task_spec=None,
    preload: bool = True,
) -> CIFAR10BinaryTask:
    """Helper to create (and optionally load) a CIFAR-10 task instance."""

    task = CIFAR10BinaryTask()
    if preload:
        load_kwargs = {}
        if task_spec is not None:
            load_kwargs["task_spec"] = task_spec
        elif requested_dim:
            load_kwargs["n_components"] = requested_dim
        task.load_data(**load_kwargs)
    return task


def _get_validator_task_specs(input_dim: int):
    """Return a deterministic list of task specs for validator scoring."""

    count = max(1, VALIDATOR_TASK_COUNT)
    cache_key = (input_dim, count)
    if cache_key not in _VALIDATOR_TASK_CACHE:
        seed = VALIDATOR_TASK_SEED + 7919 * max(1, input_dim)
        specs = CIFAR10BinaryTask.sample_task_specs(
            num_tasks=count,
            replace=False,
            seed=seed,
            n_components=input_dim,
        )
        _VALIDATOR_TASK_CACHE[cache_key] = specs
    return _VALIDATOR_TASK_CACHE[cache_key]


def verify_solution_quality(
    solution_data: Dict[str, Any], sota_threshold: float = None
) -> Tuple[bool, float]:
    """
    Verify that a submitted solution beats the global SOTA threshold using
    deterministic CIFAR-10 tasks.

    Args:
        solution_data: Dictionary containing:
            - algorithm_dsl: str - algorithm in DSL format
            - eval_score: float - optional pre-computed score
            - input_dim: int - optional projection dimension
        sota_threshold: float - global SOTA threshold to beat

    Returns:
        Tuple[bool, float]: (passed_threshold, validation_score)
    """

    task = None
    try:
        algorithm_dsl = solution_data.get("algorithm_dsl")
        if not algorithm_dsl:
            logger.warning("Missing required field `algorithm_dsl` in solution data")
            return False, -np.inf

        requested_dim = solution_data.get("input_dim")
        if requested_dim:
            input_dim = int(requested_dim)
        else:
            input_dim = DEFAULT_CIFAR_INPUT_DIM
        task = _load_cifar_task(input_dim, preload=False)

        try:
            algorithm = DSLParser.from_dsl(algorithm_dsl, input_dim)
        except Exception as e:
            logger.warning("Failed to parse algorithm DSL: %s", e)
            return False, -np.inf

        task_specs = _get_validator_task_specs(input_dim)
        if not task_specs:
            raise RuntimeError("Validator task list is empty")

        scores = []
        task_results = []
        for spec in task_specs:
            task.load_data(task_spec=spec)
            task_score = float(task.evaluate_algorithm(algorithm, epochs=1))
            scores.append(task_score)
            task_results.append(
                {
                    "task_id": int(getattr(spec, "task_id", -1)),
                    "class_pair": list(getattr(spec, "class_pair", ())),
                    "projection_seed": int(getattr(spec, "projection_seed", 0) or 0),
                    "split_seed": int(getattr(spec, "split_seed", 0) or 0),
                    "score": float(task_score),
                }
            )

        score = float(np.median(scores)) if scores else task.get_baseline_fitness()
        # Never allow untrusted submissions to trigger verbose per-task logging.
        # Enable explicitly via environment variable + DEBUG log level.
        log_all_task_scores = (
            str(os.getenv("LOG_VALIDATOR_TASK_SCORES", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if log_all_task_scores and logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug(
                    "Validator suite scores (n=%d median=%.6f): %s",
                    int(len(task_results)),
                    float(score),
                    json.dumps(task_results, separators=(",", ":"), sort_keys=False),
                )
            except Exception:
                logger.debug(
                    "Validator suite scores (n=%d median=%.6f): %s",
                    int(len(scores)),
                    float(score),
                    str(scores),
                )

        if sota_threshold is None:
            sota_threshold = 0.0

        return score >= sota_threshold, score

    except Exception as e:
        logger.exception("Error in verify_solution_quality: %s", e)
        fallback = task.get_baseline_fitness() if task else -np.inf
        return False, fallback


def score_algorithm_on_eval_suite(algorithm_dsl: str, input_dim: int = None) -> float:
    """
    Deterministically score an algorithm on the validator-style eval suite (median over tasks).
    This is the scoring policy intended for "evaluate" tasks.
    """

    if not algorithm_dsl:
        return -np.inf

    dim = int(input_dim) if input_dim else DEFAULT_CIFAR_INPUT_DIM
    task = _load_cifar_task(dim, preload=False)
    try:
        algorithm = DSLParser.from_dsl(algorithm_dsl, dim)
    except Exception as e:
        logger.warning("Failed to parse algorithm DSL: %s", e)
        return -np.inf

    task_specs = _get_validator_task_specs(dim)
    if not task_specs:
        return -np.inf

    scores = []
    for spec in task_specs:
        task.load_data(task_spec=spec)
        scores.append(float(task.evaluate_algorithm(algorithm, epochs=1)))
    return float(np.median(scores)) if scores else -np.inf


def get_task_benchmark(task_type: str) -> float:
    """Return the benchmark score for the CIFAR-10 binary task."""
    if task_type != DEFAULT_TASK_TYPE:
        raise ValueError("Only cifar10_binary benchmark is available")

    task = _load_cifar_task()
    return task.get_baseline_fitness()


def evaluate_algorithm_on_task(
    algorithm_dsl: str, task_type: str, input_dim: int = None
) -> Dict[str, Any]:
    """Evaluate an algorithm on the CIFAR-10 binary task."""
    if task_type != DEFAULT_TASK_TYPE:
        return {"error": f"Unknown task type: {task_type}"}

    try:
        task = _load_cifar_task(input_dim)
        actual_dim = input_dim or task.input_dim

        algorithm = DSLParser.from_dsl(algorithm_dsl, actual_dim)
        score = task.evaluate_algorithm(algorithm, epochs=1)
        baseline = task.get_baseline_fitness()

        return {
            "task_type": task_type,
            "score": float(score),
            "baseline": float(baseline),
            "beats_baseline": score > baseline if baseline != -np.inf else score > 0,
            "improvement": (
                float(score - baseline) if baseline != -np.inf else float(score)
            ),
        }

    except Exception as e:
        logger.exception("Algorithm evaluation failed")
        return {"error": str(e)}

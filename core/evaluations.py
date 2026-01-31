"""Evaluation utilities shared between miners and validators."""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from core.dsl_parser import DSLParser
from core.hyperparams import get_validator_hyperparams
from core.tasks.cifar10 import CIFAR10BinaryTask
from core.tasks.mnist import MNISTBinaryTask
from core.tasks.scalar_linear import ScalarLinearRegressionTask

logger = logging.getLogger(__name__)

DEFAULT_TASK_TYPE = "cifar10_binary"
DEFAULT_CIFAR_INPUT_DIM = 16

# Backward-compatible module-level defaults (used by scripts and legacy callers).
# Prefer `validator_hyperparams.json` over env vars when setting these values.
_DEFAULT_VALIDATOR_HP = get_validator_hyperparams()
VALIDATOR_TASK_COUNT = int(_DEFAULT_VALIDATOR_HP.task_count)
VALIDATOR_TASK_SEED = int(_DEFAULT_VALIDATOR_HP.task_seed)

_VALIDATOR_CIFAR_TASK_CACHE: Dict[Tuple[int, int, int, int, int], list] = {}
_VALIDATOR_MNIST_TASK_CACHE: Dict[Tuple[int, int, int, int, int], list] = {}
_VALIDATOR_SCALAR_TASK_CACHE: Dict[Tuple[int, int, int, int, int], list] = {}

TASK_REGISTRY = {
    DEFAULT_TASK_TYPE: CIFAR10BinaryTask,
    "mnist_binary": MNISTBinaryTask,
    "scalar_linear": ScalarLinearRegressionTask,
}

def _eval_seed_for_task_spec(spec: object, *, suite_seed: int) -> int:
    """Return a stable RNG seed for ArrayExecutor given a task spec."""

    task_id = int(getattr(spec, "task_id", 0) or 0)
    projection_seed = int(getattr(spec, "projection_seed", 0) or 0)
    split_seed = int(getattr(spec, "split_seed", 0) or 0)
    param_seed = int(getattr(spec, "param_seed", 0) or 0)
    data_seed = int(getattr(spec, "data_seed", 0) or 0)
    seed = (
        int(suite_seed)
        + 15485863 * (task_id + 1)
        + 104729 * (projection_seed + 1)
        + 1000003 * (split_seed + 1)
        + 1618033 * (param_seed + 1)
        + 314159 * (data_seed + 1)
    )
    seed = int(seed % (2**31 - 1))
    return seed if seed > 0 else 1


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


def _get_cifar_validator_task_specs(
    input_dim: int,
    *,
    task_count: Optional[int] = None,
    task_seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    train_split: Optional[float] = None,
):
    """Return a deterministic list of task specs for validator scoring."""

    count = max(1, int(task_count if task_count is not None else VALIDATOR_TASK_COUNT))
    base_seed = int(task_seed if task_seed is not None else VALIDATOR_TASK_SEED)

    norm_n_samples = None
    if n_samples is not None:
        norm_n_samples = max(1, int(n_samples))

    norm_train_split = None
    if train_split is not None:
        norm_train_split = float(train_split)
        if not (0.0 < norm_train_split < 1.0):
            raise ValueError("train_split must be in (0, 1)")

    seed = base_seed + 7919 * max(1, int(input_dim))
    train_split_key = -1 if norm_train_split is None else int(round(norm_train_split * 1_000_000))
    n_samples_key = -1 if norm_n_samples is None else int(norm_n_samples)
    cache_key = (int(input_dim), count, int(seed), n_samples_key, train_split_key)
    if cache_key not in _VALIDATOR_CIFAR_TASK_CACHE:
        sample_kwargs: Dict[str, Any] = {
            "num_tasks": count,
            "replace": False,
            "seed": seed,
            "n_components": int(input_dim),
        }
        if norm_n_samples is not None:
            sample_kwargs["n_samples"] = int(norm_n_samples)
        if norm_train_split is not None:
            sample_kwargs["train_split"] = float(norm_train_split)

        specs = CIFAR10BinaryTask.sample_task_specs(**sample_kwargs)
        _VALIDATOR_CIFAR_TASK_CACHE[cache_key] = specs
    return _VALIDATOR_CIFAR_TASK_CACHE[cache_key]


def _get_mnist_validator_task_specs(
    input_dim: int,
    *,
    task_count: Optional[int] = None,
    task_seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    train_split: Optional[float] = None,
):
    count = max(1, int(task_count if task_count is not None else VALIDATOR_TASK_COUNT))
    base_seed = int(task_seed if task_seed is not None else VALIDATOR_TASK_SEED)

    norm_n_samples = None
    if n_samples is not None:
        norm_n_samples = max(1, int(n_samples))

    norm_train_split = None
    if train_split is not None:
        norm_train_split = float(train_split)
        if not (0.0 < norm_train_split < 1.0):
            raise ValueError("train_split must be in (0, 1)")

    seed = base_seed + 7919 * max(1, int(input_dim))
    train_split_key = -1 if norm_train_split is None else int(round(norm_train_split * 1_000_000))
    n_samples_key = -1 if norm_n_samples is None else int(norm_n_samples)
    cache_key = (int(input_dim), count, int(seed), n_samples_key, train_split_key)
    if cache_key not in _VALIDATOR_MNIST_TASK_CACHE:
        sample_kwargs: Dict[str, Any] = {
            "num_tasks": count,
            "replace": False,
            "seed": seed,
            "n_components": int(input_dim),
        }
        if norm_n_samples is not None:
            sample_kwargs["n_samples"] = int(norm_n_samples)
        if norm_train_split is not None:
            sample_kwargs["train_split"] = float(norm_train_split)
        _VALIDATOR_MNIST_TASK_CACHE[cache_key] = MNISTBinaryTask.sample_task_specs(**sample_kwargs)
    return _VALIDATOR_MNIST_TASK_CACHE[cache_key]


def _get_scalar_validator_task_specs(
    input_dim: int,
    *,
    task_count: Optional[int] = None,
    task_seed: Optional[int] = None,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
):
    count = max(1, int(task_count if task_count is not None else VALIDATOR_TASK_COUNT))
    base_seed = int(task_seed if task_seed is not None else VALIDATOR_TASK_SEED)

    ts = None if train_samples is None else max(1, int(train_samples))
    vs = None if val_samples is None else max(1, int(val_samples))

    seed = base_seed + 7919 * max(1, int(input_dim))
    cache_key = (int(input_dim), count, int(seed), int(ts or -1), int(vs or -1))
    if cache_key not in _VALIDATOR_SCALAR_TASK_CACHE:
        sample_kwargs: Dict[str, Any] = {
            "num_tasks": count,
            "replace": False,
            "seed": seed,
            "n_features": int(input_dim),
        }
        if ts is not None:
            sample_kwargs["train_samples"] = int(ts)
        if vs is not None:
            sample_kwargs["val_samples"] = int(vs)
        _VALIDATOR_SCALAR_TASK_CACHE[cache_key] = ScalarLinearRegressionTask.sample_task_specs(**sample_kwargs)
    return _VALIDATOR_SCALAR_TASK_CACHE[cache_key]


def verify_solution_quality(
    solution_data: Dict[str, Any],
    sota_threshold: float = None,
    *,
    epochs: Optional[int] = None,
    task_count: Optional[int] = None,
    task_seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    train_split: Optional[float] = None,
) -> Tuple[bool, float]:
    """
    Verify that a submitted solution beats the global SOTA threshold using
    deterministic validator task suites.

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
        hp = get_validator_hyperparams()

        algorithm_dsl = solution_data.get("algorithm_dsl")
        if not algorithm_dsl:
            logger.warning("Missing required field `algorithm_dsl` in solution data")
            return False, -np.inf

        default_task_type = str(hp.default_task_type or DEFAULT_TASK_TYPE).strip().lower()
        if default_task_type not in TASK_REGISTRY:
            default_task_type = DEFAULT_TASK_TYPE

        task_type = str(solution_data.get("task_type") or default_task_type).strip().lower()
        if task_type not in TASK_REGISTRY:
            logger.warning("Unknown task type in solution data: %s", task_type)
            return False, -np.inf

        requested_dim = solution_data.get("input_dim")
        if requested_dim:
            input_dim = int(requested_dim)
        else:
            try:
                input_dim = int(getattr(hp, "default_input_dim", DEFAULT_CIFAR_INPUT_DIM))
            except Exception:
                input_dim = DEFAULT_CIFAR_INPUT_DIM

        try:
            algorithm = DSLParser.from_dsl(algorithm_dsl, input_dim)
        except Exception as e:
            logger.warning("Failed to parse algorithm DSL: %s", e)
            return False, -np.inf

        if epochs is None:
            epochs = int(getattr(hp, "epochs", 1) or 1)
        epochs = max(1, int(epochs))

        if task_count is None:
            task_count = int(getattr(hp, "task_count", VALIDATOR_TASK_COUNT) or VALIDATOR_TASK_COUNT)
        if task_seed is None:
            task_seed = int(getattr(hp, "task_seed", VALIDATOR_TASK_SEED) or VALIDATOR_TASK_SEED)

        task_suite_overrides = hp.tasks.get(str(task_type).strip().lower())
        if task_suite_overrides is not None:
            if n_samples is None and task_suite_overrides.n_samples is not None:
                n_samples = int(task_suite_overrides.n_samples)
            if train_split is None and task_suite_overrides.train_split is not None:
                train_split = float(task_suite_overrides.train_split)

        suite_seed = int(task_seed) + 7919 * max(1, input_dim)

        scores = []
        task_results = []
        if task_type == "cifar10_binary":
            task = _load_cifar_task(input_dim, preload=False)
            task_specs = _get_cifar_validator_task_specs(
                input_dim,
                task_count=task_count,
                task_seed=task_seed,
                n_samples=n_samples,
                train_split=train_split,
            )
            if not task_specs:
                raise RuntimeError("Validator task list is empty")

            for spec in task_specs:
                task.load_data(task_spec=spec)
                eval_seed = _eval_seed_for_task_spec(spec, suite_seed=suite_seed)
                task_score = float(task.evaluate_algorithm(algorithm, epochs=epochs, rng_seed=eval_seed))
                scores.append(task_score)
                task_results.append(
                    {
                        "task_type": task_type,
                        "task_id": int(getattr(spec, "task_id", -1)),
                        "class_pair": list(getattr(spec, "class_pair", ())),
                        "projection_seed": int(getattr(spec, "projection_seed", 0) or 0),
                        "split_seed": int(getattr(spec, "split_seed", 0) or 0),
                        "score": float(task_score),
                    }
                )
        elif task_type == "mnist_binary":
            task = MNISTBinaryTask()
            task_specs = _get_mnist_validator_task_specs(
                input_dim,
                task_count=task_count,
                task_seed=task_seed,
                n_samples=n_samples,
                train_split=train_split,
            )
            if not task_specs:
                raise RuntimeError("Validator task list is empty")
            for spec in task_specs:
                task.load_data(task_spec=spec)
                eval_seed = _eval_seed_for_task_spec(spec, suite_seed=suite_seed)
                task_score = float(task.evaluate_algorithm(algorithm, epochs=epochs, rng_seed=eval_seed))
                scores.append(task_score)
                task_results.append(
                    {
                        "task_type": task_type,
                        "task_id": int(getattr(spec, "task_id", -1)),
                        "data_seed": int(getattr(spec, "data_seed", 0) or 0),
                        "projection_seed": int(getattr(spec, "projection_seed", 0) or 0),
                        "split_seed": int(getattr(spec, "split_seed", 0) or 0),
                        "score": float(task_score),
                    }
                )
        elif task_type == "scalar_linear":
            task = ScalarLinearRegressionTask()
            if n_samples is not None:
                scalar_train = int(n_samples)
                scalar_val = int(n_samples)
            elif task_suite_overrides is not None:
                scalar_train = (
                    None
                    if task_suite_overrides.train_samples is None
                    else int(task_suite_overrides.train_samples)
                )
                scalar_val = (
                    None if task_suite_overrides.val_samples is None else int(task_suite_overrides.val_samples)
                )
            else:
                scalar_train = None
                scalar_val = None
            task_specs = _get_scalar_validator_task_specs(
                input_dim,
                task_count=task_count,
                task_seed=task_seed,
                train_samples=scalar_train,
                val_samples=scalar_val,
            )
            if not task_specs:
                raise RuntimeError("Validator task list is empty")
            for spec in task_specs:
                task.load_data(task_spec=spec)
                eval_seed = _eval_seed_for_task_spec(spec, suite_seed=suite_seed)
                task_score = float(task.evaluate_algorithm(algorithm, epochs=epochs, rng_seed=eval_seed))
                scores.append(task_score)
                task_results.append(
                    {
                        "task_type": task_type,
                        "task_id": int(getattr(spec, "task_id", -1)),
                        "param_seed": int(getattr(spec, "param_seed", 0) or 0),
                        "data_seed": int(getattr(spec, "data_seed", 0) or 0),
                        "score": float(task_score),
                    }
                )
        else:
            logger.warning("Task type not supported by verify_solution_quality: %s", task_type)
            return False, -np.inf

        score = float(np.median(scores)) if scores else task.get_baseline_fitness()
        # Never allow untrusted submissions to trigger verbose per-task logging.
        # Enable explicitly via environment variable + DEBUG log level.
        log_all_task_scores = bool(getattr(hp, "log_task_scores", False))
        if not log_all_task_scores:
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


def verify_solution_quality_on_task_type(
    solution_data: Dict[str, Any],
    task_type: str,
    sota_threshold: float = None,
    **kwargs: Any,
) -> Tuple[bool, float]:
    """Compatibility wrapper for callers that pass task_type separately."""

    payload = dict(solution_data)
    payload["task_type"] = str(task_type)
    return verify_solution_quality(payload, sota_threshold, **kwargs)


def score_algorithm_on_eval_suite(
    algorithm_dsl: str,
    input_dim: int = None,
    *,
    epochs: Optional[int] = None,
    task_count: Optional[int] = None,
    task_seed: Optional[int] = None,
    n_samples: Optional[int] = None,
    train_split: Optional[float] = None,
) -> float:
    """
    Deterministically score an algorithm on the validator-style eval suite (median over tasks).
    This is the scoring policy intended for "evaluate" tasks.
    """

    if not algorithm_dsl:
        return -np.inf

    hp = get_validator_hyperparams()
    if epochs is None:
        epochs = int(getattr(hp, "epochs", 1) or 1)
    if task_count is None:
        task_count = int(getattr(hp, "task_count", VALIDATOR_TASK_COUNT) or VALIDATOR_TASK_COUNT)
    if task_seed is None:
        task_seed = int(getattr(hp, "task_seed", VALIDATOR_TASK_SEED) or VALIDATOR_TASK_SEED)

    if n_samples is None or train_split is None:
        suite_overrides = hp.tasks.get("cifar10_binary")
        if suite_overrides is not None:
            if n_samples is None and suite_overrides.n_samples is not None:
                n_samples = int(suite_overrides.n_samples)
            if train_split is None and suite_overrides.train_split is not None:
                train_split = float(suite_overrides.train_split)

    if input_dim:
        dim = int(input_dim)
    else:
        try:
            dim = int(getattr(hp, "default_input_dim", DEFAULT_CIFAR_INPUT_DIM))
        except Exception:
            dim = DEFAULT_CIFAR_INPUT_DIM
    task = _load_cifar_task(dim, preload=False)
    try:
        algorithm = DSLParser.from_dsl(algorithm_dsl, dim)
    except Exception as e:
        logger.warning("Failed to parse algorithm DSL: %s", e)
        return -np.inf

    task_specs = _get_cifar_validator_task_specs(
        dim,
        task_count=task_count,
        task_seed=task_seed,
        n_samples=n_samples,
        train_split=train_split,
    )
    if not task_specs:
        return -np.inf

    suite_seed = int(task_seed) + 7919 * max(1, dim)

    scores = []
    for spec in task_specs:
        task.load_data(task_spec=spec)
        eval_seed = _eval_seed_for_task_spec(spec, suite_seed=suite_seed)
        scores.append(float(task.evaluate_algorithm(algorithm, epochs=max(1, int(epochs)), rng_seed=eval_seed)))
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

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _coerce_int(value: object, *, default: int, min_value: Optional[int] = None) -> int:
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), int(out))
    return int(out)


def _coerce_optional_int(value: object, *, min_value: Optional[int] = None) -> Optional[int]:
    if value is None:
        return None
    try:
        out = int(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if min_value is not None:
        out = max(int(min_value), int(out))
    return int(out)


def _coerce_float(value: object, *, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def _coerce_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _env_path(*names: str) -> Optional[Path]:
    for name in names:
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            return Path(raw).expanduser().resolve()
        except Exception:
            continue
    return None


def _default_validator_hyperparams_path() -> Path:
    override = _env_path("BITSOTA_VALIDATOR_HYPERPARAMS_PATH", "VALIDATOR_HYPERPARAMS_PATH")
    return override or (_REPO_ROOT / "validator_hyperparams.json")


def _default_miner_hyperparams_path() -> Path:
    override = _env_path("BITSOTA_MINER_HYPERPARAMS_PATH", "MINER_HYPERPARAMS_PATH")
    return override or (_REPO_ROOT / "miner_hyperparams.json")


@dataclass(frozen=True)
class ValidatorTaskSuiteHyperparams:
    n_samples: Optional[int] = None
    train_split: Optional[float] = None
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None


@dataclass(frozen=True)
class ValidatorHyperparams:
    epochs: int
    task_count: int
    task_seed: int
    default_task_type: str
    default_input_dim: int
    log_task_scores: bool
    tasks: Dict[str, ValidatorTaskSuiteHyperparams]
    source_path: Path


@dataclass(frozen=True)
class MinerHyperparams:
    miner_task_count: int
    miner_task_seed: int
    validator_task_count: Optional[int]
    fec_cache_size: int
    fec_train_examples: int
    fec_valid_examples: int
    fec_forget_every: int
    submission_cooldown_seconds: int
    submit_only_if_improved: bool
    max_submission_attempts_per_generation: Optional[int]
    validate_every_n_generations: int
    sota_cache_seconds: float
    sota_failure_backoff_seconds: float
    persist_state: bool
    persist_every_n_generations: int
    gene_dump_every: int
    source_path: Path


def _parse_validator_hyperparams(path: Path) -> ValidatorHyperparams:
    data = _read_json(path)

    defaults: Dict[str, Any] = {
        "epochs": 1,
        "task_count": 128,
        "task_seed": 1337,
        "default_task_type": "cifar10_binary",
        "default_input_dim": 16,
        "log_task_scores": False,
    }

    epochs = _coerce_int(data.get("epochs", defaults["epochs"]), default=int(defaults["epochs"]), min_value=1)
    task_count = _coerce_int(
        data.get("task_count", defaults["task_count"]), default=int(defaults["task_count"]), min_value=1
    )
    task_seed = _coerce_int(data.get("task_seed", defaults["task_seed"]), default=int(defaults["task_seed"]), min_value=0)
    default_task_type = str(data.get("default_task_type", defaults["default_task_type"]) or defaults["default_task_type"]).strip().lower()
    default_input_dim = _coerce_int(
        data.get("default_input_dim", defaults["default_input_dim"]), default=int(defaults["default_input_dim"]), min_value=1
    )
    log_task_scores = bool(data.get("log_task_scores", defaults["log_task_scores"]))

    tasks: Dict[str, ValidatorTaskSuiteHyperparams] = {}
    tasks_raw = data.get("tasks")
    if isinstance(tasks_raw, dict):
        for task_name, cfg in tasks_raw.items():
            if not isinstance(cfg, dict):
                continue
            n_samples = _coerce_optional_int(cfg.get("n_samples"), min_value=1)
            train_split = _coerce_optional_float(cfg.get("train_split"))
            if train_split is not None and not (0.0 < float(train_split) < 1.0):
                train_split = None
            train_samples = _coerce_optional_int(cfg.get("train_samples"), min_value=1)
            val_samples = _coerce_optional_int(cfg.get("val_samples"), min_value=1)
            tasks[str(task_name).strip().lower()] = ValidatorTaskSuiteHyperparams(
                n_samples=n_samples,
                train_split=train_split,
                train_samples=train_samples,
                val_samples=val_samples,
            )

    # Legacy env overrides (kept for backward compatibility).
    env_task_count = os.environ.get("VALIDATOR_TASK_COUNT")
    if env_task_count:
        task_count = _coerce_int(env_task_count, default=task_count, min_value=1)
    env_task_seed = os.environ.get("VALIDATOR_TASK_SEED")
    if env_task_seed:
        task_seed = _coerce_int(env_task_seed, default=task_seed, min_value=0)
    env_log_task_scores = os.environ.get("LOG_VALIDATOR_TASK_SCORES")
    if env_log_task_scores:
        log_task_scores = _truthy(env_log_task_scores)

    return ValidatorHyperparams(
        epochs=int(epochs),
        task_count=int(task_count),
        task_seed=int(task_seed),
        default_task_type=str(default_task_type),
        default_input_dim=int(default_input_dim),
        log_task_scores=bool(log_task_scores),
        tasks=tasks,
        source_path=path,
    )


def _parse_miner_hyperparams(path: Path) -> MinerHyperparams:
    data = _read_json(path)

    defaults: Dict[str, Any] = {
        "miner_task_count": 32,
        "miner_task_seed": 0,
        "validator_task_count": None,
        "fec_cache_size": 100000,
        "fec_train_examples": 32,
        "fec_valid_examples": 32,
        "fec_forget_every": 0,
        "submission_cooldown_seconds": 60,
        "submit_only_if_improved": False,
        "max_submission_attempts_per_generation": None,
        "validate_every_n_generations": 1,
        "sota_cache_seconds": 30.0,
        "sota_failure_backoff_seconds": 5.0,
        "persist_state": True,
        "persist_every_n_generations": 5000,
        "gene_dump_every": 1000,
    }

    miner_task_count = _coerce_int(
        data.get("miner_task_count", defaults["miner_task_count"]), default=int(defaults["miner_task_count"]), min_value=1
    )
    miner_task_seed = _coerce_int(
        data.get("miner_task_seed", defaults["miner_task_seed"]), default=int(defaults["miner_task_seed"]), min_value=0
    )
    validator_task_count = _coerce_optional_int(data.get("validator_task_count"), min_value=1)
    fec_cache_size = _coerce_int(data.get("fec_cache_size", defaults["fec_cache_size"]), default=int(defaults["fec_cache_size"]), min_value=0)
    fec_train_examples = _coerce_int(
        data.get("fec_train_examples", defaults["fec_train_examples"]), default=int(defaults["fec_train_examples"]), min_value=1
    )
    fec_valid_examples = _coerce_int(
        data.get("fec_valid_examples", defaults["fec_valid_examples"]), default=int(defaults["fec_valid_examples"]), min_value=1
    )
    fec_forget_every = _coerce_int(
        data.get("fec_forget_every", defaults["fec_forget_every"]), default=int(defaults["fec_forget_every"]), min_value=0
    )
    submission_cooldown_seconds = _coerce_int(
        data.get("submission_cooldown_seconds", defaults["submission_cooldown_seconds"]),
        default=int(defaults["submission_cooldown_seconds"]),
        min_value=0,
    )
    submit_only_if_improved = bool(data.get("submit_only_if_improved", defaults["submit_only_if_improved"]))
    max_submission_attempts_per_generation = _coerce_optional_int(
        data.get("max_submission_attempts_per_generation"), min_value=1
    )
    validate_every_n_generations = _coerce_int(
        data.get("validate_every_n_generations", defaults["validate_every_n_generations"]),
        default=int(defaults["validate_every_n_generations"]),
        min_value=1,
    )
    sota_cache_seconds = _coerce_float(data.get("sota_cache_seconds", defaults["sota_cache_seconds"]), default=float(defaults["sota_cache_seconds"]))
    sota_failure_backoff_seconds = _coerce_float(
        data.get("sota_failure_backoff_seconds", defaults["sota_failure_backoff_seconds"]),
        default=float(defaults["sota_failure_backoff_seconds"]),
    )
    persist_state = bool(data.get("persist_state", defaults["persist_state"]))
    persist_every_n_generations = _coerce_int(
        data.get("persist_every_n_generations", defaults["persist_every_n_generations"]),
        default=int(defaults["persist_every_n_generations"]),
        min_value=0,
    )
    gene_dump_every = _coerce_int(
        data.get("gene_dump_every", defaults["gene_dump_every"]), default=int(defaults["gene_dump_every"]), min_value=1
    )

    # Legacy env overrides (kept for backward compatibility).
    env_task_count = os.environ.get("MINER_TASK_COUNT")
    if env_task_count:
        miner_task_count = _coerce_int(env_task_count, default=miner_task_count, min_value=1)
    env_task_seed = os.environ.get("MINER_TASK_SEED")
    if env_task_seed:
        miner_task_seed = _coerce_int(env_task_seed, default=miner_task_seed, min_value=0)
    env_fec_cache = os.environ.get("MINER_FEC_CACHE_SIZE")
    if env_fec_cache:
        fec_cache_size = _coerce_int(env_fec_cache, default=fec_cache_size, min_value=0)
    env_fec_train = os.environ.get("MINER_FEC_TRAIN_EXAMPLES")
    if env_fec_train:
        fec_train_examples = _coerce_int(env_fec_train, default=fec_train_examples, min_value=1)
    env_fec_valid = os.environ.get("MINER_FEC_VALID_EXAMPLES")
    if env_fec_valid:
        fec_valid_examples = _coerce_int(env_fec_valid, default=fec_valid_examples, min_value=1)
    env_fec_forget = os.environ.get("MINER_FEC_FORGET_EVERY")
    if env_fec_forget:
        fec_forget_every = _coerce_int(env_fec_forget, default=fec_forget_every, min_value=0)
    env_cooldown = os.environ.get("MINER_SUBMISSION_COOLDOWN_SECONDS")
    if env_cooldown:
        submission_cooldown_seconds = _coerce_int(env_cooldown, default=submission_cooldown_seconds, min_value=0)
    env_submit_improved = os.environ.get("MINER_SUBMIT_ONLY_IF_IMPROVED")
    if env_submit_improved:
        submit_only_if_improved = _truthy(env_submit_improved)
    env_max_attempts = os.environ.get("MINER_MAX_SUBMISSION_ATTEMPTS_PER_GENERATION")
    if env_max_attempts:
        max_submission_attempts_per_generation = _coerce_optional_int(env_max_attempts, min_value=1)
    env_validate_every = os.environ.get("MINER_VALIDATE_EVERY_N_GENERATIONS")
    if env_validate_every:
        validate_every_n_generations = _coerce_int(env_validate_every, default=validate_every_n_generations, min_value=1)
    env_sota_cache = os.environ.get("MINER_SOTA_CACHE_SECONDS")
    if env_sota_cache:
        sota_cache_seconds = _coerce_float(env_sota_cache, default=sota_cache_seconds)
    env_sota_backoff = os.environ.get("MINER_SOTA_FAILURE_BACKOFF_SECONDS")
    if env_sota_backoff:
        sota_failure_backoff_seconds = _coerce_float(env_sota_backoff, default=sota_failure_backoff_seconds)
    env_persist = os.environ.get("MINER_PERSIST_STATE")
    if env_persist:
        persist_state = _truthy(env_persist)
    env_persist_every = os.environ.get("MINER_PERSIST_EVERY_N_GENERATIONS")
    if env_persist_every:
        persist_every_n_generations = _coerce_int(env_persist_every, default=persist_every_n_generations, min_value=0)
    env_gene_dump_every = os.environ.get("MINER_GENE_DUMP_EVERY")
    if env_gene_dump_every:
        gene_dump_every = _coerce_int(env_gene_dump_every, default=gene_dump_every, min_value=1)

    return MinerHyperparams(
        miner_task_count=int(miner_task_count),
        miner_task_seed=int(miner_task_seed),
        validator_task_count=validator_task_count,
        fec_cache_size=int(fec_cache_size),
        fec_train_examples=int(fec_train_examples),
        fec_valid_examples=int(fec_valid_examples),
        fec_forget_every=int(fec_forget_every),
        submission_cooldown_seconds=int(submission_cooldown_seconds),
        submit_only_if_improved=bool(submit_only_if_improved),
        max_submission_attempts_per_generation=max_submission_attempts_per_generation,
        validate_every_n_generations=int(validate_every_n_generations),
        sota_cache_seconds=float(sota_cache_seconds),
        sota_failure_backoff_seconds=float(sota_failure_backoff_seconds),
        persist_state=bool(persist_state),
        persist_every_n_generations=int(persist_every_n_generations),
        gene_dump_every=int(gene_dump_every),
        source_path=path,
    )


def _validator_env_key() -> tuple[Optional[str], Optional[str], Optional[str]]:
    return (
        os.environ.get("VALIDATOR_TASK_COUNT"),
        os.environ.get("VALIDATOR_TASK_SEED"),
        os.environ.get("LOG_VALIDATOR_TASK_SCORES"),
    )


def _miner_env_key() -> tuple[Optional[str], ...]:
    return (
        os.environ.get("MINER_TASK_COUNT"),
        os.environ.get("MINER_TASK_SEED"),
        os.environ.get("MINER_FEC_CACHE_SIZE"),
        os.environ.get("MINER_FEC_TRAIN_EXAMPLES"),
        os.environ.get("MINER_FEC_VALID_EXAMPLES"),
        os.environ.get("MINER_FEC_FORGET_EVERY"),
        os.environ.get("MINER_SUBMISSION_COOLDOWN_SECONDS"),
        os.environ.get("MINER_SUBMIT_ONLY_IF_IMPROVED"),
        os.environ.get("MINER_MAX_SUBMISSION_ATTEMPTS_PER_GENERATION"),
        os.environ.get("MINER_VALIDATE_EVERY_N_GENERATIONS"),
        os.environ.get("MINER_SOTA_CACHE_SECONDS"),
        os.environ.get("MINER_SOTA_FAILURE_BACKOFF_SECONDS"),
        os.environ.get("MINER_PERSIST_STATE"),
        os.environ.get("MINER_PERSIST_EVERY_N_GENERATIONS"),
        os.environ.get("MINER_GENE_DUMP_EVERY"),
    )


@lru_cache(maxsize=32)
def _cached_validator_hyperparams(
    path_str: str, env_key: tuple[Optional[str], Optional[str], Optional[str]]
) -> ValidatorHyperparams:
    _ = env_key
    return _parse_validator_hyperparams(Path(path_str))


@lru_cache(maxsize=32)
def _cached_miner_hyperparams(path_str: str, env_key: tuple[Optional[str], ...]) -> MinerHyperparams:
    _ = env_key
    return _parse_miner_hyperparams(Path(path_str))


def get_validator_hyperparams(path: str | Path | None = None) -> ValidatorHyperparams:
    p = Path(path).expanduser().resolve() if path else _default_validator_hyperparams_path()
    return _cached_validator_hyperparams(str(p), _validator_env_key())


def get_miner_hyperparams(path: str | Path | None = None) -> MinerHyperparams:
    p = Path(path).expanduser().resolve() if path else _default_miner_hyperparams_path()
    return _cached_miner_hyperparams(str(p), _miner_env_key())


def clear_hyperparams_cache() -> None:
    _cached_validator_hyperparams.cache_clear()
    _cached_miner_hyperparams.cache_clear()

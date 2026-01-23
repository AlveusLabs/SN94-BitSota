from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except Exception:
            return None
    return None


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def _coerce_env_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float, str)):
        return str(value)
    try:
        return str(value)
    except Exception:
        return None


def apply_env_overrides(env_overrides: Mapping[str, Any], *, environ: Optional[Dict[str, str]] = None) -> None:
    """Best-effort: set environment variables from a JSON-ish mapping."""
    target = os.environ if environ is None else environ
    for key, raw in dict(env_overrides).items():
        name = str(key).strip()
        if not name:
            continue
        value = _coerce_env_value(raw)
        if value is None:
            continue
        target[name] = value


@dataclass(frozen=True)
class ProblemConfig:
    source_path: Optional[Path] = None
    env: Dict[str, str] = field(default_factory=dict)

    miner_task_count: Optional[int] = None
    validator_task_count: Optional[int] = None
    miner_validate_every_n_generations: Optional[int] = None

    engine_type: Optional[str] = None
    checkpoint_generations: Optional[int] = None

    fec_cache_size: Optional[int] = None
    fec_train_examples: Optional[int] = None
    fec_valid_examples: Optional[int] = None
    fec_forget_every: Optional[int] = None

    engine_params: Dict[str, Any] = field(default_factory=dict)


def _resolve_path(path: str | Path) -> Optional[Path]:
    try:
        return Path(path).expanduser().resolve()
    except Exception:
        return None


def find_problem_config_path(explicit_path: Optional[str | Path] = None) -> Optional[Path]:
    if explicit_path:
        resolved = _resolve_path(explicit_path)
        if resolved and resolved.exists():
            return resolved

    env_path = os.environ.get("BITSOTA_PROBLEM_CONFIG")
    if env_path:
        resolved = _resolve_path(env_path)
        if resolved and resolved.exists():
            return resolved

    candidates = [
        Path.cwd() / "problem_config.json",
        Path.home() / ".bitsota" / "problem_config.json",
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.expanduser().resolve()
        except Exception:
            continue
    return None


def _parse_engine_params(raw: Mapping[str, Any]) -> Dict[str, Any]:
    engine_params: Dict[str, Any] = {}

    for key in ("pop_size", "tournament_size", "archive_size", "cifar_seed"):
        value = _maybe_int(raw.get(key))
        if value is not None:
            engine_params[key] = int(value)

    mutation_prob = _maybe_float(raw.get("mutation_prob"))
    if mutation_prob is not None:
        engine_params["mutation_prob"] = float(mutation_prob)

    phase_max_sizes: Dict[str, int] = {}
    for phase, source_key in (
        ("setup", "setup_max_ops"),
        ("predict", "predict_max_ops"),
        ("learn", "learn_max_ops"),
    ):
        value = _maybe_int(raw.get(source_key))
        if value is not None:
            phase_max_sizes[phase] = max(1, int(value))

    explicit_phase_max_sizes = _as_dict(raw.get("phase_max_sizes"))
    for phase, size in explicit_phase_max_sizes.items():
        if not isinstance(phase, str):
            continue
        value = _maybe_int(size)
        if value is None:
            continue
        phase_max_sizes[phase] = max(1, int(value))

    if phase_max_sizes:
        engine_params["phase_max_sizes"] = dict(phase_max_sizes)

    scalar_count = _maybe_int(raw.get("scalar_count") or raw.get("scalar_regs"))
    if scalar_count is not None:
        engine_params["scalar_count"] = max(0, int(scalar_count))

    vector_count = _maybe_int(raw.get("vector_count") or raw.get("vector_regs"))
    if vector_count is not None:
        engine_params["vector_count"] = max(0, int(vector_count))

    matrix_count = _maybe_int(raw.get("matrix_count") or raw.get("matrix_regs"))
    if matrix_count is not None:
        engine_params["matrix_count"] = max(0, int(matrix_count))

    vector_dim = _maybe_int(raw.get("vector_dim"))
    if vector_dim is not None:
        engine_params["vector_dim"] = max(1, int(vector_dim))

    return engine_params


def load_problem_config(explicit_path: Optional[str | Path] = None) -> Optional[ProblemConfig]:
    path = find_problem_config_path(explicit_path)
    if not path:
        return None

    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None

    root = _as_dict(payload)
    mining = _as_dict(root.get("mining")) or root

    env_overrides: Dict[str, Any] = {}
    env_overrides.update(_as_dict(root.get("env")))
    env_overrides.update(_as_dict(mining.get("env")))

    engine_params = _parse_engine_params(_as_dict(mining.get("engine_params")) or mining)

    engine_type = mining.get("engine_type") or mining.get("engine")
    if isinstance(engine_type, str):
        engine_type = engine_type.strip().lower() or None
    else:
        engine_type = None

    miner_task_count = _maybe_int(mining.get("miner_task_count"))
    validator_task_count = _maybe_int(mining.get("validator_task_count"))
    validate_every = _maybe_int(
        mining.get("miner_validate_every_n_generations")
        or mining.get("validate_every_n_generations")
        or mining.get("validate_every")
    )

    checkpoint_generations = _maybe_int(mining.get("checkpoint_generations"))

    return ProblemConfig(
        source_path=path,
        env={k: v for k, v in ((str(k), _coerce_env_value(v)) for k, v in env_overrides.items()) if k and v},
        miner_task_count=max(1, miner_task_count) if miner_task_count is not None else None,
        validator_task_count=max(1, validator_task_count) if validator_task_count is not None else None,
        miner_validate_every_n_generations=max(1, validate_every) if validate_every is not None else None,
        engine_type=engine_type,
        checkpoint_generations=max(1, checkpoint_generations) if checkpoint_generations is not None else None,
        fec_cache_size=_maybe_int(mining.get("fec_cache_size")),
        fec_train_examples=_maybe_int(mining.get("fec_train_examples")),
        fec_valid_examples=_maybe_int(mining.get("fec_valid_examples")),
        fec_forget_every=_maybe_int(mining.get("fec_forget_every")),
        engine_params=engine_params,
    )


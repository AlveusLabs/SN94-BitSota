from __future__ import annotations

from typing import Optional


_CPP_DEFAULT_ENGINE_PARAMS_BY_TASK = {
    # Mirrors cpp/automl_zero/run_baseline.sh memory sizes and per-phase op budgets.
    "cifar10_binary": {
        "scalar_count": 5,
        "vector_count": 9,
        "matrix_count": 2,
        "phase_max_sizes": {"setup": 7, "predict": 11, "learn": 23},
    },
    # Mirrors cpp/automl_zero/run_demo.sh memory sizes and fixed phase sizes.
    "scalar_linear": {
        "scalar_count": 4,
        "vector_count": 3,
        "matrix_count": 1,
        "phase_max_sizes": {"setup": 10, "predict": 2, "learn": 8},
    },
}


def apply_cpp_defaults_to_engine_params(
    task_type: str,
    engine_params: Optional[dict],
    *,
    explicit_engine_params: Optional[dict] = None,
) -> Optional[dict]:
    """
    Apply C++-aligned defaults for memory sizes + phase op limits.

    Values from `explicit_engine_params` (typically problem_config.engine_params)
    are treated as user overrides and are not overwritten.
    """

    base: dict = dict(engine_params) if isinstance(engine_params, dict) else {}
    explicit: dict = dict(explicit_engine_params) if isinstance(explicit_engine_params, dict) else {}
    defaults = _CPP_DEFAULT_ENGINE_PARAMS_BY_TASK.get(str(task_type), {})
    if not defaults:
        return base or None

    for key in ("scalar_count", "vector_count", "matrix_count"):
        if key in explicit:
            continue
        if key in defaults:
            base[key] = int(defaults[key])

    default_phase_sizes = defaults.get("phase_max_sizes")
    if isinstance(default_phase_sizes, dict) and "phase_max_sizes" not in explicit:
        base["phase_max_sizes"] = dict(default_phase_sizes)

    return base or None


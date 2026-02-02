from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AppConfig:
    relay_endpoint: str = "https://relay.bitsota.com"
    update_manifest_url: str = "https://relay.bitsota.com/version.json"
    pool_endpoint: str = "https://api.bitsota.com"
    cifar10_dataset_url: str = "https://cifar10.fra1.digitaloceanspaces.com/CIFAR_10_small.arff.gz"
    test_mode: bool = False
    test_invite_code: str = "TESTTEST1"
    # Default task suite sizes aligned with `cpp/automl_zero/run_*.sh`:
    # - search_tasks.num_tasks (miner fitness evaluation): 10
    # - final_tasks.num_tasks (validator verification): 100
    miner_task_count: Optional[int] = 10
    validator_task_count: Optional[int] = 100
    miner_validate_every_n_generations: int = 1000
    problem_config_path: Optional[str] = None
    population_state_path: Optional[str] = None
    miner_workers: int = 1
    miner_seed: Optional[int] = None
    miner_migration_generations: int = 0
    pool_lease_evolve_generations: int = 1000


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _find_dev_config_path() -> Optional[Path]:
    p = os.environ.get("BITSOTA_GUI_CONFIG")
    if p:
        return Path(p).expanduser().resolve()

    candidates = [
        Path.cwd() / "bitsota_gui_config.json",
        Path.cwd() / "gui_config.json",
        Path.home() / ".bitsota" / "gui_config.json",
    ]
    for c in candidates:
        if c.exists():
            return c.expanduser().resolve()
    return None


def _apply_overrides(defaults: AppConfig, overrides: Dict[str, Any]) -> AppConfig:
    allowed_strings = {
        "relay_endpoint",
        "update_manifest_url",
        "pool_endpoint",
        "cifar10_dataset_url",
        "test_invite_code",
        "problem_config_path",
        "population_state_path",
    }
    cleaned: Dict[str, Any] = {
        k: v for k, v in overrides.items() if k in allowed_strings and isinstance(v, str) and v
    }
    for key in (
        "miner_task_count",
        "validator_task_count",
        "miner_validate_every_n_generations",
        "miner_workers",
        "miner_migration_generations",
        "pool_lease_evolve_generations",
    ):
        raw = overrides.get(key)
        if raw is None:
            continue
        value: Optional[int] = None
        if isinstance(raw, int):
            value = raw
        elif isinstance(raw, str) and raw.strip().isdigit():
            try:
                value = int(raw.strip())
            except Exception:
                value = None
        if value is None:
            continue
        if key == "miner_migration_generations":
            cleaned[key] = max(0, int(value))
        elif key == "miner_validate_every_n_generations":
            cleaned[key] = max(1, int(value))
        elif key == "pool_lease_evolve_generations":
            cleaned[key] = max(1, int(value))
        else:
            if value > 0:
                cleaned[key] = int(value)
    raw_seed = overrides.get("miner_seed")
    if raw_seed is not None:
        seed_value: Optional[int] = None
        if isinstance(raw_seed, int):
            seed_value = raw_seed
        elif isinstance(raw_seed, str) and raw_seed.strip():
            try:
                seed_value = int(raw_seed.strip())
            except Exception:
                seed_value = None
        if seed_value is not None:
            cleaned["miner_seed"] = int(seed_value)
    if "test_mode" in overrides:
        cleaned["test_mode"] = bool(overrides["test_mode"])
    return replace(defaults, **cleaned) if cleaned else defaults


_CACHED: Optional[AppConfig] = None


def get_app_config(force_reload: bool = False) -> AppConfig:
    """
    Config policy:
    - PyInstaller build (sys.frozen == True): always use hardcoded defaults.
    - Local/dev (not frozen): allow overrides via `BITSOTA_GUI_CONFIG` or a well-known JSON file.
    """
    global _CACHED
    if _CACHED is not None and not force_reload:
        return _CACHED

    defaults = AppConfig()
    if is_frozen():
        _CACHED = defaults
        return _CACHED

    if os.environ.get("BITSOTA_TEST_MODE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        defaults = replace(
            defaults,
            test_mode=True,
            test_invite_code=os.environ.get("BITSOTA_TEST_INVITE_CODE", defaults.test_invite_code),
        )

    path = _find_dev_config_path()
    overrides = _read_json_file(path) if path else {}
    _CACHED = _apply_overrides(defaults, overrides)
    return _CACHED

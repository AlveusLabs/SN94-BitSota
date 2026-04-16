from __future__ import annotations

import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from gui.resource_path import resource_path


DEFAULT_GUI_CONFIG_PATH = Path.home() / ".bitsota" / "gui_config.json"
DEFAULT_MASTER_PROMPT_RESOURCE = "docs/guides/autoresearch-agent-master-prompt.md"
DEFAULT_MERKLE_METADATA_RESOURCE = "gui/assets/merklepool.json"
DEFAULT_TESTNET_PRESET = "autoresearch_testnet"
DEFAULT_LOCAL_PRESET = "local_dev"

_TESTNET_PRESET_VALUES: Dict[str, str] = {
    "pool_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com",
    "merkle_claim_endpoint": "https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims",
    "onchain_ws_url": "wss://test.finney.opentensor.ai:443",
    "onchain_contract": "5GUw1gZVfUTXWLEbA7G6Xdp8QsUHAy2xpVmaj5fRc1gW1Xyy",
    "onchain_metadata_path": "",
    "research_coordinator_endpoint": "https://chvp2wytst.eu-central-1.awsapprunner.com",
}

_LOCAL_PRESET_VALUES: Dict[str, str] = {
    "pool_endpoint": "http://127.0.0.1:8434",
    "merkle_claim_endpoint": "http://127.0.0.1:8844",
    "onchain_ws_url": "ws://127.0.0.1:9944",
    "onchain_contract": "",
    "onchain_metadata_path": "",
    "research_coordinator_endpoint": "http://127.0.0.1:8000",
}

_NETWORK_PRESET_VALUES: Dict[str, Dict[str, str]] = {
    DEFAULT_TESTNET_PRESET: _TESTNET_PRESET_VALUES,
    DEFAULT_LOCAL_PRESET: _LOCAL_PRESET_VALUES,
}

_RESEARCH_AGENT_PROVIDER_LABELS: Dict[str, str] = {
    "codex_cli": "Codex CLI",
    "claude_code": "Claude Code",
    "openai_compatible": "OpenAI-compatible API",
    "custom_command": "Custom command",
}


@dataclass(frozen=True)
class AppConfig:
    network_preset: str = DEFAULT_TESTNET_PRESET
    relay_endpoint: str = "https://relay.bitsota.com"
    update_manifest_url: str = "https://relay.bitsota.com/version.json"
    pool_endpoint: str = _TESTNET_PRESET_VALUES["pool_endpoint"]
    merkle_claim_endpoint: str = _TESTNET_PRESET_VALUES["merkle_claim_endpoint"]
    onchain_ws_url: str = _TESTNET_PRESET_VALUES["onchain_ws_url"]
    onchain_contract: str = _TESTNET_PRESET_VALUES["onchain_contract"]
    onchain_metadata_path: str = _TESTNET_PRESET_VALUES["onchain_metadata_path"]
    research_coordinator_endpoint: str = _TESTNET_PRESET_VALUES["research_coordinator_endpoint"]
    research_agent_provider: str = ""
    research_agent_command: str = ""
    research_agent_mode: str = "gui_managed"
    research_llm_base_url: str = "http://127.0.0.1:11434/v1"
    research_llm_model: str = ""
    research_llm_api_key: str = ""
    research_test_task_slug: str = ""
    cifar10_dataset_url: str = "https://cifar10.fra1.digitaloceanspaces.com/CIFAR_10_small.arff.gz"
    test_mode: bool = False
    test_invite_code: str = "TESTTEST1"
    research_test_autostart: bool = False
    research_test_auto_restart: bool = True
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
    pool_lease_evolve_generations: int = 160
    research_test_restart_delay_seconds: int = 5


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def default_gui_config_path() -> Path:
    override = os.environ.get("BITSOTA_GUI_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_GUI_CONFIG_PATH.expanduser().resolve()


def _candidate_config_paths() -> list[Path]:
    override = os.environ.get("BITSOTA_GUI_CONFIG")
    if override:
        return [default_gui_config_path()]
    return [
        (Path.cwd() / "bitsota_gui_config.json").expanduser().resolve(),
        (Path.cwd() / "gui_config.json").expanduser().resolve(),
        default_gui_config_path(),
    ]


def _strip_json_comments(raw_text: str) -> str:
    cleaned_lines: list[str] = []
    for line in str(raw_text or "").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        cleaned_lines.append(line)
    without_comments = "\n".join(cleaned_lines)
    return re.sub(r",(\s*[}\]])", r"\1", without_comments)


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(_strip_json_comments(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _find_dev_config_path() -> Optional[Path]:
    for candidate in _candidate_config_paths():
        if candidate.exists():
            return candidate
    return None


def _normalize_network_preset(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in _NETWORK_PRESET_VALUES:
        return normalized
    return DEFAULT_TESTNET_PRESET


def network_preset_values(preset: str | None) -> Dict[str, str]:
    normalized = _normalize_network_preset(preset)
    return dict(_NETWORK_PRESET_VALUES.get(normalized) or _NETWORK_PRESET_VALUES[DEFAULT_TESTNET_PRESET])


def _apply_network_preset(defaults: AppConfig, preset: str | None) -> AppConfig:
    normalized = _normalize_network_preset(preset)
    values = network_preset_values(normalized)
    values["network_preset"] = normalized
    return replace(defaults, **values)


def _resolve_resource_path(relative_path: str) -> str:
    candidate = Path(resource_path(relative_path))
    if candidate.exists():
        return str(candidate.resolve())
    return ""


def resolve_master_prompt_path() -> str:
    return _resolve_resource_path(DEFAULT_MASTER_PROMPT_RESOURCE)


def resolve_bundled_metadata_path() -> str:
    return _resolve_resource_path(DEFAULT_MERKLE_METADATA_RESOURCE)


def infer_research_agent_provider(
    *,
    configured_provider: str = "",
    agent_command: str = "",
    llm_model: str = "",
    llm_base_url: str = "",
    llm_api_key: str = "",
) -> str:
    normalized = str(configured_provider or "").strip().lower()
    if normalized in _RESEARCH_AGENT_PROVIDER_LABELS:
        return normalized

    command = str(agent_command or "").strip().lower()
    if "codex exec" in command:
        return "codex_cli"
    if "claude code" in command or command.startswith("claude "):
        return "claude_code"
    if str(agent_command or "").strip():
        return "custom_command"
    if str(llm_model or "").strip() and (str(llm_base_url or "").strip() or str(llm_api_key or "").strip()):
        return "openai_compatible"
    return ""


def research_agent_provider_label(provider: str) -> str:
    return _RESEARCH_AGENT_PROVIDER_LABELS.get(str(provider or "").strip().lower(), "")


def build_preset_research_agent_command(provider: str) -> str:
    prompt_path = resolve_master_prompt_path()
    if not prompt_path:
        return ""

    prompt_path_quoted = shlex.quote(prompt_path)
    normalized = str(provider or "").strip().lower()
    if normalized == "codex_cli":
        return (
            "bash -lc 'cat {intro_path_quoted} "
            f"{prompt_path_quoted} "
            "| codex exec --skip-git-repo-check --full-auto -C {repo_dir_quoted} "
            "--add-dir {workspace_dir_quoted} -o {submission_result_path_quoted} -'"
        )
    if normalized == "claude_code":
        return (
            "bash -lc 'cat {intro_path_quoted} "
            f"{prompt_path_quoted} "
            "| claude code --dangerously-skip-permissions -C {repo_dir_quoted} "
            "> {submission_result_path_quoted}'"
        )
    return ""


def needs_research_setup(cfg: Optional[AppConfig] = None) -> bool:
    active = cfg or get_app_config()
    provider = infer_research_agent_provider(
        configured_provider=getattr(active, "research_agent_provider", ""),
        agent_command=getattr(active, "research_agent_command", ""),
        llm_model=getattr(active, "research_llm_model", ""),
        llm_base_url=getattr(active, "research_llm_base_url", ""),
        llm_api_key=getattr(active, "research_llm_api_key", ""),
    )
    if provider:
        return False
    env_provider = infer_research_agent_provider(
        configured_provider=os.environ.get("BITSOTA_RESEARCH_AGENT_PROVIDER", ""),
        agent_command=os.environ.get("BITSOTA_RESEARCH_AGENT_COMMAND", ""),
        llm_model=os.environ.get("BITSOTA_RESEARCH_LLM_MODEL", ""),
        llm_base_url=os.environ.get("BITSOTA_RESEARCH_LLM_BASE_URL", ""),
        llm_api_key=os.environ.get("BITSOTA_RESEARCH_LLM_API_KEY", ""),
    )
    return not bool(env_provider)


def _apply_overrides(defaults: AppConfig, overrides: Dict[str, Any]) -> AppConfig:
    allowed_strings = {
        "network_preset",
        "relay_endpoint",
        "update_manifest_url",
        "pool_endpoint",
        "merkle_claim_endpoint",
        "onchain_ws_url",
        "onchain_contract",
        "onchain_metadata_path",
        "research_coordinator_endpoint",
        "research_agent_provider",
        "research_agent_command",
        "research_agent_mode",
        "research_llm_base_url",
        "research_llm_model",
        "research_llm_api_key",
        "research_test_task_slug",
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
        "research_test_restart_delay_seconds",
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
    def _coerce_bool(raw: Any) -> Optional[bool]:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

    for key in ("test_mode", "research_test_autostart", "research_test_auto_restart"):
        if key not in overrides:
            continue
        value = _coerce_bool(overrides.get(key))
        if value is not None:
            cleaned[key] = value
    return replace(defaults, **cleaned) if cleaned else defaults


_CACHED: Optional[AppConfig] = None


def _apply_test_mode_env(defaults: AppConfig) -> AppConfig:
    if os.environ.get("BITSOTA_TEST_MODE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        return replace(
            defaults,
            test_mode=True,
            test_invite_code=os.environ.get("BITSOTA_TEST_INVITE_CODE", defaults.test_invite_code),
        )
    return defaults


def get_app_config(force_reload: bool = False) -> AppConfig:
    """
    Config policy:
    - Source and packaged builds both load the same persisted GUI config file.
    - `BITSOTA_GUI_CONFIG` overrides the path used for both reading and writing.
    - Current-working-directory config files still override the user config for dev workflows.
    """
    global _CACHED
    if _CACHED is not None and not force_reload:
        return _CACHED

    defaults = _apply_test_mode_env(AppConfig())

    path = _find_dev_config_path()
    overrides = _read_json_file(path) if path else {}
    defaults = _apply_network_preset(defaults, overrides.get("network_preset"))
    _CACHED = _apply_overrides(defaults, overrides)
    return _CACHED


def save_app_config(updates: Dict[str, Any]) -> Path:
    path = _find_dev_config_path() or default_gui_config_path()
    payload = _read_json_file(path) if path.exists() else {}
    for key, value in dict(updates or {}).items():
        if value is None:
            payload.pop(str(key), None)
            continue
        payload[str(key)] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    get_app_config(force_reload=True)
    return path

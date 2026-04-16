from __future__ import annotations

from pathlib import Path

from gui.app_config import (
    build_preset_research_agent_command,
    get_app_config,
    needs_research_setup,
    resolve_bundled_metadata_path,
    resolve_pool_coldkey_update_endpoint,
)


def test_get_app_config_defaults_to_shared_testnet_endpoints(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BITSOTA_GUI_CONFIG", str(tmp_path / "missing.json"))

    cfg = get_app_config(force_reload=True)

    assert cfg.network_preset == "autoresearch_testnet"
    assert cfg.pool_endpoint == "https://3fhi3ukpyw.eu-central-1.awsapprunner.com"
    assert cfg.merkle_claim_endpoint == "https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims"
    assert cfg.onchain_ws_url == "wss://test.finney.opentensor.ai:443"
    assert cfg.onchain_contract == "5GUw1gZVfUTXWLEbA7G6Xdp8QsUHAy2xpVmaj5fRc1gW1Xyy"
    assert cfg.research_coordinator_endpoint == "https://chvp2wytst.eu-central-1.awsapprunner.com"
    assert cfg.onchain_metadata_path == ""
    assert resolve_bundled_metadata_path().endswith(
        str(Path("gui") / "assets" / "merklepool.json")
    )


def test_get_app_config_supports_commented_dev_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "bitsota_gui_config.json"
    config_path.write_text(
        "{\n"
        '  "research_coordinator_endpoint": "http://127.0.0.1:8000",\n'
        '  "research_agent_command": "codex exec {intro_path_quoted}",\n'
        "  // Disabled while using Codex external-agent mode.\n"
        '  // "research_llm_model": "prism-ml/Bonsai-8B-gguf"\n'
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BITSOTA_GUI_CONFIG", str(config_path))

    cfg = get_app_config(force_reload=True)

    assert cfg.research_coordinator_endpoint == "http://127.0.0.1:8000"
    assert cfg.research_agent_command == "codex exec {intro_path_quoted}"
    assert cfg.research_llm_model == ""


def test_get_app_config_reads_user_config_even_when_frozen(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "gui_config.json"
    config_path.write_text(
        '{\n'
        '  "network_preset": "local_dev",\n'
        '  "research_agent_provider": "claude_code"\n'
        '}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("BITSOTA_GUI_CONFIG", str(config_path))
    monkeypatch.setattr("gui.app_config.sys", type("FrozenSys", (), {"frozen": True})())

    cfg = get_app_config(force_reload=True)

    assert cfg.network_preset == "local_dev"
    assert cfg.pool_endpoint == "http://127.0.0.1:8434"
    assert cfg.research_agent_provider == "claude_code"


def test_build_preset_research_agent_command_uses_bundled_prompt() -> None:
    command = build_preset_research_agent_command("codex_cli")

    assert "codex exec" in command
    assert "autoresearch-agent-master-prompt.md" in command
    assert "{intro_path_quoted}" in command


def test_needs_research_setup_false_when_provider_configured(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "gui_config.json"
    config_path.write_text(
        '{\n'
        '  "research_agent_provider": "codex_cli"\n'
        '}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("BITSOTA_GUI_CONFIG", str(config_path))

    cfg = get_app_config(force_reload=True)

    assert needs_research_setup(cfg) is False


def test_resolve_pool_coldkey_update_endpoint_defaults_to_pool_base_url() -> None:
    endpoint = resolve_pool_coldkey_update_endpoint(
        explicit="",
        pool_endpoint="https://3fhi3ukpyw.eu-central-1.awsapprunner.com/",
    )

    assert endpoint == "https://3fhi3ukpyw.eu-central-1.awsapprunner.com"


def test_resolve_pool_coldkey_update_endpoint_prefers_explicit_override() -> None:
    endpoint = resolve_pool_coldkey_update_endpoint(
        explicit="https://claims.bitsota.example/custom-recipient",
        pool_endpoint="https://3fhi3ukpyw.eu-central-1.awsapprunner.com/",
    )

    assert endpoint == "https://claims.bitsota.example/custom-recipient"

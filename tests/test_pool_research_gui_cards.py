from __future__ import annotations

from substrateinterface import Keypair

from gui.screens.mining.pool_mining_screen import (
    _configured_research_pools,
    _ephemeral_hotkey_address,
    _fallback_research_pools,
    _find_research_task,
    _normalize_research_task_pool,
    _resolve_research_launch_task,
    _research_runtime_settings,
    _research_gui_test_settings,
    _select_research_test_pool,
)


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self):
        return self._payload


def test_fallback_research_pools_expose_builtin_competitions() -> None:
    pools = _fallback_research_pools(
        coordinator_url="http://127.0.0.1:8000",
        agent_label="Codex CLI",
    )

    assert len(pools) >= 5
    assert pools[0]["is_research_pool"] is True
    assert pools[0]["mode"] == "research_pool"
    assert pools[0]["task_slug"] == "nanogpt-default"
    assert pools[0]["task_source"] == "template"
    assert any(pool["task_slug"] == "eggroll-efficiency" for pool in pools)


def test_normalize_research_task_pool_includes_mode_and_metric_labels() -> None:
    pool = _normalize_research_task_pool(
        task={
            "id": "task-123",
            "slug": "nanogpt-default",
            "title": "nanoGPT Replay",
            "competition_mode": "peer_evaluation",
            "metric_name": "val_bpb",
            "metric_direction": "minimize",
        },
        coordinator_url="http://127.0.0.1:8000",
        agent_label="Claude Code",
    )

    assert pool["task_id"] == "task-123"
    assert pool["coordinator_url"] == "http://127.0.0.1:8000"
    assert pool["task_source"] == "live"
    assert pool["competition_mode_label"] == "Peer Evaluation"
    assert pool["metric_label"] == "val_bpb (minimize)"
    assert pool["agent_model"] == "Claude Code"


def test_configured_research_pools_prefers_live_coordinator_tasks(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen._research_runtime_settings",
        lambda: {
            "coordinator_url": "http://127.0.0.1:8000",
            "provider": "openai_compatible",
            "provider_label": "OpenAI-compatible API",
            "agent_command": "",
            "agent_mode": "gui_managed",
            "llm_base_url": "http://127.0.0.1:11434/v1",
            "llm_model": "local-model",
            "llm_api_key": "",
        },
    )
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen.requests.get",
        lambda *args, **kwargs: _DummyResponse(
            [
                {
                    "id": "task-live-1",
                    "slug": "nanogpt-default",
                    "title": "Default nanoGPT-style five-minute replay",
                    "competition_mode": "centerless",
                    "metric_name": "val_bpb",
                    "metric_direction": "minimize",
                    "is_active": True,
                }
            ]
        ),
    )

    pools = _configured_research_pools()

    assert len(pools) == 1
    assert pools[0]["task_id"] == "task-live-1"
    assert pools[0]["coordinator_url"] == "http://127.0.0.1:8000"
    assert pools[0]["task_source"] == "live"
    assert pools[0]["competition_mode_label"] == "Centerless"


def test_research_runtime_settings_support_external_agent_command(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen.get_app_config",
        lambda: type(
            "Cfg",
            (),
            {
                "research_coordinator_endpoint": "http://127.0.0.1:8000",
                "research_agent_provider": "custom_command",
                "research_agent_command": "bash -lc 'echo test'",
                "research_agent_mode": "gui_managed",
                "research_llm_base_url": "",
                "research_llm_model": "",
                "research_llm_api_key": "",
            },
        )(),
    )
    monkeypatch.delenv("BITSOTA_RESEARCH_AGENT_COMMAND", raising=False)
    monkeypatch.delenv("BITSOTA_RESEARCH_AGENT_MODE", raising=False)
    monkeypatch.delenv("BITSOTA_RESEARCH_AGENT_PROVIDER", raising=False)

    settings = _research_runtime_settings()

    assert settings["coordinator_url"] == "http://127.0.0.1:8000"
    assert settings["provider"] == "custom_command"
    assert settings["agent_command"] == "bash -lc 'echo test'"
    assert settings["agent_mode"] == "gui_managed"


def test_research_runtime_settings_builds_codex_cli_command_from_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen.get_app_config",
        lambda: type(
            "Cfg",
            (),
            {
                "research_coordinator_endpoint": "http://127.0.0.1:8000",
                "research_agent_provider": "codex_cli",
                "research_agent_command": "",
                "research_agent_mode": "gui_managed",
                "research_llm_base_url": "",
                "research_llm_model": "",
                "research_llm_api_key": "",
            },
        )(),
    )
    monkeypatch.delenv("BITSOTA_RESEARCH_AGENT_COMMAND", raising=False)
    monkeypatch.delenv("BITSOTA_RESEARCH_AGENT_PROVIDER", raising=False)

    settings = _research_runtime_settings()

    assert settings["provider"] == "codex_cli"
    assert settings["provider_label"] == "Codex CLI"
    assert "codex exec" in settings["agent_command"]


def test_configured_research_pools_show_provider_label_when_no_llm_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen._research_runtime_settings",
        lambda: {
            "coordinator_url": "http://127.0.0.1:8000",
            "provider": "codex_cli",
            "provider_label": "Codex CLI",
            "agent_command": "bash -lc 'echo test'",
            "agent_mode": "gui_managed",
            "llm_base_url": "",
            "llm_model": "",
            "llm_api_key": "",
        },
    )
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen.requests.get",
        lambda *args, **kwargs: _DummyResponse(
            [
                {
                    "id": "task-live-1",
                    "slug": "nanogpt-default",
                    "title": "Default nanoGPT-style five-minute replay",
                    "competition_mode": "standard",
                    "metric_name": "val_bpb",
                    "metric_direction": "minimize",
                    "is_active": True,
                }
            ]
        ),
    )

    pools = _configured_research_pools()

    assert pools[0]["agent_model"] == "Codex CLI"


def test_select_research_test_pool_prefers_live_task_card() -> None:
    pools = [
        {"task_slug": "cifar10-matrix-decomposition-frontier", "task_id": "", "is_research_pool": True},
        {"task_slug": "cifar10-matrix-decomposition-frontier", "task_id": "task-live-1", "is_research_pool": True},
    ]

    selected = _select_research_test_pool(pools, "cifar10-matrix-decomposition-frontier")

    assert selected is not None
    assert selected["task_id"] == "task-live-1"


def test_find_research_task_matches_task_id_before_slug() -> None:
    tasks = [
        {"id": "task-1", "slug": "nanogpt-default"},
        {"id": "task-2", "slug": "bitnet-cpu-ternary-kernel"},
    ]

    selected = _find_research_task(tasks, task_id="task-2", task_slug="nanogpt-default")

    assert selected is not None
    assert selected["id"] == "task-2"


def test_resolve_research_launch_task_returns_live_task(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen._fetch_research_tasks",
        lambda **_kwargs: [{"id": "task-live-1", "slug": "bitnet-cpu-ternary-kernel"}],
    )

    task, error = _resolve_research_launch_task(
        coordinator_url="https://example.com",
        task_slug="bitnet-cpu-ternary-kernel",
    )

    assert error is None
    assert task is not None
    assert task["id"] == "task-live-1"


def test_resolve_research_launch_task_reports_missing_live_task(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen._fetch_research_tasks",
        lambda **_kwargs: [],
    )

    task, error = _resolve_research_launch_task(
        coordinator_url="https://example.com",
        task_slug="bitnet-cpu-ternary-kernel",
    )

    assert task is None
    assert error is not None
    assert "not live on coordinator" in error


def test_resolve_research_launch_task_reports_coordinator_failure(monkeypatch) -> None:
    def _boom(**_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("gui.screens.mining.pool_mining_screen._fetch_research_tasks", _boom)

    task, error = _resolve_research_launch_task(
        coordinator_url="http://127.0.0.1:8000",
        task_slug="bitnet-cpu-ternary-kernel",
    )

    assert task is None
    assert error is not None
    assert "failed to reach research coordinator" in error


def test_research_gui_test_settings_defaults_to_cifar_task_in_test_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen.get_app_config",
        lambda: type(
            "Cfg",
            (),
            {
                "test_mode": True,
                "research_test_autostart": True,
                "research_test_task_slug": "",
                "research_test_auto_restart": True,
                "research_test_restart_delay_seconds": 7,
            },
        )(),
    )
    monkeypatch.delenv("BITSOTA_RESEARCH_GUI_AUTOSTART", raising=False)
    monkeypatch.delenv("BITSOTA_RESEARCH_GUI_TASK_SLUG", raising=False)
    monkeypatch.delenv("BITSOTA_RESEARCH_GUI_AUTO_RESTART", raising=False)

    settings = _research_gui_test_settings()

    assert settings["enabled"] is True
    assert settings["task_slug"] == "cifar10-matrix-decomposition-frontier"
    assert settings["auto_restart"] is True
    assert settings["restart_delay_seconds"] == 7


def test_ephemeral_hotkey_address_uses_substrate_interface_generated_mnemonic() -> None:
    mnemonic = Keypair.generate_mnemonic()

    address = _ephemeral_hotkey_address(mnemonic)

    assert isinstance(address, str)
    assert len(address) > 20

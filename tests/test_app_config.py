from __future__ import annotations

from gui.app_config import get_app_config


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

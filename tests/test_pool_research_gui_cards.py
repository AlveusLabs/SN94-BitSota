from __future__ import annotations

from gui.screens.mining.pool_mining_screen import (
    _configured_research_pools,
    _fallback_research_pools,
    _normalize_research_task_pool,
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
        llm_model="local-model",
    )

    assert len(pools) == 5
    assert pools[0]["is_research_pool"] is True
    assert pools[0]["mode"] == "research_pool"
    assert pools[0]["task_slug"] == "nanogpt-default"


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
        llm_model="gpt-local",
    )

    assert pool["task_id"] == "task-123"
    assert pool["competition_mode_label"] == "Peer Evaluation"
    assert pool["metric_label"] == "val_bpb (minimize)"
    assert pool["agent_model"] == "gpt-local"


def test_configured_research_pools_prefers_live_coordinator_tasks(monkeypatch) -> None:
    monkeypatch.setattr(
        "gui.screens.mining.pool_mining_screen._research_runtime_settings",
        lambda: {
            "coordinator_url": "http://127.0.0.1:8000",
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
    assert pools[0]["competition_mode_label"] == "Centerless"

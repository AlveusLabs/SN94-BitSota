from __future__ import annotations

from types import SimpleNamespace

from miner.research_coordinator_client import CoordinatorClient


def test_cancel_claim_uses_signed_claim_cancel_route() -> None:
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk-self"))
    client = CoordinatorClient(base_url="http://127.0.0.1:8000", wallet=wallet)

    calls: list[tuple[str, str, dict | None, bool]] = []

    class _Response:
        def json(self) -> dict[str, str]:
            return {"id": "claim-123", "status": "cancelled"}

    def _fake_request(method: str, path: str, *, body=None, params=None, sign=False):
        calls.append((method, path, body, sign))
        return _Response()

    client._request = _fake_request  # type: ignore[method-assign]

    result = client.cancel_claim(claim_id="claim-123")

    assert result == {"id": "claim-123", "status": "cancelled"}
    assert calls == [("POST", "/api/v1/claims/claim-123/cancel", None, True)]


def test_select_task_pool_resumes_self_claimed_work_item() -> None:
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk-self"))
    client = CoordinatorClient(base_url="http://127.0.0.1:8000", wallet=wallet)

    client.list_tasks = lambda: [  # type: ignore[method-assign]
        {"id": "task-1", "slug": "bitnet-cpu-ternary-kernel", "title": "BitNet"}
    ]
    client.list_work_items = lambda *, task_id=None, status="open": [  # type: ignore[method-assign]
        {
            "id": "work-claimed-self",
            "task_id": "task-1",
            "priority": 100,
            "status": "claimed",
            "claim_id": "claim-self",
            "created_at": "2026-04-01T20:13:19",
        }
    ] if status == "claimed" else [
        {
            "id": "work-open-other",
            "task_id": "task-1",
            "priority": 50,
            "status": "open",
            "claim_id": None,
            "created_at": "2026-04-01T20:14:19",
        }
    ]
    client.list_claims = lambda *, task_id=None, status=None: [  # type: ignore[method-assign]
        {
            "id": "claim-self",
            "task_id": "task-1",
            "miner_hotkey": "hk-self",
            "status": "active",
        }
    ]

    selection = client.select_task(task_id="task-1", participation_style="pool")

    assert selection.task["id"] == "task-1"
    assert selection.work_item is not None
    assert selection.work_item["id"] == "work-claimed-self"

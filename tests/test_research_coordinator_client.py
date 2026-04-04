from __future__ import annotations

from typing import Any

from miner.research_coordinator_client import CoordinatorClient


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = dict(payload)

    def json(self) -> dict[str, Any]:
        return dict(self._payload)


def test_coordinator_client_cancel_claim_posts_cancel_endpoint() -> None:
    client = CoordinatorClient(base_url="http://127.0.0.1:8787", wallet=object())
    recorded: dict[str, Any] = {}

    def fake_request(method: str, path: str, *, body=None, params=None, sign=False):  # type: ignore[no-untyped-def]
        recorded.update(
            {
                "method": method,
                "path": path,
                "body": body,
                "params": params,
                "sign": sign,
            }
        )
        return _FakeResponse({"id": "claim-123", "status": "cancelled"})

    client._request = fake_request  # type: ignore[method-assign]

    result = client.cancel_claim(claim_id="claim-123")

    assert recorded == {
        "method": "POST",
        "path": "/api/v1/claims/claim-123/cancel",
        "body": None,
        "params": None,
        "sign": True,
    }
    assert result == {"id": "claim-123", "status": "cancelled"}

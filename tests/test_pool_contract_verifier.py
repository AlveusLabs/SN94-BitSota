from __future__ import annotations

from validator.pool_contract_verifier import check_pool_contract


class _Response:
    def __init__(self, payload, *, status_code: int = 200, text: str = "json") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str, *, timeout: float):
        self.urls.append(url)
        payload = self.responses[url]
        return _Response(payload)


def _status_payload(**overrides):
    payload = {
        "status": "healthy",
        "window_number": 1144,
        "current_block": 8239292,
        "onchain_runtime": {
            "enabled": True,
            "contract_status": {
                "active_veto_count": 0,
                "is_veto_active": False,
                "claim_ready": True,
                "read_error": None,
            },
            "proof_api": {"enabled": True},
            "accounts": {"publisher": "5Publisher"},
            "processes": [{"name": "publisher", "running": True}],
            "verifier_diagnostics": {"verifier_count": 0},
        },
    }
    payload.update(overrides)
    return payload


def test_pool_contract_check_passes_for_healthy_status() -> None:
    session = _Session(
        {
            "https://pool.example/status": _status_payload(),
            "https://pool.example/claims/epochs": {
                "epochs": [123],
                "default_epoch": 123,
            },
        }
    )

    check = check_pool_contract(
        pool_url="https://pool.example",
        expected_publisher="5Publisher",
        session=session,  # type: ignore[arg-type]
    )

    assert check.ok is True
    assert check.errors == []
    assert check.claim_epochs == [123]
    assert check.default_claim_epoch == 123
    assert check.publisher_running is True


def test_pool_contract_check_warns_for_empty_claim_epochs_by_default() -> None:
    session = _Session(
        {
            "https://pool.example/status": _status_payload(),
            "https://pool.example/claims/epochs": {
                "epochs": [],
                "default_epoch": None,
            },
        }
    )

    check = check_pool_contract(pool_url="https://pool.example", session=session)  # type: ignore[arg-type]

    assert check.ok is True
    assert check.warnings == ["no claimable Pool/Merkle epoch is currently exposed"]


def test_pool_contract_check_can_require_claimable_epoch() -> None:
    session = _Session(
        {
            "https://pool.example/status": _status_payload(),
            "https://pool.example/claims/epochs": {
                "epochs": [],
                "default_epoch": None,
            },
        }
    )

    check = check_pool_contract(
        pool_url="https://pool.example",
        require_claimable_epoch=True,
        session=session,  # type: ignore[arg-type]
    )

    assert check.ok is False
    assert "no claimable Pool/Merkle epoch is currently exposed" in check.errors


def test_pool_contract_check_fails_for_veto_and_missing_publisher() -> None:
    payload = _status_payload()
    onchain = payload["onchain_runtime"]
    onchain["contract_status"]["active_veto_count"] = 2
    onchain["contract_status"]["is_veto_active"] = True
    onchain["processes"] = []
    session = _Session(
        {
            "https://pool.example/status": payload,
            "https://pool.example/claims/epochs": {
                "epochs": [123],
                "default_epoch": 123,
            },
        }
    )

    check = check_pool_contract(pool_url="https://pool.example", session=session)  # type: ignore[arg-type]

    assert check.ok is False
    assert "contract veto is active: active_veto_count=2" in check.errors
    assert "Pool publisher process is not running" in check.errors


def test_pool_contract_check_treats_positive_veto_count_as_active() -> None:
    payload = _status_payload()
    onchain = payload["onchain_runtime"]
    onchain["contract_status"]["active_veto_count"] = 1
    onchain["contract_status"]["is_veto_active"] = False
    session = _Session(
        {
            "https://pool.example/status": payload,
            "https://pool.example/claims/epochs": {
                "epochs": [123],
                "default_epoch": 123,
            },
        }
    )

    check = check_pool_contract(pool_url="https://pool.example", session=session)  # type: ignore[arg-type]

    assert check.ok is False
    assert "contract veto is active: active_veto_count=1" in check.errors

from __future__ import annotations

import requests
from types import SimpleNamespace
from pathlib import Path

from gui.merkle_claim_client import (
    MerkleClaimClient,
    _normalize_claim_package,
    resolve_claim_endpoint,
)


class _Response:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"http {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return self._payload


def test_normalize_claim_package_accepts_hotkey_fallback() -> None:
    claim = _normalize_claim_package(
        {
            "epoch": 7,
            "index": 2,
            "hotkey": "5HotkeyAddr",
            "amount_units": 123_000_000_000,
            "proof": ["0x" + ("11" * 32)],
            "root": "0x" + ("22" * 32),
        }
    )

    assert claim.epoch == 7
    assert claim.index == 2
    assert claim.recipient_address == "5HotkeyAddr"
    assert claim.identity_address == "5HotkeyAddr"
    assert claim.amount_rao == 123_000_000_000


def test_resolve_claim_endpoint_derives_local_claimserver_port() -> None:
    endpoint = resolve_claim_endpoint(
        explicit="",
        pool_endpoint="http://127.0.0.1:8434",
    )

    assert endpoint == "http://127.0.0.1:8844"


def test_list_claim_packages_skips_epochs_without_claim_for_hotkey(monkeypatch) -> None:
    responses = {
        "http://127.0.0.1:8844/epochs": _Response({"epochs": [4, 5]}),
        "http://127.0.0.1:8844/epoch/5/claim/5MinerHotkey": _Response(
            {"detail": "not found"}, status_code=404
        ),
        "http://127.0.0.1:8844/epoch/4/claim/5MinerHotkey": _Response(
            {
                "epoch": 4,
                "index": 0,
                "recipient_coldkey": "5ColdkeyAddr",
                "hotkey": "5MinerHotkey",
                "amount": 77_000_000_000,
                "proof": ["0x" + ("ab" * 32)],
                "root": "0x" + ("cd" * 32),
            }
        ),
    }

    def _fake_get(url: str, timeout: float):
        assert timeout == 10.0
        return responses[url]

    monkeypatch.setattr("gui.merkle_claim_client.requests.get", _fake_get)

    client = MerkleClaimClient(
        claim_endpoint="http://127.0.0.1:8844",
        onchain_ws_url="ws://127.0.0.1:9944",
        contract_address="5ContractAddr",
        metadata_path=__file__,
    )

    claims = client.list_claim_packages(hotkey="5MinerHotkey")

    assert len(claims) == 1
    assert claims[0].epoch == 4
    assert claims[0].recipient_address == "5ColdkeyAddr"
    assert claims[0].identity_address == "5MinerHotkey"


def test_submit_claim_uses_explicit_gas_limit(monkeypatch) -> None:
    calls = {}

    class _RuntimeConfig:
        def update_type_registry_types(self, types):
            calls["runtime_types"] = dict(types)

    class _Substrate:
        def __init__(self, url: str):
            calls["url"] = url
            self.runtime_config = _RuntimeConfig()

        def close(self):
            calls["closed"] = True

    class _Receipt:
        is_success = True
        error_message = None
        extrinsic_hash = "0x1234"

    class _Contract:
        def exec(self, signer, method: str, **kwargs):
            calls["signer"] = signer
            calls["method"] = method
            calls["kwargs"] = kwargs
            return _Receipt()

    def _fake_create_from_address(*, contract_address: str, metadata_file: str, substrate):
        calls["contract_address"] = contract_address
        calls["metadata_file"] = metadata_file
        calls["substrate"] = substrate
        return _Contract()

    monkeypatch.setattr("gui.merkle_claim_client.SubstrateInterface", _Substrate)
    monkeypatch.setattr("gui.merkle_claim_client.ContractInstance.create_from_address", _fake_create_from_address)

    client = MerkleClaimClient(
        claim_endpoint="http://127.0.0.1:8844",
        onchain_ws_url="ws://127.0.0.1:9944",
        contract_address="5ContractAddr",
        metadata_path=__file__,
    )
    claim = _normalize_claim_package(
        {
            "epoch": 2,
            "index": 1,
            "recipient_coldkey": "5ColdkeyAddr",
            "hotkey": "5MinerHotkey",
            "amount": 35_000_000_000,
            "proof": ["0x" + ("11" * 32)],
            "root": "0x" + ("22" * 32),
        }
    )

    result = client.submit_claim(signer=SimpleNamespace(name="signer"), claim=claim)

    assert result["ok"] is True
    assert calls["url"] == "ws://127.0.0.1:9944"
    assert calls["method"] == "claim_single"
    assert calls["kwargs"]["gas_limit"] == {"ref_time": 50_000_000_000, "proof_size": 2_000_000}
    assert calls["kwargs"]["storage_deposit_limit"] == 1_000_000_000
    assert calls["kwargs"]["wait_for_inclusion"] is True
    assert calls["closed"] is True


def test_list_claim_packages_filters_locally_claimed_keys(monkeypatch, tmp_path: Path) -> None:
    state_path = tmp_path / "merkle_claim_state.json"
    state_path.write_text(
        '{\n'
        '  "5ContractAddr::5MinerHotkey": ["5:0"]\n'
        '}\n',
        encoding="utf-8",
    )

    responses = {
        "http://127.0.0.1:8844/epochs": _Response({"epochs": [4, 5]}),
        "http://127.0.0.1:8844/epoch/5/claim/5MinerHotkey": _Response(
            {
                "epoch": 5,
                "index": 0,
                "hotkey": "5MinerHotkey",
                "amount_units": 99_000_000_000,
                "proof": ["0x" + ("aa" * 32)],
                "root": "0x" + ("bb" * 32),
            }
        ),
        "http://127.0.0.1:8844/epoch/4/claim/5MinerHotkey": _Response(
            {
                "epoch": 4,
                "index": 0,
                "hotkey": "5MinerHotkey",
                "amount_units": 77_000_000_000,
                "proof": ["0x" + ("ab" * 32)],
                "root": "0x" + ("cd" * 32),
            }
        ),
    }

    def _fake_get(url: str, timeout: float):
        return responses[url]

    monkeypatch.setattr("gui.merkle_claim_client._claim_state_path", lambda: state_path)
    monkeypatch.setattr("gui.merkle_claim_client.requests.get", _fake_get)

    client = MerkleClaimClient(
        claim_endpoint="http://127.0.0.1:8844",
        onchain_ws_url="ws://127.0.0.1:9944",
        contract_address="5ContractAddr",
        metadata_path=__file__,
    )

    claims = client.list_claim_packages(hotkey="5MinerHotkey")

    assert [(claim.epoch, claim.index) for claim in claims] == [(4, 0)]

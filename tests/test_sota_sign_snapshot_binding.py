from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from substrateinterface import Keypair, KeypairType


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_sign_snapshot_binding.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_sign_snapshot_binding", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    values = {
        "claims_api_url": "https://claims-api-test.example.invalid",
        "reward_address": "0x1111111111111111111111111111111111111111",
        "message_file": None,
        "wallet_name": "default",
        "wallet_path": str(tmp_path / "wallets"),
        "password_env": "BT_WALLET_PASSWORD",
        "out": tmp_path / "signed-binding.json",
        "submit": False,
        "timeout": 1.0,
        "dev_coldkey_uri": "//Alice",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _binding_response(coldkey: str) -> dict:
    message = {
        "sota_domain": "sota.base.reward-address-binding",
        "version": "1",
        "snapshot_id": "sota-genesis-test",
        "bittensor_block_number": 123,
        "bittensor_block_hash": "0x" + "ab" * 32,
        "coldkey": coldkey,
        "base_chain_id": 84532,
        "distributor_contract": "0x2222222222222222222222222222222222222222",
        "reward_address": "0x1111111111111111111111111111111111111111",
        "allocation_amount": 350000000000,
        "claim_id": "sota-genesis:123:test",
    }
    signing_payload = json.dumps(message, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "sota-snapshot-binding-message/v1",
        "status": "message_ready",
        "message": message,
        "signing_payload": signing_payload,
        "signing_payload_sha256": "abc123",
        "snapshot_claim": {"amount_units": "350000000000"},
    }


def test_signs_binding_payload_and_optionally_submits(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    alice = Keypair.create_from_uri("//Alice", crypto_type=KeypairType.SR25519)
    calls: list[tuple[str, dict]] = []

    def fake_post_json(url: str, payload: dict, **kwargs) -> dict:
        calls.append((url, payload))
        if url.endswith("/binding-message"):
            assert payload == {"coldkey": alice.ss58_address, "reward_address": "0x1111111111111111111111111111111111111111"}
            return _binding_response(alice.ss58_address)
        assert url.endswith("/bindings")
        assert alice.verify(
            _binding_response(alice.ss58_address)["signing_payload"].encode("utf-8"),
            bytes.fromhex(str(payload["signature"]).removeprefix("0x")),
        )
        return {"status": "accepted", "accepted": True, "binding_hash": "hash-1"}

    monkeypatch.setattr(module, "_post_json", fake_post_json)

    result = module.run(_args(tmp_path, submit=True))
    written = json.loads((tmp_path / "signed-binding.json").read_text(encoding="utf-8"))

    assert result["status"] == "submitted"
    assert result["submit_result"]["status"] == "accepted"
    assert written["message"]["coldkey"] == alice.ss58_address
    assert written["signature"].startswith("0x")
    assert [item[0].rsplit("/", 1)[-1] for item in calls] == ["binding-message", "bindings"]


def test_refuses_to_sign_payload_for_different_coldkey(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    bob = Keypair.create_from_uri("//Bob", crypto_type=KeypairType.SR25519)
    monkeypatch.setattr(module, "_post_json", lambda *args, **kwargs: _binding_response(bob.ss58_address))

    try:
        module.run(_args(tmp_path, submit=False))
    except ValueError as exc:
        assert "does not match binding message coldkey" in str(exc)
    else:
        raise AssertionError("expected coldkey mismatch to fail")

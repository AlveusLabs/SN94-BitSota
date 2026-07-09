from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_local_claim_proof.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_local_claim_proof", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state() -> dict:
    return {
        "chain_id": 31337,
        "accounts": {"alice_reward": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"},
        "genesis": {"old_coldkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"},
        "autoresearch": {"subnet": {"id": "base:sota-local"}},
        "urls": {
            "claims_ui": "http://127.0.0.1:3000/claims",
            "anvil_rpc": "http://127.0.0.1:8545",
        },
    }


def _args(tmp_path: Path, **overrides):
    state = tmp_path / "state.json"
    state.write_text(json.dumps(_state()) + "\n", encoding="utf-8")
    values = {
        "state": state,
        "claims_url": "",
        "rpc_url": "",
        "private_key": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
        "report_out": tmp_path / "proof.json",
        "evidence_out": tmp_path / "evidence.json",
        "timeout": 1.0,
        "reset_after": False,
        "reset_timeout": 1.0,
        "json": False,
        "allow_blocked": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _eligible(raw: int) -> dict:
    return {
        "eligible": raw > 0,
        "credits": {"unclaimed_sota": {"raw": str(raw), "formatted": str(raw)}},
    }


def _tx(label: str) -> dict:
    return {"to": "0x" + ("11" if label == "genesis" else "22") * 20, "data": "0x1234", "value": "0x0", "chainId": 31337}


def test_claim_proof_sends_ui_generated_transactions_and_runs_evidence(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    sent = []

    monkeypatch.setattr(module, "_chain_id", lambda rpc_url: 31337)
    monkeypatch.setattr(module, "_read_eligibility", lambda base_url, state, timeout: {"genesis": _eligible(15), "emission": _eligible(20)})
    monkeypatch.setattr(module, "_claim_transactions", lambda base_url, state, timeout: {"genesis": _tx("genesis"), "emission": _tx("emission")})
    monkeypatch.setattr(module, "_expect_duplicate_claim_rejected", lambda rpc_url, private_key, tx: {"rejected": True, "error": "execution reverted"})

    def fake_send(rpc_url, private_key, tx):
        sent.append(tx)
        return "0x" + str(len(sent)) * 64

    monkeypatch.setattr(module, "_send_claim_tx", fake_send)
    monkeypatch.setattr(module, "_run_evidence", lambda **kwargs: {"ok": True, "summary": {"green": 27, "yellow": 0, "red": 0}})

    report = module.run_proof(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert sent == [_tx("genesis"), _tx("emission")]
    assert report["transactions"]["genesis"]["tx_hash"] == "0x" + "1" * 64
    assert report["transactions"]["emission"]["tx_hash"] == "0x" + "2" * 64
    assert report["double_spend_checks"]["genesis"]["rejected"] is True
    assert report["double_spend_checks"]["emission"]["rejected"] is True
    assert report["evidence_summary"] == {"green": 27, "yellow": 0, "red": 0}


def test_claim_proof_refuses_nonlocal_chain_before_reading_claims(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    monkeypatch.setattr(module, "_chain_id", lambda rpc_url: 84532)
    monkeypatch.setattr(module, "_read_eligibility", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not read claims")))

    report = module.run_proof(args)

    assert report["ok"] is False
    assert report["status"] == "red"
    assert report["transactions"] == {}
    assert "local Anvil" in report["checks"][0]["detail"]


def test_claim_proof_refuses_already_claimed_state(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    monkeypatch.setattr(module, "_chain_id", lambda rpc_url: 31337)
    monkeypatch.setattr(module, "_read_eligibility", lambda base_url, state, timeout: {"genesis": _eligible(0), "emission": _eligible(20)})
    monkeypatch.setattr(module, "_claim_transactions", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not build txs")))

    report = module.run_proof(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is False
    assert report["transactions"] == {}
    assert checks["genesis_unclaimed"]["status"] == "red"
    assert checks["genesis_unclaimed"]["remediation"] == "Reset the local stack before running a fresh claim proof."


def test_claim_proof_can_reset_after_success(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, reset_after=True)
    reset = {}

    monkeypatch.setattr(module, "_chain_id", lambda rpc_url: 31337)
    monkeypatch.setattr(module, "_read_eligibility", lambda base_url, state, timeout: {"genesis": _eligible(15), "emission": _eligible(20)})
    monkeypatch.setattr(module, "_claim_transactions", lambda base_url, state, timeout: {"genesis": _tx("genesis"), "emission": _tx("emission")})
    monkeypatch.setattr(module, "_send_claim_tx", lambda rpc_url, private_key, tx: "0x" + tx["to"][2:4] * 32)
    monkeypatch.setattr(module, "_expect_duplicate_claim_rejected", lambda rpc_url, private_key, tx: {"rejected": True, "error": "execution reverted"})
    monkeypatch.setattr(module, "_run_evidence", lambda **kwargs: {"ok": True, "summary": {"green": 27, "yellow": 0, "red": 0}})

    def fake_restart(timeout):
        reset["timeout"] = timeout
        return "SOTA Base local demo is ready."

    monkeypatch.setattr(module, "_restart_local_stack", fake_restart)

    report = module.run_proof(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is True
    assert checks["reset_after"]["status"] == "green"
    assert reset["timeout"] == 1.0
    assert "ready" in report["reset_stdout_tail"]


def test_claim_proof_reset_launches_without_recursive_proof(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="SOTA Base local demo is ready.", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    output = module._restart_local_stack(12.0)

    assert output == "SOTA Base local demo is ready."
    command, kwargs = calls[0]
    assert command[-3:] == ["launch", "--skip-claim-proof", "--skip-miner-swarm-proof"]
    assert kwargs["timeout"] == 12.0


def test_claim_proof_marks_duplicate_claim_acceptance_red(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)

    monkeypatch.setattr(module, "_chain_id", lambda rpc_url: 31337)
    monkeypatch.setattr(module, "_read_eligibility", lambda base_url, state, timeout: {"genesis": _eligible(15), "emission": _eligible(20)})
    monkeypatch.setattr(module, "_claim_transactions", lambda base_url, state, timeout: {"genesis": _tx("genesis"), "emission": _tx("emission")})
    monkeypatch.setattr(module, "_send_claim_tx", lambda rpc_url, private_key, tx: "0x" + "1" * 64)
    monkeypatch.setattr(module, "_expect_duplicate_claim_rejected", lambda rpc_url, private_key, tx: {"rejected": False, "tx_hash": "0x" + "2" * 64})
    monkeypatch.setattr(module, "_run_evidence", lambda **kwargs: {"ok": True, "summary": {"green": 27, "yellow": 0, "red": 0}})

    report = module.run_proof(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is False
    assert checks["genesis_double_spend_rejected"]["status"] == "red"
    assert checks["emission_double_spend_rejected"]["status"] == "red"
    assert report["next_actions"] == [
        "Fix distributor claimed-leaf checks before releasing testnet testers.",
        "Fix distributor claimed-leaf checks before releasing testnet testers.",
    ]

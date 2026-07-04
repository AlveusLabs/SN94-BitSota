from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_claim_tx_evidence.py"

WALLET = "0x5555555555555555555555555555555555555555"
TOKEN = "0x0000000000000000000000000000000000000001"
VAULT = "0x0000000000000000000000000000000000000002"
GENESIS = "0x0000000000000000000000000000000000000005"
EMISSION = "0x0000000000000000000000000000000000000006"
GENESIS_TX = "0x" + "a" * 64
EMISSION_TX = "0x" + "b" * 64


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_claim_tx_evidence", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _topic_address(address: str) -> str:
    return "0x" + address.lower()[2:].rjust(64, "0")


def _word(value: int | str) -> str:
    if isinstance(value, str) and value.startswith("0x"):
        return value[2:].rjust(64, "0")
    return f"{int(value):064x}"


def _log(address: str, topic0: str, topics: list[str], data_words: list[int | str]) -> dict:
    return {
        "address": address,
        "topics": [topic0, *topics],
        "data": "0x" + "".join(_word(item) for item in data_words),
    }


def _args(tmp_path: Path, **overrides):
    artifacts_dir = tmp_path / "artifacts"
    values = {
        "environment": "testnet",
        "state": None,
        "artifacts_dir": artifacts_dir,
        "manifest": artifacts_dir / "base-sepolia-deployment-manifest.json",
        "env_file": artifacts_dir / "base-sota.env.testnet",
        "rpc_url": "https://sepolia.example.invalid",
        "chain_id": 0,
        "wallet_address": "",
        "sota_token_address": "",
        "vault_address": "",
        "genesis_distributor_address": "",
        "emission_distributor_address": "",
        "genesis_tx": GENESIS_TX,
        "emission_tx": EMISSION_TX,
        "timeout": 0.1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_artifacts(args: argparse.Namespace, *, chain_id: str = "84532") -> None:
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "chain": {"chain_id": 84532, "public_browser_rpc_url": "https://sepolia.example.invalid"},
                "contracts": {
                    "sota_token": {"address": TOKEN},
                    "vault": {"address": VAULT},
                    "genesis_distributor": {"address": GENESIS},
                    "emission_distributor": {"address": EMISSION},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args.env_file.write_text(
        "\n".join(
            [
                f"SOTA_BASE_CHAIN_ID={chain_id}",
                "SOTA_BASE_RPC_URL=https://sepolia.example.invalid",
                f"SOTA_TEST_WALLET_ADDRESS={WALLET}",
                f"SOTA_TOKEN_ADDRESS={TOKEN}",
                f"SOTA_VAULT_ADDRESS={VAULT}",
                f"SOTA_GENESIS_DISTRIBUTOR_ADDRESS={GENESIS}",
                f"SOTA_EMISSION_DISTRIBUTOR_ADDRESS={EMISSION}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _receipt(module, *, label: str, transfer: bool = True, to: str | None = None) -> dict:
    is_genesis = label == "genesis"
    distributor = GENESIS if is_genesis else EMISSION
    event_topic = module.GENESIS_CLAIMED_TOPIC if is_genesis else module.EMISSION_CLAIMED_TOPIC
    amount = 1_500_000_000_000_000_000 if is_genesis else 2_000_000_000_000_000_000
    root = "0x" + ("1" if is_genesis else "2") * 64
    leaf = "0x" + ("3" if is_genesis else "4") * 64
    logs = [
        _log(
            distributor,
            module.CLAIM_RECORDED_TOPIC,
            [root, leaf, _topic_address(WALLET)],
            [amount, amount],
        )
    ]
    if is_genesis:
        logs.append(
            _log(
                distributor,
                event_topic,
                [root, _topic_address(WALLET)],
                [amount, "0x" + "5" * 64, leaf],
            )
        )
    else:
        logs.append(
            _log(
                distributor,
                event_topic,
                [root, "0x" + "6" * 64, _topic_address(WALLET)],
                [1, amount, "0x" + "7" * 64, leaf],
            )
        )
    if transfer:
        logs.append(_log(TOKEN, module.TRANSFER_TOPIC, [_topic_address(VAULT), _topic_address(WALLET)], [amount]))
        logs.append(_log(VAULT, module.SOTA_RELEASED_TOPIC, [_topic_address(distributor), _topic_address(WALLET)], [amount]))
    return {
        "status": "0x1",
        "blockNumber": "0x7b",
        "transactionIndex": "0x0",
        "gasUsed": "0x5208",
        "to": to or distributor,
        "logs": logs,
    }


def _tx(module, *, label: str, chain_id: str = "0x14a34", to: str | None = None) -> dict:
    selector = module.GENESIS_CLAIM_SELECTOR if label == "genesis" else module.EMISSION_CLAIM_SELECTOR
    distributor = GENESIS if label == "genesis" else EMISSION
    return {
        "from": WALLET,
        "to": to or distributor,
        "chainId": chain_id,
        "input": selector + "00" * 64,
    }


def _install_rpc(module, monkeypatch, *, chain_id: str = "0x14a34", emission_transfer: bool = True, emission_to: str | None = None):
    def fake_rpc(rpc_url: str, method: str, params=None, timeout: float = 0.1):
        if method == "eth_chainId":
            return chain_id
        if method == "eth_getTransactionReceipt":
            if params[0] == GENESIS_TX:
                return _receipt(module, label="genesis")
            if params[0] == EMISSION_TX:
                return _receipt(module, label="emission", transfer=emission_transfer, to=emission_to)
        if method == "eth_getTransactionByHash":
            if params[0] == GENESIS_TX:
                return _tx(module, label="genesis", chain_id=chain_id)
            if params[0] == EMISSION_TX:
                return _tx(module, label="emission", chain_id=chain_id, to=emission_to)
        if method == "eth_call":
            return hex(3_500_000_000_000_000_000)
        raise AssertionError(f"unexpected rpc {method} {params}")

    monkeypatch.setattr(module, "_json_rpc", fake_rpc)


def test_claim_tx_evidence_green_for_valid_receipts(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    _install_rpc(module, monkeypatch)

    report = module.run_evidence(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["summary"]["red"] == 0
    names = {check["name"]: check for check in report["checks"]}
    assert names["genesis_claim_event"]["status"] == "green"
    assert names["emission_claim_event"]["status"] == "green"
    assert names["sota_balance"]["status"] == "green"


def test_claim_tx_evidence_rejects_base_mainnet_chain_id(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args, chain_id="8453")
    _install_rpc(module, monkeypatch, chain_id="0x2105")

    report = module.run_evidence(args)
    chain_config = next(check for check in report["checks"] if check["name"] == "chain_config")

    assert report["ok"] is False
    assert chain_config["status"] == "red"
    assert "expected Base Sepolia 84532" in chain_config["detail"]


def test_claim_tx_evidence_rejects_wrong_distributor(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    _install_rpc(module, monkeypatch, emission_to="0x9999999999999999999999999999999999999999")

    report = module.run_evidence(args)
    to_distributor = next(check for check in report["checks"] if check["name"] == "emission_to_distributor")

    assert report["ok"] is False
    assert to_distributor["status"] == "red"


def test_claim_tx_evidence_rejects_missing_transfer_event(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    _install_rpc(module, monkeypatch, emission_transfer=False)

    report = module.run_evidence(args)
    transfer = next(check for check in report["checks"] if check["name"] == "emission_sota_transfer")

    assert report["ok"] is False
    assert transfer["status"] == "red"


def test_claim_tx_evidence_supports_local_state_config(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "chain_id": 31337,
                "urls": {"anvil_rpc": "http://127.0.0.1:8545"},
                "accounts": {"alice_reward": WALLET},
                "contracts": {
                    "sota_token": TOKEN,
                    "vault": VAULT,
                    "genesis_distributor": GENESIS,
                    "emission_distributor": EMISSION,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _args(tmp_path, environment="local", state=state, chain_id=31337)
    _install_rpc(module, monkeypatch, chain_id="0x7a69")

    report = module.run_evidence(args)

    assert report["ok"] is True
    assert report["config"]["expected_chain_id"] == "31337"

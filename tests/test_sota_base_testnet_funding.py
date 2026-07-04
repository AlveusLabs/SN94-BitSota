from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_funding.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_funding", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    values = {
        "rpc_url": "https://sepolia.base.org",
        "aws_profile": "moonrocklab-frankfurt",
        "region": "eu-central-1",
        "deployer_secret_id": "base-sota/test/base-sepolia/deployer",
        "root_publisher_secret_id": "base-sota/test/base-sepolia/root-publisher",
        "test_wallet_address": "0x00000000000000000000000000000000000000bb",
        "local_state": tmp_path / "state.json",
        "fallback_report": [],
        "target": [],
        "min_balance": [],
        "timeout": 1.0,
        "report_out": tmp_path / "funding.json",
        "json": False,
        "allow_blocked": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_funding_report_green_when_required_targets_have_gas(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)

    def fake_aws(command, *, profile, region, timeout):
        assert "get-secret-value" not in command
        if command[:2] == ["sts", "get-caller-identity"]:
            return {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/test"}
        if command[:2] == ["secretsmanager", "describe-secret"]:
            suffix = "aa" if command[-1].endswith("/deployer") else "cc"
            return {"Tags": [{"Key": "sota-address", "Value": f"0x00000000000000000000000000000000000000{suffix}"}]}
        raise AssertionError(command)

    monkeypatch.setattr(module, "_run_aws", fake_aws)
    monkeypatch.setattr(module, "_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_balance_wei", lambda rpc_url, address, timeout: 10**18)

    report = module.build_report(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["read_secret_values"] is False
    assert "read_secret_values" in report["does_not"]
    assert [target["label"] for target in report["funding_targets"]] == [
        "deployer",
        "root_publisher",
        "test_wallet",
    ]
    deployer = next(target for target in report["funding_targets"] if target["label"] == "deployer")
    assert deployer["minimum_balance_eth"] == "0.02000000"
    assert deployer["needed_eth"] == "0.00000000"
    assert deployer["explorer_url"].endswith(deployer["address"])
    assert report["faucet_sources"][0]["url"].startswith("https://docs.base.org/")


def test_funding_report_marks_zero_balances_red(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    monkeypatch.setattr(
        module,
        "_run_aws",
        lambda command, **kwargs: (
            {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/test"}
            if command[:2] == ["sts", "get-caller-identity"]
            else {"Tags": [{"Key": "sota-address", "Value": "0x00000000000000000000000000000000000000aa"}]}
        ),
    )
    monkeypatch.setattr(module, "_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_balance_wei", lambda rpc_url, address, timeout: 0)

    report = module.build_report(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is False
    assert report["status"] == "red"
    assert checks["funding_deployer"]["status"] == "red"
    assert checks["funding_root_publisher"]["status"] == "red"
    assert checks["funding_test_wallet"]["status"] == "red"
    assert "Fund 0x00000000000000000000000000000000000000aa" in checks["funding_deployer"]["remediation"]
    deployer = next(target for target in report["funding_targets"] if target["label"] == "deployer")
    assert deployer["minimum_balance_eth"] == "0.02000000"
    assert deployer["needed_eth"] == "0.02000000"


def test_funding_report_marks_underfunded_balances_red(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    monkeypatch.setattr(
        module,
        "_run_aws",
        lambda command, **kwargs: (
            {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/test"}
            if command[:2] == ["sts", "get-caller-identity"]
            else {"Tags": [{"Key": "sota-address", "Value": "0x00000000000000000000000000000000000000aa"}]}
        ),
    )
    monkeypatch.setattr(module, "_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_balance_wei", lambda rpc_url, address, timeout: 10**15)

    report = module.build_report(args)
    deployer = next(target for target in report["funding_targets"] if target["label"] == "deployer")

    assert report["ok"] is False
    assert report["status"] == "red"
    assert deployer["balance_eth"] == "0.00100000"
    assert deployer["needed_eth"] == "0.01900000"
    assert "minimum is 0.02000000 ETH" in next(check for check in report["checks"] if check["name"] == "funding_deployer")["detail"]


def test_funding_report_accepts_custom_minimums(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, min_balance=["deployer=0.001", "root_publisher=0.001", "test_wallet=0.001"])
    monkeypatch.setattr(
        module,
        "_run_aws",
        lambda command, **kwargs: (
            {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/test"}
            if command[:2] == ["sts", "get-caller-identity"]
            else {"Tags": [{"Key": "sota-address", "Value": "0x00000000000000000000000000000000000000aa"}]}
        ),
    )
    monkeypatch.setattr(module, "_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_balance_wei", lambda rpc_url, address, timeout: 10**15)

    report = module.build_report(args)

    assert report["ok"] is True
    assert report["status"] == "green"


def test_funding_report_can_use_cached_public_addresses_when_aws_tags_fail(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    blocker_report = tmp_path / "blockers.json"
    blocker_report.write_text(
        json.dumps(
            {
                "schema": "sota-base-testnet-blockers/v1",
                "checks": [
                    {
                        "name": "gas_deployer",
                        "status": "red",
                        "detail": "deployer 0x00000000000000000000000000000000000000aa has 0 ETH.",
                    },
                    {
                        "name": "gas_root_publisher",
                        "status": "red",
                        "detail": "root 0x00000000000000000000000000000000000000cc has 0 ETH.",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _args(
        tmp_path,
        fallback_report=[blocker_report],
        min_balance=["deployer=0.001", "root_publisher=0.001", "test_wallet=0.001"],
    )

    def fake_aws(command, **kwargs):
        if command[:2] == ["sts", "get-caller-identity"]:
            return {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/test"}
        raise RuntimeError("AccessDenied")

    monkeypatch.setattr(module, "_run_aws", fake_aws)
    monkeypatch.setattr(module, "_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_balance_wei", lambda rpc_url, address, timeout: 10**15)

    report = module.build_report(args)
    deployer = next(target for target in report["funding_targets"] if target["label"] == "deployer")
    root_publisher = next(target for target in report["funding_targets"] if target["label"] == "root_publisher")

    assert report["ok"] is True
    assert deployer["address"] == "0x00000000000000000000000000000000000000aa"
    assert deployer["source"] == f"fallback-report:{blocker_report}"
    assert deployer["fallback_reason"] == "AccessDenied"
    assert root_publisher["address"] == "0x00000000000000000000000000000000000000cc"


def test_funding_report_can_load_test_wallet_from_local_state(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps({"accounts": {"alice_reward": "0x00000000000000000000000000000000000000dd"}}) + "\n",
        encoding="utf-8",
    )
    args = _args(tmp_path, test_wallet_address="", local_state=state)
    monkeypatch.setattr(
        module,
        "_run_aws",
        lambda command, **kwargs: (
            {"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/test"}
            if command[:2] == ["sts", "get-caller-identity"]
            else {"Tags": [{"Key": "sota-address", "Value": "0x00000000000000000000000000000000000000aa"}]}
        ),
    )
    monkeypatch.setattr(module, "_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_balance_wei", lambda rpc_url, address, timeout: 10**18)

    report = module.build_report(args)
    test_wallet = next(target for target in report["funding_targets"] if target["label"] == "test_wallet")

    assert test_wallet["address"] == "0x00000000000000000000000000000000000000dd"
    assert test_wallet["source"] == str(state)


def test_funding_main_writes_report(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "build_report",
        lambda args: {
            "schema": "sota-base-testnet-funding/v1",
            "ok": True,
            "status": "green",
            "summary": {"green": 1, "yellow": 0, "red": 0},
            "funding_targets": [],
            "checks": [],
        },
    )
    out = tmp_path / "funding.json"

    exit_code = module.main(["--report-out", str(out), "--json"])

    assert exit_code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "sota-base-testnet-funding/v1"

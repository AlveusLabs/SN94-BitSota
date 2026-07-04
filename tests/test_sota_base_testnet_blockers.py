from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import socket
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_blockers.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_blockers", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    artifacts_dir = tmp_path / "artifacts"
    values = {
        "artifacts_dir": artifacts_dir,
        "deployment": artifacts_dir / "base-sepolia-compact-deployment.json",
        "manifest": artifacts_dir / "base-sepolia-deployment-manifest.json",
        "env_file": artifacts_dir / "base-sota.env.testnet",
        "readiness_file": artifacts_dir / "base-sota-testnet-readiness.json",
        "service_pack": artifacts_dir / "base-sota-testnet-service-pack.json",
        "container_pack": artifacts_dir / "base-sota-testnet-container-pack.json",
        "apprunner_source_pack": artifacts_dir / "base-sota-testnet-apprunner-source-pack.json",
        "readiness_url": "https://claims-test.example.invalid/base-sota-testnet-readiness.json",
        "rpc_url": "https://sepolia.base.org",
        "host": [],
        "timeout": 0.1,
        "skip_aws": False,
        "aws_profile": "",
        "skip_readiness_url": False,
        "deployer_secret_id": "base-sota/test/base-sepolia/deployer",
        "root_publisher_secret_id": "base-sota/test/base-sepolia/root-publisher",
        "gas_address": [],
        "skip_gas": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_artifacts(args: argparse.Namespace, *, readiness_ok: bool = True, env_chain_id: str = "84532") -> None:
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.deployment.write_text(json.dumps({"contracts": {}}) + "\n", encoding="utf-8")
    args.manifest.write_text(json.dumps({"environment": "base-sepolia"}) + "\n", encoding="utf-8")
    args.env_file.write_text(f"NEXT_PUBLIC_SOTA_BASE_CHAIN_ID={env_chain_id}\n", encoding="utf-8")
    args.readiness_file.write_text(
        json.dumps(
            {
                "schema": "sota-base-testnet-readiness/v1",
                "ok": readiness_ok,
                "status": "green" if readiness_ok else "red",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args.service_pack.write_text(
        json.dumps(
            {
                "schema": "sota-base-testnet-service-pack/v1",
                "ok": True,
                "status": "green",
                "deployment_ready": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args.container_pack.write_text(
        json.dumps(
            {
                "schema": "sota-base-testnet-container-pack/v1",
                "ok": True,
                "status": "green",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args.apprunner_source_pack.write_text(
        json.dumps(
            {
                "schema": "sota-base-testnet-apprunner-source-pack/v1",
                "ok": True,
                "status": "green",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_blocker_report_green_when_external_and_artifact_checks_pass(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    monkeypatch.setattr(module, "_aws_identity_payload", lambda timeout, profile="": {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"})
    monkeypatch.setattr(module, "_rpc_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_secret_tag", lambda secret_id, tag_key, profile="", timeout=0.1: "0x00000000000000000000000000000000000000aa")
    monkeypatch.setattr(module, "_native_balance_wei", lambda rpc_url, address, timeout=0.1: 1)
    monkeypatch.setattr(module, "_resolve_host", lambda host, timeout: ["192.0.2.10"])
    monkeypatch.setattr(module, "_http_status", lambda url, timeout: (200, "ok"))

    report = module.run_blocker_report(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["summary"]["red"] == 0
    assert report["read_only"] is True
    assert "broadcast_transactions" in report["does_not"]


def test_blocker_report_marks_missing_infra_and_artifacts_red(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    monkeypatch.setattr(module, "_aws_identity_payload", lambda timeout, profile="": (_ for _ in ()).throw(RuntimeError("NoCredentials")))
    monkeypatch.setattr(module, "_rpc_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_secret_tag", lambda secret_id, tag_key, profile="", timeout=0.1: (_ for _ in ()).throw(RuntimeError("AccessDenied")))
    monkeypatch.setattr(module, "_resolve_host", lambda host, timeout: (_ for _ in ()).throw(socket.gaierror("missing dns")))
    monkeypatch.setattr(module, "_http_status", lambda url, timeout: (None, "missing dns"))

    report = module.run_blocker_report(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is False
    assert report["status"] == "red"
    assert checks["aws_identity"]["status"] == "red"
    assert checks["dns_claims_ui"]["status"] == "red"
    assert checks["artifact_compact_deployment"]["status"] == "red"
    assert checks["artifact_service_pack"]["status"] == "red"
    assert checks["artifact_apprunner_source_pack"]["status"] == "red"
    assert checks["artifact_readiness"]["status"] == "red"


def test_blocker_report_rejects_base_mainnet_rpc(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    monkeypatch.setattr(module, "_aws_identity_payload", lambda timeout, profile="": {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"})
    monkeypatch.setattr(module, "_rpc_chain_id", lambda rpc_url, timeout: 8453)
    monkeypatch.setattr(module, "_secret_tag", lambda secret_id, tag_key, profile="", timeout=0.1: "0x00000000000000000000000000000000000000aa")
    monkeypatch.setattr(module, "_native_balance_wei", lambda rpc_url, address, timeout=0.1: 1)
    monkeypatch.setattr(module, "_resolve_host", lambda host, timeout: ["192.0.2.10"])
    monkeypatch.setattr(module, "_http_status", lambda url, timeout: (200, "ok"))

    report = module.run_blocker_report(args)
    rpc_check = next(check for check in report["checks"] if check["name"] == "base_sepolia_rpc")

    assert report["ok"] is False
    assert rpc_check["status"] == "red"
    assert "mainnet" in rpc_check["detail"].lower()


def test_blocker_report_rejects_readiness_artifact_that_is_not_ok(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args, readiness_ok=False)
    monkeypatch.setattr(module, "_aws_identity_payload", lambda timeout, profile="": {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"})
    monkeypatch.setattr(module, "_rpc_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_secret_tag", lambda secret_id, tag_key, profile="", timeout=0.1: "0x00000000000000000000000000000000000000aa")
    monkeypatch.setattr(module, "_native_balance_wei", lambda rpc_url, address, timeout=0.1: 1)
    monkeypatch.setattr(module, "_resolve_host", lambda host, timeout: ["192.0.2.10"])
    monkeypatch.setattr(module, "_http_status", lambda url, timeout: (200, "ok"))

    report = module.run_blocker_report(args)
    readiness_check = next(check for check in report["checks"] if check["name"] == "artifact_readiness")

    assert report["ok"] is False
    assert readiness_check["status"] == "red"
    assert "ok=false" in readiness_check["detail"]


def test_blocker_report_rejects_service_pack_that_is_not_deployment_ready(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    service_pack = json.loads(args.service_pack.read_text(encoding="utf-8"))
    service_pack["status"] = "yellow"
    service_pack["deployment_ready"] = False
    service_pack["next_actions"] = ["Implement root publisher worker."]
    args.service_pack.write_text(json.dumps(service_pack) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "_aws_identity_payload", lambda timeout, profile="": {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"})
    monkeypatch.setattr(module, "_rpc_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_secret_tag", lambda secret_id, tag_key, profile="", timeout=0.1: "0x00000000000000000000000000000000000000aa")
    monkeypatch.setattr(module, "_native_balance_wei", lambda rpc_url, address, timeout=0.1: 1)
    monkeypatch.setattr(module, "_resolve_host", lambda host, timeout: ["192.0.2.10"])
    monkeypatch.setattr(module, "_http_status", lambda url, timeout: (200, "ok"))

    report = module.run_blocker_report(args)
    service_pack_check = next(check for check in report["checks"] if check["name"] == "artifact_service_pack")

    assert report["ok"] is False
    assert service_pack_check["status"] == "red"
    assert "deployment_ready=false" in service_pack_check["detail"]
    assert service_pack_check["remediation"] == "Implement root publisher worker."


def test_blocker_report_marks_source_pack_yellow_when_branch_is_not_ready(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, skip_readiness_url=True)
    _write_artifacts(args)
    source_pack = json.loads(args.apprunner_source_pack.read_text(encoding="utf-8"))
    source_pack["ok"] = False
    source_pack["status"] = "yellow"
    source_pack["next_actions"] = ["Commit and push the Base SOTA service changes."]
    args.apprunner_source_pack.write_text(json.dumps(source_pack) + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "_aws_identity_payload", lambda timeout, profile="": {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"})
    monkeypatch.setattr(module, "_rpc_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_secret_tag", lambda secret_id, tag_key, profile="", timeout=0.1: "0x00000000000000000000000000000000000000aa")
    monkeypatch.setattr(module, "_native_balance_wei", lambda rpc_url, address, timeout=0.1: 1)
    monkeypatch.setattr(module, "_resolve_host", lambda host, timeout: ["192.0.2.10"])

    report = module.run_blocker_report(args)
    source_pack_check = next(check for check in report["checks"] if check["name"] == "artifact_apprunner_source_pack")

    assert report["ok"] is False
    assert report["status"] == "yellow"
    assert source_pack_check["status"] == "yellow"
    assert source_pack_check["remediation"] == "Commit and push the Base SOTA service changes."


def test_blocker_report_passes_aws_profile_to_sts(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, aws_profile="moonrocklab-frankfurt", skip_readiness_url=True)
    _write_artifacts(args)
    seen = {}

    def fake_identity(timeout, profile=""):
        seen["profile"] = profile
        return {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"}

    monkeypatch.setattr(module, "_aws_identity_payload", fake_identity)
    monkeypatch.setattr(module, "_rpc_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_secret_tag", lambda secret_id, tag_key, profile="", timeout=0.1: "0x00000000000000000000000000000000000000aa")
    monkeypatch.setattr(module, "_native_balance_wei", lambda rpc_url, address, timeout=0.1: 1)
    monkeypatch.setattr(module, "_resolve_host", lambda host, timeout: ["192.0.2.10"])

    report = module.run_blocker_report(args)
    aws_check = next(check for check in report["checks"] if check["name"] == "aws_identity")

    assert report["status"] == "yellow"
    assert seen["profile"] == "moonrocklab-frankfurt"
    assert "moonrocklab-frankfurt" in aws_check["detail"]


def test_blocker_report_marks_zero_gas_signers_and_wallet_red(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(
        tmp_path,
        skip_readiness_url=True,
        gas_address=["test_wallet=0x00000000000000000000000000000000000000bb"],
    )
    _write_artifacts(args)
    monkeypatch.setattr(module, "_aws_identity_payload", lambda timeout, profile="": {"Account": "123456789012", "Arn": "arn:aws:iam::123456789012:user/test"})
    monkeypatch.setattr(module, "_rpc_chain_id", lambda rpc_url, timeout: 84532)
    monkeypatch.setattr(module, "_secret_tag", lambda secret_id, tag_key, profile="", timeout=0.1: "0x00000000000000000000000000000000000000aa")
    monkeypatch.setattr(module, "_native_balance_wei", lambda rpc_url, address, timeout=0.1: 0)
    monkeypatch.setattr(module, "_resolve_host", lambda host, timeout: ["192.0.2.10"])

    report = module.run_blocker_report(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is False
    assert checks["gas_deployer"]["status"] == "red"
    assert checks["gas_root_publisher"]["status"] == "red"
    assert checks["gas_test_wallet"]["status"] == "red"
    assert "Fund 0x00000000000000000000000000000000000000aa" in checks["gas_deployer"]["remediation"]
    assert "Fund 0x00000000000000000000000000000000000000bb" in checks["gas_test_wallet"]["remediation"]


def test_blocker_json_without_report_out_only_prints(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module,
        "run_blocker_report",
        lambda args: {
            "schema": "sota-base-testnet-blockers/v1",
            "ok": True,
            "status": "green",
            "checks": [],
            "summary": {"green": 0, "yellow": 0, "red": 0},
        },
    )

    exit_code = module.main(["--json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "green"
    assert tmp_path.is_dir()

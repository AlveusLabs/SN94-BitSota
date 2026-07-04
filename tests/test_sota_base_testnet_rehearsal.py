from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_rehearsal.py"
TEMPLATE = REPO / "docs" / "base" / "manifests" / "base-sepolia-deployment-manifest.template.json"


def _load_module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("sota_base_testnet_rehearsal", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contract(name: str, index: int) -> dict[str, object]:
    return {
        "name": name,
        "address": f"0x{index:040x}",
        "deployment_tx_hash": "0x" + f"{index:064x}",
        "deployment_block": 12345 + index,
        "constructor_args": ["fixture"],
    }


def _deployment(tmp_path: Path) -> Path:
    payload = {
        "version": "sota-base-deployment-v1",
        "environment": "base-sepolia",
        "chain_id": 84532,
        "chain_name": "base-sepolia",
        "block_explorer_url": "https://sepolia.basescan.org",
        "deployment_block": 12345,
        "deployer": "0x1111111111111111111111111111111111111111",
        "roles": {
            "owner": "0x1111111111111111111111111111111111111111",
            "supply_authority": "0x2222222222222222222222222222222222222222",
            "emission_authority": "0x3333333333333333333333333333333333333333",
            "root_publisher": "0x4444444444444444444444444444444444444444",
        },
        "contracts": {
            "sota_token": _contract("SOTAToken", 1),
            "vault": _contract("SOTAVault", 2),
            "root_registry": _contract("SOTARootRegistry", 3),
            "lane_registry": _contract("SOTALaneRegistry", 4),
            "genesis_distributor": _contract("GenesisClaimDistributor", 5),
            "emission_distributor": _contract("EmissionClaimDistributor", 6),
        },
    }
    path = tmp_path / "compact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_rehearsal_generates_artifacts_and_offline_preflight_without_red(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "assert_base_sepolia_rpc", lambda rpc_url, timeout: 84532)
    report_path = tmp_path / "report.json"

    code = module.main(
        [
            "--deployment",
            str(_deployment(tmp_path)),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--template",
            str(TEMPLATE),
            "--claims-ui-url",
            "https://claims-test.example.invalid",
            "--claims-ui-health-url",
            "https://claims-test.example.invalid/health",
            "--indexer-api-url",
            "https://claims-api-test.example.invalid",
            "--indexer-api-health-url",
            "https://claims-api-test.example.invalid/health",
            "--root-publisher-url",
            "https://root-test.example.invalid",
            "--root-publisher-health-url",
            "https://root-test.example.invalid/health",
            "--attestation-builder-url",
            "https://attestation-test.example.invalid",
            "--attestation-builder-health-url",
            "https://attestation-test.example.invalid/health",
            "--monitoring-url",
            "https://monitoring-test.example.invalid",
            "--autoresearch-api-url",
            "https://coordinator-test.example.invalid",
            "--test-wallet-address",
            "0x5555555555555555555555555555555555555555",
            "--test-old-coldkey",
            "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "--test-epoch",
            "3",
            "--offline",
            "--allow-blocked",
            "--report-out",
            str(report_path),
        ]
    )

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["deployed"] is False
    assert Path(report["manifest"]).exists()
    assert Path(report["env"]).exists()
    assert Path(report["readiness"]).exists()
    assert report["preflight"]["summary"]["red"] == 0
    assert report["preflight"]["summary"]["yellow"] > 0
    readiness = json.loads(Path(report["readiness"]).read_text(encoding="utf-8"))
    assert readiness["schema"] == "sota-base-testnet-readiness/v1"
    assert readiness["status"] == "yellow"
    assert readiness["ok"] is False
    assert "not open" in readiness["tester_message"]
    assert readiness["checks"]


def test_rehearsal_requires_deployment_or_deploy_flag() -> None:
    module = _load_module()

    with pytest.raises(SystemExit, match="provide --deployment"):
        module.main(["--offline", "--allow-blocked"])


def test_rehearsal_refuses_base_mainnet_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_chain_id", lambda rpc_url, timeout: 8453)

    with pytest.raises(SystemExit, match="mainnet"):
        module.assert_base_sepolia_rpc("https://mainnet.base.org", timeout=1)


def test_rehearsal_deploy_requires_private_key_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "assert_base_sepolia_rpc", lambda rpc_url, timeout: 84532)
    monkeypatch.delenv("SOTA_DEPLOYER_PRIVATE_KEY", raising=False)

    with pytest.raises(SystemExit, match="SOTA_DEPLOYER_PRIVATE_KEY"):
        module.main(["--deploy", "--artifacts-dir", str(tmp_path), "--offline", "--allow-blocked"])


def test_rehearsal_run_surfaces_child_stderr_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="deployer has no native gas balance\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="deployer has no native gas balance"):
        module._run(["deploy"], cwd=REPO)

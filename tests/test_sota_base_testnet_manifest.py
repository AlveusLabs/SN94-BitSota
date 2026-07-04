from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_manifest.py"
PREFLIGHT = REPO / "scripts" / "sota_base_testnet_preflight.py"
VALIDATOR = REPO / "scripts" / "validate_base_sota_manifest.py"
TEMPLATE = REPO / "docs" / "base" / "manifests" / "base-sepolia-deployment-manifest.template.json"


def _load_module(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _deployment(tmp_path: Path, *, chain_id: int = 84532) -> Path:
    payload = {
        "version": "sota-base-deployment-v1",
        "environment": "base-sepolia",
        "chain_id": chain_id,
        "chain_name": "base-sepolia" if chain_id == 84532 else "base",
        "block_explorer_url": "https://sepolia.basescan.org",
        "deployment_block": 12345,
        "deployed_at": "2026-07-04T00:00:00+00:00",
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
        "post_deploy_checks": {
            "root_publisher_enabled": True,
            "genesis_releaser_enabled": True,
            "emission_releaser_enabled": True,
            "vault_sota_balance_units": "1000000000000000000000",
        },
    }
    path = tmp_path / "compact-deployment.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _contract(name: str, index: int) -> dict[str, object]:
    return {
        "name": name,
        "address": f"0x{index:040x}",
        "deployment_tx_hash": "0x" + f"{index:064x}",
        "deployment_block": 12345 + index,
        "constructor_args": ["fixture"],
    }


def test_generated_manifest_env_validate_and_preflight_without_red(tmp_path: Path) -> None:
    manifest_module = _load_module(SCRIPT, "sota_base_testnet_manifest")
    validator = _load_module(VALIDATOR, "validate_base_sota_manifest_for_generated")
    preflight = _load_module(PREFLIGHT, "sota_base_testnet_preflight_for_generated")
    manifest_out = tmp_path / "base-sepolia-manifest.json"
    env_out = tmp_path / "base-sota.env"

    exit_code = manifest_module.main(
        [
            "--template",
            str(TEMPLATE),
            "--deployment",
            str(_deployment(tmp_path)),
            "--manifest-out",
            str(manifest_out),
            "--env-out",
            str(env_out),
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
            "7",
            "--readiness-url",
            "https://claims-test.example.invalid/base-sota-testnet-readiness.json",
        ]
    )

    assert exit_code == 0
    validated = validator.validate_manifest(manifest_out)
    assert validated["chain_id"] == 84532
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert manifest["contracts"]["sota_token"]["address"] == "0x0000000000000000000000000000000000000001"
    assert manifest["browser_safe"]["service_urls"]["indexer_api"] == "https://claims-api-test.example.invalid"
    assert (
        manifest["services"]["claims_ui"]["browser_safe_env"]["NEXT_PUBLIC_SOTA_TOKEN_ADDRESS"]
        == "0x0000000000000000000000000000000000000001"
    )
    env_text = env_out.read_text(encoding="utf-8")
    assert "PRIVATE_KEY" not in env_text
    assert "SOTA_TEST_OLD_COLDKEY=5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY" in env_text
    assert "SOTA_TEST_EPOCH=7" in env_text
    assert "NEXT_PUBLIC_SOTA_TOKEN_ADDRESS=0x0000000000000000000000000000000000000001" in env_text
    assert "NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL=https://coordinator-test.example.invalid" in env_text
    assert "NEXT_PUBLIC_SOTA_READINESS_URL=https://claims-test.example.invalid/base-sota-testnet-readiness.json" in env_text

    report = preflight.run_preflight(manifest_out, env_file=env_out, offline=True)
    assert report["summary"]["red"] == 0
    assert report["summary"]["yellow"] > 0


def test_manifest_adapter_rejects_non_base_sepolia_deployment(tmp_path: Path) -> None:
    manifest_module = _load_module(SCRIPT, "sota_base_testnet_manifest_reject_mainnet")

    with pytest.raises(SystemExit, match="84532"):
        manifest_module.main(
            [
                "--template",
                str(TEMPLATE),
                "--deployment",
                str(_deployment(tmp_path, chain_id=8453)),
                "--manifest-out",
                str(tmp_path / "manifest.json"),
                "--env-out",
                str(tmp_path / "env"),
            ]
        )


def test_manifest_adapter_rejects_zero_contract_address(tmp_path: Path) -> None:
    manifest_module = _load_module(SCRIPT, "sota_base_testnet_manifest_reject_zero")
    deployment_path = _deployment(tmp_path)
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    deployment["contracts"]["sota_token"]["address"] = "0x0000000000000000000000000000000000000000"
    deployment_path.write_text(json.dumps(deployment, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="nonzero EVM address"):
        manifest_module.main(
            [
                "--template",
                str(TEMPLATE),
                "--deployment",
                str(deployment_path),
                "--manifest-out",
                str(tmp_path / "manifest.json"),
                "--env-out",
                str(tmp_path / "env"),
            ]
        )

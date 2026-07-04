from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_preflight.py"
TEMPLATE = REPO / "docs" / "base" / "manifests" / "base-sepolia-deployment-manifest.template.json"


def _load_module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("sota_base_testnet_preflight", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _filled_manifest(tmp_path: Path) -> Path:
    manifest = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    manifest["status"] = "testnet-preflight-fixture"
    manifest["deployer"]["address"] = "0x1111111111111111111111111111111111111111"
    for source in manifest["source"].values():
        source["branch"] = "testnet"
        source["commit_sha"] = "a" * 40
    manifest["abi_bundle"]["artifact_sha256"] = "b" * 64
    manifest["abi_bundle"]["generated_from_contracts_commit_sha"] = "a" * 40
    for key, item in manifest["roles"].items():
        if isinstance(item, dict) and "address" in item:
            item["address"] = "0x1111111111111111111111111111111111111111"
    for index, contract in enumerate(manifest["contracts"].values(), start=1):
        contract["address"] = f"0x{index:040x}"
        contract["source_verification_url"] = f"https://sepolia.basescan.org/address/0x{index:040x}#code"
    for key, service in manifest["services"].items():
        service["source_branch"] = "testnet"
        service["commit_sha"] = "a" * 40
        if key == "claims_ui":
            service["public_url"] = "https://claims-test.example.invalid"
            service["health_url"] = "https://claims-test.example.invalid/health"
        elif key == "indexer_api":
            service["public_base_url"] = "https://claims-api-test.example.invalid"
            service["health_url"] = "https://claims-api-test.example.invalid/health"
        else:
            service["service_url"] = f"https://{key}.example.invalid"
            service["health_url"] = f"https://{key}.example.invalid/health"
    manifest["browser_safe"]["contract_addresses"] = {
        key: item["address"] for key, item in manifest["contracts"].items()
    }
    manifest["browser_safe"]["service_urls"]["claims_ui"] = "https://claims-test.example.invalid"
    manifest["browser_safe"]["service_urls"]["indexer_api"] = "https://claims-api-test.example.invalid"
    manifest["rollback"]["owner"] = "sre_devops_engineer"
    manifest["rollback"]["rollback_plan_url"] = "https://docs.example.invalid/rollback"
    return _write_json(tmp_path / "manifest.json", manifest)


def _env_file(tmp_path: Path, **overrides: str) -> Path:
    values = {
        "BASE_CHAIN_ID": "84532",
        "BASE_RPC_URL": "https://sepolia.base.org",
        "NEXT_PUBLIC_SOTA_ENVIRONMENT": "testnet",
        "NEXT_PUBLIC_SOTA_BASE_CHAIN_ID": "84532",
        "NEXT_PUBLIC_SOTA_BASE_CHAIN_NAME": "Base Sepolia",
        "NEXT_PUBLIC_SOTA_BASE_RPC_URL": "https://sepolia.base.org",
        "NEXT_PUBLIC_SOTA_BASE_EXPLORER_URL": "https://sepolia.basescan.org",
        "NEXT_PUBLIC_SOTA_CLAIMS_API_URL": "https://claims-api-test.example.invalid",
        "NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL": "https://coordinator-test.example.invalid",
        "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID": "base:sota-local",
        "NEXT_PUBLIC_SOTA_GENESIS_DISTRIBUTOR_ADDRESS": "0x0000000000000000000000000000000000000005",
        "NEXT_PUBLIC_SOTA_EMISSION_DISTRIBUTOR_ADDRESS": "0x0000000000000000000000000000000000000006",
        "NEXT_PUBLIC_SOTA_TOKEN_ADDRESS": "0x0000000000000000000000000000000000000001",
        "SOTA_TEST_WALLET_ADDRESS": "0x1111111111111111111111111111111111111111",
    }
    values.update(overrides)
    path = tmp_path / "base-sota.env"
    path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    return path


def test_template_preflight_reports_missing_deployed_contracts_and_services() -> None:
    module = _load_module()

    report = module.run_preflight(TEMPLATE, offline=True)

    assert report["ok"] is False
    assert report["summary"]["red"] > 0
    names = {check["name"]: check for check in report["checks"]}
    assert names["contract_sota_token"]["status"] == "red"
    assert names["service_claims_ui"]["status"] == "red"


def test_offline_filled_manifest_with_public_env_is_yellow_not_red(tmp_path: Path) -> None:
    module = _load_module()

    report = module.run_preflight(_filled_manifest(tmp_path), env_file=_env_file(tmp_path), offline=True)

    assert report["ok"] is False
    assert report["summary"]["red"] == 0
    assert report["summary"]["yellow"] > 0


def test_preflight_rejects_base_mainnet_chain_id(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = _filled_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chain"]["chain_id"] = 8453
    _write_json(manifest_path, manifest)

    report = module.run_preflight(manifest_path, env_file=_env_file(tmp_path, BASE_CHAIN_ID="8453"), offline=True)

    chain_check = next(check for check in report["checks"] if check["name"] == "chain_config")
    assert chain_check["status"] == "red"
    assert "mainnet" in chain_check["detail"].lower()


def test_preflight_rejects_zero_public_contract_env(tmp_path: Path) -> None:
    module = _load_module()

    report = module.run_preflight(
        _filled_manifest(tmp_path),
        env_file=_env_file(tmp_path, NEXT_PUBLIC_SOTA_TOKEN_ADDRESS="0x0000000000000000000000000000000000000000"),
        offline=True,
    )

    token_check = next(check for check in report["checks"] if check["name"] == "env_NEXT_PUBLIC_SOTA_TOKEN_ADDRESS")
    assert token_check["status"] == "red"


def test_preflight_rejects_zero_test_wallet(tmp_path: Path) -> None:
    module = _load_module()

    report = module.run_preflight(
        _filled_manifest(tmp_path),
        env_file=_env_file(tmp_path, SOTA_TEST_WALLET_ADDRESS="0x0000000000000000000000000000000000000000"),
        offline=True,
    )

    wallet_check = next(check for check in report["checks"] if check["name"] == "test_wallet")
    assert wallet_check["status"] == "red"

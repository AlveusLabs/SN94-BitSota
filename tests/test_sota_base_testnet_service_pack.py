from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_service_pack.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_service_pack", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    values = {
        "manifest": tmp_path / "base-sepolia-deployment-manifest.json",
        "env_file": tmp_path / "base-sota.env.testnet",
        "claims_ui": "",
        "claims_api": "",
        "coordinator": "",
        "attestation": "",
        "root_publisher": "",
        "claim_artifacts": "",
        "monitoring": "",
        "readiness_url": "https://claims-test.bitsota.com/base-sota-testnet-readiness.json",
        "json_out": tmp_path / "base-sota-testnet-service-pack.json",
        "markdown_out": tmp_path / "base-sota-testnet-service-pack.md",
        "html_out": tmp_path / "base-sota-testnet-service-pack.html",
        "apprunner_out_dir": tmp_path / "apprunner",
        "json": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _manifest(tmp_path: Path, *, chain_id: int = 84532) -> Path:
    payload = {
        "manifest_schema_version": "sota-base-sepolia-deployment-manifest/v1",
        "environment": "base-sepolia" if chain_id == 84532 else "base",
        "chain": {
            "chain_id": chain_id,
            "chain_name": "base-sepolia" if chain_id == 84532 else "base",
            "network_display_name": "Base Sepolia" if chain_id == 84532 else "Base",
            "public_browser_rpc_url": "https://sepolia.base.org",
        },
        "browser_safe": {
            "contract_addresses": {
                "sota_token": "0x0000000000000000000000000000000000000001",
                "vault": "0x0000000000000000000000000000000000000002",
                "root_registry": "0x0000000000000000000000000000000000000003",
                "lane_registry": "0x0000000000000000000000000000000000000004",
                "genesis_distributor": "0x0000000000000000000000000000000000000005",
                "emission_distributor": "0x0000000000000000000000000000000000000006",
            },
            "service_urls": {
                "claims_ui": "https://claims-test.example.invalid",
                "indexer_api": "https://claims-api-test.example.invalid",
            },
        },
        "services": {
            "claims_ui": {"public_url": "https://claims-test.example.invalid"},
            "indexer_api": {"public_base_url": "https://claims-api-test.example.invalid"},
            "root_publisher": {"service_url": "https://root-test.example.invalid"},
            "attestation_builder": {"service_url": "https://attestation-test.example.invalid"},
            "monitoring": {"dashboard_url": "https://monitoring-test.example.invalid"},
        },
        "secret_handles": {
            "base_sepolia_rpc": "secret://base-sepolia/rpc",
            "base_sepolia_root_publisher_signer": "secret://base-sepolia/root-publisher",
            "base_sepolia_indexer_database_url": "secret://base-sepolia/indexer-db",
            "base_sepolia_indexer_admin_token": "secret://base-sepolia/indexer-admin",
            "base_sepolia_monitoring_api_key": "secret://base-sepolia/monitoring",
        },
    }
    path = tmp_path / "base-sepolia-deployment-manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_default_service_pack_is_generated_without_deployed_manifest(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)

    pack = module.build_service_pack(args)

    assert pack["schema"] == "sota-base-testnet-service-pack/v1"
    assert pack["ok"] is True
    assert pack["status"] == "yellow"
    assert pack["deployment_ready"] is False
    assert pack["summary"]["red"] == 0
    assert len(pack["services"]) == 8
    assert pack["urls"]["claims_ui"] == "https://claims-test.bitsota.com"
    assert "broadcast_transactions" in pack["does_not"]
    assert "touch_base_mainnet" in pack["does_not"]
    assert pack["aws_deploy_plan"]["public_app_runner_services"] == [
        "base-sota-claims-ui-test",
        "base-sota-indexer-api-test",
        "base-sota-autoresearch-coordinator-test",
    ]
    checks = {check["name"]: check for check in pack["checks"]}
    assert checks["chain_id"]["status"] == "green"
    assert checks["manifest_present"]["status"] == "yellow"
    assert checks["env_file_present"]["status"] == "yellow"
    assert checks["no_raw_secrets"]["status"] == "green"
    assert checks["worker_wrappers"]["status"] == "green"
    root_publisher = next(service for service in pack["services"] if service["key"] == "root_publisher")
    assert root_publisher["implementation_status"] == "ready_to_configure"
    assert "sota_base_publish_root.py" in root_publisher["run_command"]
    assert root_publisher["deployment_recipe"]["target"] == "operator_controlled_worker"
    claims_ui = next(service for service in pack["services"] if service["key"] == "claims_ui")
    assert claims_ui["deployment_recipe"]["target"] == "aws_apprunner_public_service"
    assert claims_ui["deployment_recipe"]["create_service_input_file"] == "apprunner/base-sota-claims-ui-test.json"
    assert claims_ui["env_public_values"]["NEXT_PUBLIC_SOTA_CLAIMS_CONTRACT_ADDRESS"] == claims_ui["env_public_values"]["NEXT_PUBLIC_SOTA_GENESIS_DISTRIBUTOR_ADDRESS"]
    claim_artifacts = next(service for service in pack["services"] if service["key"] == "claim_artifacts")
    assert "sota_base_testnet_seed_artifacts.py build" in claim_artifacts["run_command"]


def test_service_pack_uses_manifest_urls_and_secret_handles(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _manifest(tmp_path)
    (tmp_path / "base-sota.env.testnet").write_text("NEXT_PUBLIC_SOTA_BASE_CHAIN_ID=84532\n", encoding="utf-8")
    args = _args(tmp_path, manifest=manifest)

    pack = module.build_service_pack(args)
    services = {service["key"]: service for service in pack["services"]}

    assert pack["contracts"]["root_registry"] == "0x0000000000000000000000000000000000000003"
    assert services["claims_ui"]["public_url"] == "https://claims-test.example.invalid"
    assert services["indexer_api"]["public_url"] == "https://claims-api-test.example.invalid"
    assert services["root_publisher"]["public_url"] == "https://root-test.example.invalid"
    assert services["attestation_builder"]["public_url"] == "https://attestation-test.example.invalid"
    assert services["claim_artifacts"]["public_url"] == "https://claims-test.bitsota.com/base-sota-testnet-seed-artifacts-finalized.json"
    assert services["claims_ui"]["env_file"] == str(tmp_path / "base-sota.env.testnet")
    assert "secret://base-sepolia/indexer-admin" in services["indexer_api"]["env_secret_handles"]
    assert services["indexer_api"]["env_secret_map"]["SOTA_BASE_INDEXER_ADMIN_TOKEN"] == "secret://base-sepolia/indexer-admin"
    assert services["indexer_api"]["env_public_values"]["SOTA_BASE_CLAIM_ARTIFACT_REQUIRED"] == "true"
    assert "base-sota-testnet-genesis-claim-artifact.json" in services["indexer_api"]["env_public_values"]["SOTA_BASE_CLAIM_ARTIFACT_URLS"]
    assert not module._secret_findings(pack)


def test_service_pack_ignores_manifest_todo_secret_handle_overrides(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["secret_handles"]["base_sepolia_indexer_admin_token"] = "TODO:secret-handle:base-sepolia-indexer-admin-token"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args = _args(tmp_path, manifest=manifest)

    pack = module.build_service_pack(args)
    services = {service["key"]: service for service in pack["services"]}

    assert services["indexer_api"]["env_secret_map"]["SOTA_BASE_INDEXER_ADMIN_TOKEN"] == "base-sota/test/base-sepolia/indexer-admin-token"


def test_service_pack_rejects_base_mainnet_manifest(tmp_path: Path) -> None:
    module = _load_module()
    manifest = _manifest(tmp_path, chain_id=8453)
    args = _args(tmp_path, manifest=manifest)

    pack = module.build_service_pack(args)
    chain_check = next(check for check in pack["checks"] if check["name"] == "chain_id")

    assert pack["ok"] is False
    assert pack["status"] == "red"
    assert chain_check["status"] == "red"
    assert "mainnet" in chain_check["detail"].lower()


def test_service_pack_main_writes_json_markdown_and_html(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)

    exit_code = module.main(
        [
            "--manifest",
            str(args.manifest),
            "--env-file",
            str(args.env_file),
            "--json-out",
            str(args.json_out),
            "--markdown-out",
            str(args.markdown_out),
            "--html-out",
            str(args.html_out),
            "--apprunner-out-dir",
            str(args.apprunner_out_dir),
        ]
    )

    assert exit_code == 0
    assert args.json_out.exists()
    assert args.markdown_out.exists()
    assert args.html_out.exists()
    assert (args.apprunner_out_dir / "base-sota-claims-ui-test.json").exists()
    claims_input = json.loads((args.apprunner_out_dir / "base-sota-claims-ui-test.json").read_text(encoding="utf-8"))
    claims_values = claims_input["SourceConfiguration"]["CodeRepository"]["CodeConfiguration"]["CodeConfigurationValues"]
    assert claims_values["Runtime"] == "NODEJS_22"
    assert " && env " in claims_values["BuildCommand"]
    assert "NEXT_PUBLIC_SOTA_BASE_CHAIN_ID=84532" in claims_values["BuildCommand"]
    assert "NEXT_PUBLIC_SOTA_BASE_CHAIN_NAME" not in claims_values["BuildCommand"]
    assert "NEXT_PUBLIC_SOTA_CLAIMS_CONTRACT_ADDRESS=" in claims_values["BuildCommand"]
    assert "corepack pnpm build" in claims_values["BuildCommand"]
    indexer_input = json.loads((args.apprunner_out_dir / "base-sota-indexer-api-test.json").read_text(encoding="utf-8"))
    code_config = indexer_input["SourceConfiguration"]["CodeRepository"]["CodeConfiguration"]
    values = code_config["CodeConfigurationValues"]
    assert code_config["ConfigurationSource"] == "API"
    assert values["Runtime"] == "PYTHON_311"
    assert "pip install -e ." not in values["BuildCommand"]
    assert "fastapi==0.115.8" in values["BuildCommand"]
    assert "sqlalchemy==2.0.38" in values["BuildCommand"]
    assert values["RuntimeEnvironmentSecrets"]["SOTA_BASE_INDEXER_ADMIN_TOKEN"] == "base-sota/test/base-sepolia/indexer-admin-token"
    assert indexer_input["InstanceConfiguration"]["InstanceRoleArn"] == "${SOTA_APPRUNNER_INSTANCE_ROLE_ARN}"
    assert "base-sota-testnet-emission-claim-artifact.json" in values["RuntimeEnvironmentVariables"]["SOTA_BASE_CLAIM_ARTIFACT_URLS"]
    assert "Root Publisher Worker" in args.markdown_out.read_text(encoding="utf-8")
    assert "aws_apprunner_public_service" in args.markdown_out.read_text(encoding="utf-8")
    assert "sota_base_publish_root.py" in args.html_out.read_text(encoding="utf-8")

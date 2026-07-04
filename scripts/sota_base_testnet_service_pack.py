#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any
from urllib.parse import urljoin, urlparse


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_MANIFEST = DEFAULT_ARTIFACTS_DIR / "base-sepolia-deployment-manifest.json"
DEFAULT_ENV_FILE = DEFAULT_ARTIFACTS_DIR / "base-sota.env.testnet"
DEFAULT_APPRUNNER_OUT_DIR = DEFAULT_ARTIFACTS_DIR / "apprunner"
DEFAULT_AWS_REGION = "eu-central-1"
DEFAULT_AWS_PROFILE = "moonrocklab-frankfurt"
DEFAULT_APPRUNNER_CONNECTION_NAME = "bitsota"
APPRUNNER_CONNECTION_ARN_ENV = "SOTA_APPRUNNER_CONNECTION_ARN"
APPRUNNER_INSTANCE_ROLE_ARN_ENV = "SOTA_APPRUNNER_INSTANCE_ROLE_ARN"
DEFAULT_SECRET_HANDLES = {
    "base_sepolia_rpc": "base-sota/test/base-sepolia/rpc-url",
    "base_sepolia_deployer_private_key": "base-sota/test/base-sepolia/deployer",
    "base_sepolia_root_publisher_signer": "base-sota/test/base-sepolia/root-publisher",
    "base_sepolia_indexer_admin_token": "base-sota/test/base-sepolia/indexer-admin-token",
    "base_sepolia_autoresearch_database_url": "base-sota/test/base-sepolia/autoresearch-database-url",
    "base_sepolia_autoresearch_admin_token": "base-sota/test/base-sepolia/autoresearch-admin-token",
    "base_sepolia_monitoring_api_key": "base-sota/test/base-sepolia/monitoring-api-key",
}
BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
INDEXER_API_BUILD_COMMAND = (
    'python3 -m pip install '
    '"fastapi==0.115.8" '
    '"uvicorn[standard]==0.34.0" '
    '"web3==7.13.0" '
    '"eth-abi==5.2.0" '
    '"eth-utils==5.3.1" '
    '"sqlalchemy==2.0.38"'
)
DEFAULT_URLS = {
    "claims_ui": "https://claims-test.bitsota.com",
    "claims_api": "https://claims-api-test.bitsota.com",
    "coordinator": "https://coordinator-test.bitsota.com",
    "attestation": "https://attestation-test.bitsota.com",
    "root_publisher": "https://root-publisher-test.bitsota.com",
    "claim_artifacts": "https://claims-test.bitsota.com/base-sota-testnet-seed-artifacts-finalized.json",
    "monitoring": "https://monitoring-test.bitsota.com",
    "readiness": "https://claims-test.bitsota.com/base-sota-testnet-readiness.json",
}
SECRET_NAME_RE = re.compile(
    r"(private[_-]?key|mnemonic|seed|secret|password|database_url|admin[_-]?token|auth[_-]?token|api[_-]?key)",
    re.IGNORECASE,
)
RAW_PRIVATE_KEY_RE = re.compile(r"0x[a-fA-F0-9]{64}")
RAW_SECRET_URL_RE = re.compile(r"(?i)(postgres|mysql|redis)://[^\\s]+:[^\\s@]+@")
APPROVED_SECRET_HANDLE_PREFIXES = ("TODO:secret-handle:", "secret://", "$", "base-sota/test/base-sepolia/")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _url(base: str, path: str = "") -> str:
    if not path:
        return base.rstrip("/")
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _host(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return parsed.hostname or value


def _repo_info(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {"path": str(path), "branch": None, "commit_sha": None, "remote_url": None}
    branch = None
    commit = None
    remote_url = None
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip() or None
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip() or None
        remote_name = "origin"
        if branch:
            remote_name = (
                subprocess.run(
                    ["git", "config", "--get", f"branch.{branch}.remote"],
                    cwd=path,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip()
                or remote_name
            )
        remote_url = subprocess.check_output(
            ["git", "config", "--get", f"remote.{remote_name}.url"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip() or None
    except Exception:
        pass
    return {"path": str(path), "branch": branch, "commit_sha": commit, "remote_url": remote_url}


def _manifest_service_url(manifest: dict[str, Any] | None, service_key: str) -> str | None:
    service = dict(dict(manifest or {}).get("services", {}).get(service_key, {}) or {})
    for key in ("public_url", "public_base_url", "service_url", "health_url", "dashboard_url"):
        value = str(service.get(key) or "").strip()
        if value:
            return value.rstrip("/")
    browser_safe = dict(dict(manifest or {}).get("browser_safe", {}).get("service_urls", {}) or {})
    value = str(browser_safe.get(service_key) or "").strip()
    return value.rstrip("/") if value else None


def _arg_or_manifest_url(args: argparse.Namespace, attr: str, manifest: dict[str, Any] | None, service_key: str) -> str:
    value = str(getattr(args, attr) or "").strip()
    if value:
        return value.rstrip("/")
    return _manifest_service_url(manifest, service_key) or DEFAULT_URLS[attr]


def _secret_handles(manifest: dict[str, Any] | None) -> dict[str, str]:
    handles = dict(dict(manifest or {}).get("secret_handles") or {})
    normalized: dict[str, str] = {}
    for key, value in handles.items():
        key_s = str(key)
        value_s = str(value).strip()
        if not value_s:
            continue
        if value_s.startswith("TODO:secret-handle:") and key_s in DEFAULT_SECRET_HANDLES:
            continue
        normalized[key_s] = value_s
    return {**DEFAULT_SECRET_HANDLES, **normalized}


def _contract_addresses(manifest: dict[str, Any] | None) -> dict[str, str | None]:
    browser_safe = dict(dict(manifest or {}).get("browser_safe", {}).get("contract_addresses", {}) or {})
    contracts = dict(dict(manifest or {}).get("contracts", {}) or {})
    keys = (
        "sota_token",
        "vault",
        "root_registry",
        "lane_registry",
        "genesis_distributor",
        "emission_distributor",
    )
    return {
        key: (
            str(browser_safe.get(key) or "").strip()
            or str(dict(contracts.get(key) or {}).get("address") or "").strip()
            or None
        )
        for key in keys
    }


def _chain_check(manifest: dict[str, Any] | None) -> dict[str, str]:
    chain = dict(dict(manifest or {}).get("chain") or {})
    raw_chain_id = chain.get("chain_id") if chain else BASE_SEPOLIA_CHAIN_ID
    try:
        chain_id = int(raw_chain_id)
    except Exception:
        return {
            "name": "chain_id",
            "status": "red",
            "detail": f"Manifest chain id is invalid: {raw_chain_id!r}.",
            "remediation": "Regenerate the manifest for Base Sepolia chain id 84532.",
        }
    if chain_id == BASE_MAINNET_CHAIN_ID:
        return {
            "name": "chain_id",
            "status": "red",
            "detail": "Manifest points at Base mainnet chain id 8453.",
            "remediation": "Never deploy the testnet service pack against Base mainnet.",
        }
    if chain_id != BASE_SEPOLIA_CHAIN_ID:
        return {
            "name": "chain_id",
            "status": "red",
            "detail": f"Manifest chain id is {chain_id}, expected Base Sepolia 84532.",
            "remediation": "Use only Base Sepolia for this service pack.",
        }
    return {"name": "chain_id", "status": "green", "detail": "Service pack is pinned to Base Sepolia chain id 84532."}


def _claim_artifact_urls(finalized_report_url: str) -> str:
    base = finalized_report_url.rsplit("/", 1)[0].rstrip("/")
    if not base:
        return ""
    return ",".join(
        [
            f"{base}/base-sota-testnet-genesis-claim-artifact.json",
            f"{base}/base-sota-testnet-emission-claim-artifact.json",
        ]
    )


def _check(name: str, ok: bool, detail: str, remediation: str = "") -> dict[str, str]:
    out = {"name": name, "status": "green" if ok else "red", "detail": detail}
    if remediation:
        out["remediation"] = remediation
    return out


def _yellow(name: str, detail: str, remediation: str = "") -> dict[str, str]:
    out = {"name": name, "status": "yellow", "detail": detail}
    if remediation:
        out["remediation"] = remediation
    return out


def _summary(checks: list[dict[str, str]]) -> dict[str, int]:
    return {
        "green": sum(1 for check in checks if check["status"] == "green"),
        "yellow": sum(1 for check in checks if check["status"] == "yellow"),
        "red": sum(1 for check in checks if check["status"] == "red"),
    }


def _worst(checks: list[dict[str, str]]) -> str:
    rank = {"green": 0, "yellow": 1, "red": 2}
    return max((check["status"] for check in checks), key=lambda status: rank.get(status, 2), default="green")


def _service_name(key: str) -> str:
    return f"base-sota-{key.replace('_', '-')}-test"


def _deployment_recipe(service: dict[str, Any]) -> dict[str, Any]:
    source = dict(service.get("source") or {})
    service_name = _service_name(str(service["key"]))
    port = service.get("port")
    if port:
        return {
            "target": "aws_apprunner_public_service",
            "service_name": service_name,
            "region": DEFAULT_AWS_REGION,
            "aws_profile": DEFAULT_AWS_PROFILE,
            "connection_name": DEFAULT_APPRUNNER_CONNECTION_NAME,
            "connection_arn_env": APPRUNNER_CONNECTION_ARN_ENV,
            "source_repository": source.get("remote_url"),
            "source_branch": source.get("branch") or "testing",
            "source_directory": "/",
            "health_check_path": service.get("health_path") or "/",
            "port": port,
            "create_service_input_file": f"apprunner/{service_name}.json",
            "create_service_command": (
                "aws apprunner create-service "
                f"--cli-input-json file://apprunner/{service_name}.json "
                f"--profile {DEFAULT_AWS_PROFILE} --region {DEFAULT_AWS_REGION}"
            ),
            "required_before_create": [
                "Push the source branch that contains the Base SOTA fork code.",
                "Set SOTA_APPRUNNER_CONNECTION_ARN to an approved App Runner GitHub connection ARN.",
                "Put public env values and secret-handle lookups in the repository App Runner config.",
                "Do not reuse production services for Base Sepolia testing.",
            ],
        }
    if service.get("type") == "static_readiness":
        return {
            "target": "public_static_artifact",
            "service_name": service_name,
            "region": DEFAULT_AWS_REGION,
            "publish_command": str(service.get("run_command") or ""),
            "required_before_publish": [
                "Generate a green Base Sepolia readiness artifact.",
                "Publish the artifact at the configured NEXT_PUBLIC_SOTA_READINESS_URL.",
            ],
        }
    return {
        "target": "operator_controlled_worker",
        "service_name": service_name,
        "region": DEFAULT_AWS_REGION,
        "run_command": str(service.get("run_command") or ""),
        "required_before_run": [
            "Run only from the testnet operator environment.",
            "Load signer material from approved secret handles at runtime.",
            "Record JSON outputs under the Base Sepolia artifacts directory.",
        ],
    }


def _secret_findings(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f"{path}.{key}"
            if isinstance(item, str):
                if RAW_PRIVATE_KEY_RE.search(item):
                    findings.append(f"{next_path} contains a raw 32-byte hex private-key-shaped value")
                if RAW_SECRET_URL_RE.search(item):
                    findings.append(f"{next_path} contains a raw secret-bearing URL")
                if SECRET_NAME_RE.search(str(key)) and item and not item.startswith(APPROVED_SECRET_HANDLE_PREFIXES):
                    findings.append(f"{next_path} should be a secret handle or environment reference, not a raw value")
            else:
                findings.extend(_secret_findings(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_secret_findings(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        if RAW_PRIVATE_KEY_RE.search(value):
            findings.append(f"{path} contains a raw 32-byte hex private-key-shaped value")
        if RAW_SECRET_URL_RE.search(value):
            findings.append(f"{path} contains a raw secret-bearing URL")
    return findings


def _service(
    *,
    key: str,
    title: str,
    service_type: str,
    owner: str,
    repo: str,
    cwd: Path,
    url: str,
    port: int | None,
    health_path: str | None,
    build_commands: list[str],
    run_command: str,
    env_file: Path,
    env_public_keys: list[str],
    env_public_values: dict[str, str] | None = None,
    env_secret_handles: list[str],
    env_secret_map: dict[str, str] | None = None,
    depends_on: list[str],
    notes: list[str] | None = None,
    implementation_status: str = "ready_to_configure",
) -> dict[str, Any]:
    health_url = _url(url, health_path) if health_path else None
    source = _repo_info(cwd)
    payload: dict[str, Any] = {
        "key": key,
        "title": title,
        "type": service_type,
        "owner": owner,
        "repo": repo,
        "cwd": str(cwd),
        "source": source,
        "public_url": url,
        "dns_host": _host(url),
        "port": port,
        "health_path": health_path,
        "health_url": health_url,
        "build_commands": build_commands,
        "run_command": run_command,
        "env_file": str(env_file),
        "env_public_keys": env_public_keys,
        "env_public_values": env_public_values or {},
        "env_secret_handles": env_secret_handles,
        "env_secret_map": env_secret_map or {},
        "depends_on": depends_on,
        "implementation_status": implementation_status,
        "notes": notes or [],
    }
    payload["deployment_recipe"] = _deployment_recipe(payload)
    return payload


def _aws_deploy_plan(services: list[dict[str, Any]], handles: dict[str, str]) -> dict[str, Any]:
    public_services = [
        service
        for service in services
        if dict(service.get("deployment_recipe") or {}).get("target") == "aws_apprunner_public_service"
    ]
    worker_jobs = [
        service
        for service in services
        if dict(service.get("deployment_recipe") or {}).get("target") == "operator_controlled_worker"
    ]
    static_artifacts = [
        service
        for service in services
        if dict(service.get("deployment_recipe") or {}).get("target") == "public_static_artifact"
    ]
    service_secret_handles = sorted(
        {
            str(handle)
            for service in services
            for handle in service.get("env_secret_handles") or []
        }
    )
    if handles.get("base_sepolia_deployer_private_key"):
        service_secret_handles.append(handles["base_sepolia_deployer_private_key"])
    return {
        "region": DEFAULT_AWS_REGION,
        "aws_profile": DEFAULT_AWS_PROFILE,
        "app_runner_connection_name": DEFAULT_APPRUNNER_CONNECTION_NAME,
        "app_runner_connection_arn_env": APPRUNNER_CONNECTION_ARN_ENV,
        "public_app_runner_services": [
            dict(service["deployment_recipe"])["service_name"] for service in public_services
        ],
        "operator_controlled_workers": [
            dict(service["deployment_recipe"])["service_name"] for service in worker_jobs
        ],
        "public_static_artifacts": [
            dict(service["deployment_recipe"])["service_name"] for service in static_artifacts
        ],
        "required_secret_handles": sorted(set(service_secret_handles)),
        "required_commands": [
            "python3 scripts/sota_base_testnet_operator.py --deploy --private-key-secret-id <base-sota/test/deployer>",
            "python3 scripts/sota_base_testnet_operator.py --deployment <base-sepolia-compact-deployment.json> --emission-evidence <accepted-emission-evidence.json> --broadcast-roots --root-publisher-private-key-secret-id <base-sota/test/root-publisher> --import-artifacts",
            "python3 scripts/sota_base_release_status.py --testnet-artifacts-dir /home/mekaneeky/repos/.sota-base-testnet",
        ],
        "safe_boundary": [
            "Base Sepolia only.",
            "No Base mainnet deployment.",
            "No production Bittensor or production TAO mutation.",
            "No raw private keys in manifests, docs, or reports.",
        ],
    }


def _next_public_env_build_command(env_public_values: dict[str, Any]) -> str:
    entries = {
        key: str(value)
        for key, value in sorted(env_public_values.items())
        if key.startswith("NEXT_PUBLIC_") and not re.search(r"\s", str(value))
    }
    if not entries:
        return ""
    values = " ".join(f"{key}={shlex.quote(value)}" for key, value in entries.items())
    return f"env {values}"


def _apprunner_input(service: dict[str, Any]) -> dict[str, Any] | None:
    recipe = dict(service.get("deployment_recipe") or {})
    if recipe.get("target") != "aws_apprunner_public_service":
        return None
    repository_url = str(recipe.get("source_repository") or "").strip()
    if not repository_url:
        repository_url = "<GITHUB_REPOSITORY_URL>"
    runtime = "NODEJS_22" if service.get("key") == "claims_ui" else "PYTHON_311"
    build_commands = [str(command) for command in service.get("build_commands") or []]
    if service.get("key") == "claims_ui":
        env_build_command = _next_public_env_build_command(dict(service.get("env_public_values") or {}))
        if env_build_command:
            build_commands = [
                f"{env_build_command} {command}" if "pnpm build" in command else command
                for command in build_commands
            ]
    build_command = " && ".join(build_commands)
    code_configuration_values = {
        "Runtime": runtime,
        "BuildCommand": build_command,
        "StartCommand": str(service.get("run_command") or ""),
        "Port": str(recipe.get("port") or service.get("port") or ""),
        "RuntimeEnvironmentVariables": dict(service.get("env_public_values") or {}),
        "RuntimeEnvironmentSecrets": dict(service.get("env_secret_map") or {}),
    }
    payload = {
        "ServiceName": recipe["service_name"],
        "SourceConfiguration": {
            "CodeRepository": {
                "RepositoryUrl": repository_url,
                "SourceCodeVersion": {
                    "Type": "BRANCH",
                    "Value": str(recipe.get("source_branch") or "testing"),
                },
                "CodeConfiguration": {
                    "ConfigurationSource": "API",
                    "CodeConfigurationValues": code_configuration_values,
                },
                "SourceDirectory": str(recipe.get("source_directory") or "/"),
            },
            "AutoDeploymentsEnabled": True,
            "AuthenticationConfiguration": {
                "ConnectionArn": f"${{{APPRUNNER_CONNECTION_ARN_ENV}}}",
            },
        },
        "InstanceConfiguration": {
            "Cpu": "1024",
            "Memory": "2048",
        },
        "HealthCheckConfiguration": {
            "Protocol": "HTTP",
            "Path": str(recipe.get("health_check_path") or "/"),
            "Interval": 10,
            "Timeout": 5,
            "HealthyThreshold": 1,
            "UnhealthyThreshold": 5,
        },
        "Tags": [
            {"Key": "project", "Value": "base-sota"},
            {"Key": "environment", "Value": "base-sepolia"},
            {"Key": "component", "Value": str(service.get("key") or "")},
        ],
    }
    if code_configuration_values["RuntimeEnvironmentSecrets"]:
        payload["InstanceConfiguration"]["InstanceRoleArn"] = f"${{{APPRUNNER_INSTANCE_ROLE_ARN_ENV}}}"
    return payload


def write_apprunner_inputs(pack: dict[str, Any], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for service in pack.get("services") or []:
        payload = _apprunner_input(dict(service))
        if payload is None:
            continue
        path = out_dir / f"{payload['ServiceName']}.json"
        _write_json(path, payload)
        written.append(str(path))
    return written


def build_service_pack(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_json(args.manifest) if args.manifest.exists() else None
    handles = _secret_handles(manifest)
    urls = {
        "claims_ui": _arg_or_manifest_url(args, "claims_ui", manifest, "claims_ui"),
        "claims_api": _arg_or_manifest_url(args, "claims_api", manifest, "indexer_api"),
        "coordinator": _arg_or_manifest_url(args, "coordinator", manifest, "coordinator"),
        "attestation": _arg_or_manifest_url(args, "attestation", manifest, "attestation_builder"),
        "root_publisher": _arg_or_manifest_url(args, "root_publisher", manifest, "root_publisher"),
        "claim_artifacts": _arg_or_manifest_url(args, "claim_artifacts", manifest, "claim_artifacts"),
        "monitoring": _arg_or_manifest_url(args, "monitoring", manifest, "monitoring"),
        "readiness": str(args.readiness_url or DEFAULT_URLS["readiness"]).strip(),
    }
    contracts = _contract_addresses(manifest)
    contract = lambda key: str(contracts.get(key) or "")
    docs_repo = REPOS / "SN94-BitSota-live-docs"
    website_repo = REPOS / "bitsota_website"
    community_repo = REPOS / "94-agent-community"
    autoresearch_repo = REPOS / "autoresearch-bittensor"
    pool_repo = REPOS / "Pool"

    services = [
        _service(
            key="claims_ui",
            title="Base Sepolia Claims UI",
            service_type="public_web",
            owner="full_stack_product_engineer",
            repo="bitsota_website",
            cwd=website_repo,
            url=urls["claims_ui"],
            port=3000,
            health_path="/claims",
            build_commands=["corepack pnpm install --frozen-lockfile", "corepack pnpm build"],
            run_command="corepack pnpm exec next start -H 0.0.0.0 -p 3000",
            env_file=args.env_file,
            env_public_keys=[
                "NEXT_PUBLIC_SOTA_ENVIRONMENT",
                "NEXT_PUBLIC_SOTA_BASE_CHAIN_ID",
                "NEXT_PUBLIC_SOTA_BASE_CHAIN_NAME",
                "NEXT_PUBLIC_SOTA_BASE_RPC_URL",
                "NEXT_PUBLIC_SOTA_BASE_EXPLORER_URL",
                "NEXT_PUBLIC_SOTA_CLAIMS_API_URL",
                "NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL",
                "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID",
                "NEXT_PUBLIC_SOTA_CLAIMS_CONTRACT_ADDRESS",
                "NEXT_PUBLIC_SOTA_GENESIS_DISTRIBUTOR_ADDRESS",
                "NEXT_PUBLIC_SOTA_EMISSION_DISTRIBUTOR_ADDRESS",
                "NEXT_PUBLIC_SOTA_TOKEN_ADDRESS",
                "NEXT_PUBLIC_SOTA_READINESS_URL",
                "NEXT_PUBLIC_SOTA_DEMO_ENABLED",
            ],
            env_public_values={
                "NEXT_PUBLIC_SOTA_ENVIRONMENT": "testnet",
                "NEXT_PUBLIC_SOTA_BASE_CHAIN_ID": str(BASE_SEPOLIA_CHAIN_ID),
                "NEXT_PUBLIC_SOTA_BASE_CHAIN_NAME": "Base Sepolia",
                "NEXT_PUBLIC_SOTA_BASE_RPC_URL": "https://sepolia.base.org",
                "NEXT_PUBLIC_SOTA_BASE_EXPLORER_URL": "https://sepolia.basescan.org",
                "NEXT_PUBLIC_SOTA_CLAIMS_API_URL": urls["claims_api"],
                "NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL": urls["coordinator"],
                "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID": "base:sota-local",
                "NEXT_PUBLIC_SOTA_CLAIMS_CONTRACT_ADDRESS": contract("genesis_distributor"),
                "NEXT_PUBLIC_SOTA_GENESIS_DISTRIBUTOR_ADDRESS": contract("genesis_distributor"),
                "NEXT_PUBLIC_SOTA_EMISSION_DISTRIBUTOR_ADDRESS": contract("emission_distributor"),
                "NEXT_PUBLIC_SOTA_TOKEN_ADDRESS": contract("sota_token"),
                "NEXT_PUBLIC_SOTA_READINESS_URL": urls["readiness"],
                "NEXT_PUBLIC_SOTA_DEMO_ENABLED": "false",
            },
            env_secret_handles=[],
            depends_on=["indexer_api", "autoresearch_coordinator", "static_readiness", "base_sepolia_contracts"],
        ),
        _service(
            key="indexer_api",
            title="Claims Indexer/API",
            service_type="public_api",
            owner="sre_devops_engineer",
            repo="94-agent-community",
            cwd=community_repo,
            url=urls["claims_api"],
            port=8010,
            health_path="/health",
            build_commands=[INDEXER_API_BUILD_COMMAND],
            run_command=(
                "python3 -m uvicorn experiments.base_protocol_design.sota_base_indexer.api:create_app "
                "--factory --host 0.0.0.0 --port 8010"
            ),
            env_file=args.env_file,
            env_public_keys=[
                "SOTA_BASE_INDEXER_DB",
                "SOTA_BASE_CLAIM_ARTIFACT_URLS",
                "SOTA_BASE_CLAIM_ARTIFACT_REQUIRED",
                "SOTA_BASE_CHAIN_ID",
                "SOTA_BASE_SYNC_FROM_BLOCK",
                "SOTA_BASE_CONTRACTS_ABI_DIR",
                "SOTA_TOKEN_ADDRESS",
                "SOTA_VAULT_ADDRESS",
                "SOTA_ROOT_REGISTRY_ADDRESS",
                "SOTA_LANE_REGISTRY_ADDRESS",
                "SOTA_GENESIS_DISTRIBUTOR_ADDRESS",
                "SOTA_EMISSION_DISTRIBUTOR_ADDRESS",
            ],
            env_public_values={
                "SOTA_BASE_INDEXER_DB": "/tmp/sota-base-indexer.sqlite3",
                "SOTA_BASE_CLAIM_ARTIFACT_URLS": _claim_artifact_urls(urls["claim_artifacts"]),
                "SOTA_BASE_CLAIM_ARTIFACT_REQUIRED": "true",
                "SOTA_BASE_CHAIN_ID": str(BASE_SEPOLIA_CHAIN_ID),
                "SOTA_BASE_RPC_URL": "https://sepolia.base.org",
                "SOTA_BASE_SYNC_FROM_BLOCK": "0",
                "SOTA_BASE_CONTRACTS_ABI_DIR": "/home/mekaneeky/repos/Pool/contracts/sota-base/abi",
                "SOTA_TOKEN_ADDRESS": contract("sota_token"),
                "SOTA_VAULT_ADDRESS": contract("vault"),
                "SOTA_ROOT_REGISTRY_ADDRESS": contract("root_registry"),
                "SOTA_LANE_REGISTRY_ADDRESS": contract("lane_registry"),
                "SOTA_GENESIS_DISTRIBUTOR_ADDRESS": contract("genesis_distributor"),
                "SOTA_EMISSION_DISTRIBUTOR_ADDRESS": contract("emission_distributor"),
            },
            env_secret_handles=[
                handles["base_sepolia_indexer_admin_token"],
            ],
            env_secret_map={
                "SOTA_BASE_INDEXER_ADMIN_TOKEN": handles["base_sepolia_indexer_admin_token"],
            },
            depends_on=["base_sepolia_contracts", "claim_artifacts", "static_readiness"],
            notes=[
                "The indexer uses SQLite for the read model and rehydrates from SOTA_BASE_CLAIM_ARTIFACT_URLS on startup.",
                "SOTA_BASE_CLAIM_ARTIFACT_REQUIRED should be true for public testnet so the service fails closed if finalized artifacts are missing.",
            ],
        ),
        _service(
            key="autoresearch_coordinator",
            title="Autoresearch Coordinator",
            service_type="public_api",
            owner="autoresearch_backend_engineer",
            repo="autoresearch-bittensor",
            cwd=autoresearch_repo,
            url=urls["coordinator"],
            port=8000,
            health_path="/readyz",
            build_commands=["python3 -m pip install -e ."],
            run_command=(
                "python3 -m uvicorn autoresearch_bittensor.api.app:create_app "
                "--factory --host 0.0.0.0 --port 8000"
            ),
            env_file=args.env_file,
            env_public_keys=["SOTA_DEFAULT_LANE_ID", "VALIDATOR_HOTKEYS", "MINER_AUTH_STAKE_GATE_ENABLED"],
            env_public_values={
                "SOTA_DEFAULT_LANE_ID": "base:sota-local",
                "VALIDATOR_HOTKEYS": "",
                "MINER_AUTH_STAKE_GATE_ENABLED": "false",
            },
            env_secret_handles=[
                handles["base_sepolia_autoresearch_database_url"],
                handles["base_sepolia_autoresearch_admin_token"],
            ],
            env_secret_map={
                "DATABASE_URL": handles["base_sepolia_autoresearch_database_url"],
                "ADMIN_TOKEN": handles["base_sepolia_autoresearch_admin_token"],
            },
            depends_on=["base_sepolia_contracts", "self_validation_policy"],
        ),
        _service(
            key="claim_artifacts",
            title="Seed Claim Artifact Builder",
            service_type="private_artifact_worker",
            owner="sre_devops_engineer",
            repo="SN94-BitSota-live-docs",
            cwd=docs_repo,
            url=urls["claim_artifacts"],
            port=None,
            health_path=None,
            build_commands=["python3 -m pip install eth-abi eth-utils"],
            run_command=(
                "python3 scripts/sota_base_testnet_seed_artifacts.py build "
                "--manifest $SOTA_BASE_DEPLOYMENT_MANIFEST --emission-evidence $SOTA_EMISSION_EVIDENCE_JSON "
                "--test-wallet-address $SOTA_TEST_WALLET_ADDRESS --test-old-coldkey $SOTA_TEST_OLD_COLDKEY "
                "--lane-id $NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID && "
                "python3 scripts/sota_base_testnet_seed_artifacts.py finalize "
                "--build-report $SOTA_SEED_BUILD_REPORT_JSON "
                "--genesis-publish-result $SOTA_GENESIS_ROOT_PUBLISH_RESULT_JSON "
                "--emission-publish-result $SOTA_EMISSION_ROOT_PUBLISH_RESULT_JSON"
            ),
            env_file=args.env_file,
            env_public_keys=[
                "SOTA_BASE_DEPLOYMENT_MANIFEST",
                "SOTA_EMISSION_EVIDENCE_JSON",
                "SOTA_TEST_WALLET_ADDRESS",
                "SOTA_TEST_OLD_COLDKEY",
                "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID",
                "SOTA_SEED_BUILD_REPORT_JSON",
                "SOTA_GENESIS_ROOT_PUBLISH_RESULT_JSON",
                "SOTA_EMISSION_ROOT_PUBLISH_RESULT_JSON",
            ],
            env_secret_handles=[],
            depends_on=["autoresearch_coordinator", "base_sepolia_contracts"],
            notes=[
                "Build refuses emission artifacts that lack accepted self-validation committee consensus.",
                "Finalize requires broadcast root-publish results with emitted on-chain root IDs before indexer import.",
            ],
        ),
        _service(
            key="attestation_builder",
            title="Claim And Root Attestation Builder",
            service_type="private_worker",
            owner="smart_contract_security_engineer",
            repo="Pool",
            cwd=pool_repo,
            url=urls["attestation"],
            port=None,
            health_path=None,
            build_commands=["python3 -m pip install -r requirements.txt || true"],
            run_command=(
                "python3 scripts/base_claim_attestation.py manifest --snapshot $SOTA_SNAPSHOT_CONTEXT_JSON "
                "--bindings-jsonl $SOTA_BINDINGS_JSONL --out $SOTA_GENESIS_CLAIM_MANIFEST_JSON && "
                "python3 scripts/foundation_attestation.py build --environment test --root-kind genesis "
                "--epoch-json $SOTA_GENESIS_CLAIM_MANIFEST_JSON --out $SOTA_GENESIS_ATTESTATION_JSON "
                "--bittensor-netuid 0 --bittensor-chain-endpoint $SOTA_SNAPSHOT_SOURCE "
                "--pool-contract ${SOTA_ROOT_REGISTRY_ADDRESS} --publisher ${SOTA_ROOT_PUBLISHER_ADDRESS} "
                "--signer ${SOTA_FOUNDATION_SIGNER_ADDRESS} --quorum 1"
            ),
            env_file=args.env_file,
            env_public_keys=[
                "SOTA_SNAPSHOT_CONTEXT_JSON",
                "SOTA_BINDINGS_JSONL",
                "SOTA_GENESIS_CLAIM_MANIFEST_JSON",
                "SOTA_GENESIS_ATTESTATION_JSON",
                "SOTA_ROOT_REGISTRY_ADDRESS",
                "SOTA_ROOT_PUBLISHER_ADDRESS",
                "SOTA_FOUNDATION_SIGNER_ADDRESS",
            ],
            env_secret_handles=[],
            depends_on=["genesis_snapshot", "coldkey_bindings", "base_sepolia_contracts"],
        ),
        _service(
            key="root_publisher",
            title="Root Publisher Worker",
            service_type="private_worker",
            owner="sre_devops_engineer",
            repo="SN94-BitSota-live-docs",
            cwd=docs_repo,
            url=urls["root_publisher"],
            port=None,
            health_path=None,
            build_commands=["python3 -m pip install web3 eth-abi eth-utils"],
            run_command=(
                "python3 scripts/sota_base_publish_root.py --manifest $SOTA_BASE_DEPLOYMENT_MANIFEST "
                "--root-artifact $SOTA_ROOT_ARTIFACT_JSON --kind $SOTA_ROOT_KIND "
                "--nonce $SOTA_ROOT_NONCE --broadcast --out $SOTA_ROOT_PUBLISH_RESULT_JSON"
            ),
            env_file=args.env_file,
            env_public_keys=[
                "SOTA_BASE_DEPLOYMENT_MANIFEST",
                "SOTA_ROOT_ARTIFACT_JSON",
                "SOTA_ROOT_KIND",
                "SOTA_ROOT_NONCE",
            ],
            env_secret_handles=[
                handles["base_sepolia_rpc"],
                handles["base_sepolia_root_publisher_signer"],
            ],
            depends_on=["claim_artifacts", "attestation_builder", "base_sepolia_contracts"],
            notes=[
                "The wrapper dry-runs unless --broadcast is passed and refuses Base mainnet chain id 8453.",
                "The signer must come from the approved SOTA_ROOT_PUBLISHER_PRIVATE_KEY secret handle at runtime.",
            ],
        ),
        _service(
            key="static_readiness",
            title="Public Testnet Readiness Artifact",
            service_type="static_readiness",
            owner="sre_devops_engineer",
            repo="SN94-BitSota-live-docs",
            cwd=docs_repo,
            url=urls["readiness"],
            port=None,
            health_path=None,
            build_commands=[],
            run_command=(
                "publish /home/mekaneeky/repos/.sota-base-testnet/base-sota-testnet-readiness.json "
                f"to {urls['readiness']}"
            ),
            env_file=args.env_file,
            env_public_keys=["NEXT_PUBLIC_SOTA_READINESS_URL"],
            env_secret_handles=[],
            depends_on=["testnet_preflight"],
        ),
        _service(
            key="monitoring",
            title="Base Sepolia Monitoring",
            service_type="monitoring",
            owner="sre_devops_engineer",
            repo="SN94-BitSota-live-docs",
            cwd=docs_repo,
            url=urls["monitoring"],
            port=None,
            health_path=None,
            build_commands=[],
            run_command="configure uptime, index lag, RPC, root publication, contract pause, API error, and claim failure monitors",
            env_file=args.env_file,
            env_public_keys=[],
            env_secret_handles=[],
            depends_on=["claims_ui", "indexer_api", "autoresearch_coordinator", "root_publisher"],
            notes=[
                f"Optional: use {handles['base_sepolia_monitoring_api_key']} if the chosen monitoring provider requires an API key.",
                "This key is not required for claim, mining, or self-validation testing.",
            ],
        ),
    ]

    checks: list[dict[str, str]] = [_chain_check(manifest)]
    if args.manifest.exists():
        checks.append(_check("manifest_present", True, f"{args.manifest} exists."))
    else:
        checks.append(
            _yellow(
                "manifest_present",
                f"{args.manifest} is missing; service pack is using default URLs and empty contract addresses.",
                "Generate the Base Sepolia deployment manifest before infrastructure deployment.",
            )
        )
    if args.env_file.exists():
        checks.append(_check("env_file_present", True, f"{args.env_file} exists."))
    else:
        checks.append(
            _yellow(
                "env_file_present",
                f"{args.env_file} is missing; service commands still need the generated testnet env file.",
                "Generate base-sota.env.testnet before building or launching public services.",
            )
        )
    checks.append(_check("service_count", len(services) == 8, f"{len(services)} service definitions generated."))
    checks.append(
        _check(
            "dns_hosts_present",
            all(service.get("dns_host") for service in services),
            "Every service has a DNS host or public artifact URL.",
            "Set all Base Sepolia public URLs before assigning infrastructure deployment.",
        )
    )
    checks.append(
        _check(
            "base_sepolia_only",
            not any("basescan.org" in str(service.get("public_url") or "") and "sepolia.basescan.org" not in str(service.get("public_url") or "") for service in services),
            "Service definitions do not configure Base mainnet URLs or chain settings.",
            "Remove any Base mainnet service URL before using this pack.",
        )
    )
    secret_findings = _secret_findings({"services": services, "secret_handles": handles})
    checks.append(
        _check(
            "no_raw_secrets",
            not secret_findings,
            "Service pack contains secret handle references only." if not secret_findings else "; ".join(secret_findings[:5]),
            "Replace raw secrets with approved secret-handle references.",
        )
    )
    missing_wrappers = [service["key"] for service in services if service.get("implementation_status") != "ready_to_configure"]
    if missing_wrappers:
        checks.append(
            _yellow(
                "worker_wrappers",
                f"Missing reusable worker wrapper for: {', '.join(missing_wrappers)}.",
                "Implement the missing guarded worker before public Base Sepolia root publication.",
            )
        )
    else:
        checks.append(_check("worker_wrappers", True, "All worker wrappers are ready to configure."))

    summary = _summary(checks)
    status = _worst(checks)
    return {
        "schema": "sota-base-testnet-service-pack/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": summary["red"] == 0,
        "status": status,
        "deployment_ready": status == "green",
        "message": (
            "Base Sepolia service deployment pack is fully ready to configure."
            if status == "green"
            else "Base Sepolia service deployment pack is generated, but implementation or deployment evidence remains incomplete."
        ),
        "read_only": True,
        "does_not": ["deploy", "sign", "broadcast_transactions", "touch_production_bittensor", "touch_base_mainnet"],
        "artifacts": {
            "manifest": str(args.manifest),
            "env_file": str(args.env_file),
            "readiness_url": urls["readiness"],
        },
        "urls": urls,
        "contracts": contracts,
        "secret_handles": handles,
        "aws_deploy_plan": _aws_deploy_plan(services, handles),
        "services": services,
        "checks": checks,
        "summary": summary,
        "next_actions": [
            check["remediation"]
            for check in checks
            if check["status"] != "green" and check.get("remediation")
        ],
    }


def render_markdown(pack: dict[str, Any]) -> str:
    lines = [
        "# Base SOTA Base Sepolia Service Pack",
        "",
        f"Generated: {pack.get('generated_at')}",
        f"Status: {pack.get('status')}",
        f"Deployment ready: {str(pack.get('deployment_ready')).lower()}",
        "",
        "## Safety",
        "",
    ]
    for item in pack.get("does_not") or []:
        lines.append(f"- Does not {str(item).replace('_', ' ')}.")
    lines.extend(["", "## Services", ""])
    for service in pack.get("services") or []:
        service = dict(service)
        lines.append(f"### {service.get('title')}")
        lines.append("")
        lines.append(f"- Key: `{service.get('key')}`")
        lines.append(f"- Type: `{service.get('type')}`")
        lines.append(f"- Owner: `{service.get('owner')}`")
        lines.append(f"- URL: {service.get('public_url')}")
        if service.get("health_url"):
            lines.append(f"- Health: {service.get('health_url')}")
        lines.append(f"- CWD: `{service.get('cwd')}`")
        lines.append(f"- Status: `{service.get('implementation_status')}`")
        if service.get("build_commands"):
            lines.append("- Build:")
            for command in service.get("build_commands") or []:
                lines.append(f"  - `{command}`")
        lines.append(f"- Run: `{service.get('run_command')}`")
        recipe = dict(service.get("deployment_recipe") or {})
        if recipe:
            lines.append(f"- Deploy target: `{recipe.get('target')}`")
            lines.append(f"- Deploy name: `{recipe.get('service_name')}`")
            if recipe.get("create_service_command"):
                lines.append(f"- App Runner create: `{recipe.get('create_service_command')}`")
        if service.get("env_secret_handles"):
            lines.append("- Secret handles:")
            for handle in service.get("env_secret_handles") or []:
                lines.append(f"  - `{handle}`")
        if service.get("notes"):
            lines.append("- Notes:")
            for note in service.get("notes") or []:
                lines.append(f"  - {note}")
        lines.append("")
    lines.extend(["## Checks", ""])
    for check in pack.get("checks") or []:
        check = dict(check)
        lines.append(f"- [{check.get('status')}] {check.get('name')}: {check.get('detail')}")
    return "\n".join(lines).rstrip() + "\n"


def render_html(pack: dict[str, Any]) -> str:
    status = escape(str(pack.get("status") or "unknown"))
    blocks = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Base SOTA Service Pack</title>",
        "<style>body{font-family:Inter,system-ui,sans-serif;margin:32px;background:#f8fbfb;color:#12252b;line-height:1.55}main{max-width:1100px;margin:0 auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.card{background:white;border:1px solid #d5dde1;border-radius:6px;padding:16px}.status{display:inline-block;border-radius:6px;padding:6px 10px;background:#eef2ff;font-weight:700}code{word-break:break-all;background:#edf5f5;border-radius:4px;padding:2px 4px}li{margin:4px 0}</style>",
        "</head><body><main>",
        "<h1>Base SOTA Base Sepolia Service Pack</h1>",
        f'<p><span class="status">Status: {status}</span></p>',
        f"<p>{escape(str(pack.get('message') or ''))}</p>",
        '<section class="grid">',
    ]
    for service in pack.get("services") or []:
        service = dict(service)
        blocks.append('<article class="card">')
        blocks.append(f"<h2>{escape(str(service.get('title')))}</h2>")
        blocks.append(f"<p><strong>{escape(str(service.get('type')))}</strong> / {escape(str(service.get('implementation_status')))}</p>")
        blocks.append(f"<p><a href=\"{escape(str(service.get('public_url')))}\">{escape(str(service.get('public_url')))}</a></p>")
        if service.get("health_url"):
            blocks.append(f"<p>Health: <a href=\"{escape(str(service.get('health_url')))}\">{escape(str(service.get('health_url')))}</a></p>")
        blocks.append(f"<p>CWD: <code>{escape(str(service.get('cwd')))}</code></p>")
        blocks.append(f"<p>Run: <code>{escape(str(service.get('run_command')))}</code></p>")
        recipe = dict(service.get("deployment_recipe") or {})
        if recipe:
            blocks.append(
                f"<p>Deploy: <code>{escape(str(recipe.get('target')))}</code> as "
                f"<code>{escape(str(recipe.get('service_name')))}</code></p>"
            )
            if recipe.get("create_service_command"):
                blocks.append(f"<p>Create: <code>{escape(str(recipe.get('create_service_command')))}</code></p>")
        blocks.append("</article>")
    blocks.append("</section><h2>Checks</h2><ul>")
    for check in pack.get("checks") or []:
        check = dict(check)
        blocks.append(f"<li><strong>{escape(str(check.get('status')))}</strong> {escape(str(check.get('name')))}: {escape(str(check.get('detail')))}</li>")
    blocks.append("</ul></main></body></html>")
    return "\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Base SOTA Base Sepolia service deployment pack.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--claims-ui", default="")
    parser.add_argument("--claims-api", default="")
    parser.add_argument("--coordinator", default="")
    parser.add_argument("--attestation", default="")
    parser.add_argument("--root-publisher", default="")
    parser.add_argument("--claim-artifacts", default="")
    parser.add_argument("--monitoring", default="")
    parser.add_argument("--readiness-url", default=DEFAULT_URLS["readiness"])
    parser.add_argument("--json-out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-service-pack.json")
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-service-pack.md")
    parser.add_argument("--html-out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-service-pack.html")
    parser.add_argument("--apprunner-out-dir", type=Path, default=DEFAULT_APPRUNNER_OUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    pack = build_service_pack(args)
    _write_json(args.json_out, pack)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(pack), encoding="utf-8")
    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(render_html(pack), encoding="utf-8")
    apprunner_inputs = write_apprunner_inputs(pack, args.apprunner_out_dir)
    output = {
        "ok": bool(pack.get("ok")),
        "status": pack.get("status"),
        "deployment_ready": bool(pack.get("deployment_ready")),
        "json": str(args.json_out),
        "markdown": str(args.markdown_out),
        "html": str(args.html_out),
        "apprunner_inputs": apprunner_inputs,
    }
    print(json.dumps(pack if args.json else output, indent=2 if args.json else None, sort_keys=True))
    return 0 if pack.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

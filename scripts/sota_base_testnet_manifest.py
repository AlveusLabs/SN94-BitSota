#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from pathlib import Path
from typing import Any


BASE_SEPOLIA_CHAIN_ID = 84532
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
CONTRACT_KEYS = (
    "sota_token",
    "vault",
    "root_registry",
    "lane_registry",
    "genesis_distributor",
    "emission_distributor",
)
SERVICE_ARGS = {
    "claims_ui": ("claims_ui_url", "claims_ui_health_url"),
    "indexer_api": ("indexer_api_url", "indexer_api_health_url"),
    "root_publisher": ("root_publisher_url", "root_publisher_health_url"),
    "attestation_builder": ("attestation_builder_url", "attestation_builder_health_url"),
    "monitoring": ("monitoring_url", None),
}
PUBLIC_ENV_KEYS = (
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
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "0x" + sha256(payload).hexdigest()


def _require_address(value: Any, label: str) -> str:
    if not isinstance(value, str) or not EVM_ADDRESS_RE.fullmatch(value) or value.lower() == ZERO_ADDRESS:
        raise SystemExit(f"{label} must be a nonzero EVM address")
    return value


def _optional_address(value: str | None, label: str) -> str | None:
    if value is None or value == "":
        return None
    return _require_address(value, label)


def _nonempty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _deployment_contracts(deployment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contracts = deployment.get("contracts")
    if not isinstance(contracts, dict):
        raise SystemExit("deployment manifest is missing contracts")
    missing = sorted(set(CONTRACT_KEYS) - set(contracts))
    if missing:
        raise SystemExit(f"deployment manifest is missing contracts: {', '.join(missing)}")
    return contracts


def _assert_base_sepolia(deployment: dict[str, Any]) -> None:
    chain_id = int(deployment.get("chain_id") or 0)
    if chain_id != BASE_SEPOLIA_CHAIN_ID:
        raise SystemExit(f"deployment chain_id must be Base Sepolia 84532, got {chain_id}")
    chain_name = str(deployment.get("chain_name") or "")
    if chain_name != "base-sepolia":
        raise SystemExit(f"deployment chain_name must be base-sepolia, got {chain_name!r}")


def _service_url_for(manifest: dict[str, Any], key: str) -> str | None:
    service = manifest.get("services", {}).get(key, {})
    if not isinstance(service, dict):
        return None
    return (
        _nonempty(service.get("public_url"))
        or _nonempty(service.get("public_base_url"))
        or _nonempty(service.get("service_url"))
        or _nonempty(service.get("health_url"))
    )


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    template = _load_json(args.template)
    deployment = _load_json(args.deployment)
    _assert_base_sepolia(deployment)
    deployed_contracts = _deployment_contracts(deployment)

    manifest = deepcopy(template)
    manifest["status"] = "base-sepolia-deployed"
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["deployment_source_manifest"] = str(args.deployment)
    manifest["environment"] = "base-sepolia"
    manifest["chain"]["chain_id"] = BASE_SEPOLIA_CHAIN_ID
    manifest["chain"]["chain_name"] = "base-sepolia"
    manifest["chain"]["network_display_name"] = "Base Sepolia"
    manifest["chain"]["block_explorer_url"] = str(
        deployment.get("block_explorer_url") or "https://sepolia.basescan.org"
    )
    manifest["chain"]["public_browser_rpc_url"] = args.public_rpc_url

    manifest["deployer"]["address"] = _optional_address(deployment.get("deployer"), "deployer.address")
    roles = deployment.get("roles") if isinstance(deployment.get("roles"), dict) else {}
    for role_key in ("owner", "supply_authority", "emission_authority", "root_publisher"):
        address = _optional_address(roles.get(role_key), f"roles.{role_key}")
        if address is not None:
            manifest["roles"][role_key]["address"] = address

    if args.pause_guardian_address:
        manifest["roles"]["pause_guardian"]["address"] = _optional_address(
            args.pause_guardian_address, "pause_guardian_address"
        )
    if args.owner_address:
        manifest["roles"]["owner"]["address"] = _optional_address(args.owner_address, "owner_address")
    if args.root_publisher_address:
        manifest["roles"]["root_publisher"]["address"] = _optional_address(
            args.root_publisher_address, "root_publisher_address"
        )

    deployment_block = int(deployment.get("deployment_block") or 0)
    for key in CONTRACT_KEYS:
        source = deployed_contracts[key]
        target = manifest["contracts"][key]
        address = _require_address(source.get("address"), f"contracts.{key}.address")
        target["address"] = address
        target["deployment_tx_hash"] = source.get("deployment_tx_hash")
        target["deployment_block"] = source.get("deployment_block") or deployment_block
        target["constructor_args_hash"] = _canonical_hash(source.get("constructor_args", []))
        if args.source_verification_base_url:
            target["source_verification_url"] = (
                args.source_verification_base_url.rstrip("/") + f"/address/{address}#code"
            )
        manifest["browser_safe"]["contract_addresses"][key] = address

    manifest["browser_safe"]["public_browser_rpc_url"] = args.public_rpc_url
    manifest["browser_safe"]["block_explorer_url"] = manifest["chain"]["block_explorer_url"]
    manifest["browser_safe"]["chain_id"] = BASE_SEPOLIA_CHAIN_ID
    manifest["browser_safe"]["chain_name"] = "base-sepolia"
    manifest["browser_safe"]["network_display_name"] = "Base Sepolia"

    for key, (url_attr, health_attr) in SERVICE_ARGS.items():
        url = _nonempty(getattr(args, url_attr))
        health_url = _nonempty(getattr(args, health_attr)) if health_attr else None
        service = manifest["services"][key]
        if key == "claims_ui":
            service["public_url"] = url
        elif key == "indexer_api":
            service["public_base_url"] = url
        elif key in {"root_publisher", "attestation_builder"}:
            service["service_url"] = url
        elif key == "monitoring":
            service["dashboard_url"] = url
            service["alert_policy_url"] = _nonempty(args.monitoring_alert_policy_url)
            service["log_group_or_sink"] = _nonempty(args.monitoring_log_group_or_sink)
        if health_url:
            service["health_url"] = health_url

    manifest["browser_safe"]["service_urls"]["claims_ui"] = _service_url_for(manifest, "claims_ui")
    manifest["browser_safe"]["service_urls"]["indexer_api"] = _service_url_for(manifest, "indexer_api")
    claims_ui_env = manifest["services"]["claims_ui"].setdefault("browser_safe_env", {})
    claims_ui_env.update(
        {
            "NEXT_PUBLIC_SOTA_ENVIRONMENT": "testnet",
            "NEXT_PUBLIC_SOTA_BASE_CHAIN_ID": str(BASE_SEPOLIA_CHAIN_ID),
            "NEXT_PUBLIC_SOTA_BASE_CHAIN_NAME": "Base Sepolia",
            "NEXT_PUBLIC_SOTA_BASE_RPC_URL": args.public_rpc_url,
            "NEXT_PUBLIC_SOTA_BASE_EXPLORER_URL": manifest["chain"]["block_explorer_url"],
            "NEXT_PUBLIC_SOTA_CLAIMS_API_URL": manifest["browser_safe"]["service_urls"]["indexer_api"] or "",
            "NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL": _nonempty(args.autoresearch_api_url) or "",
            "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID": args.default_lane_id,
            "NEXT_PUBLIC_SOTA_GENESIS_DISTRIBUTOR_ADDRESS": manifest["contracts"]["genesis_distributor"]["address"],
            "NEXT_PUBLIC_SOTA_EMISSION_DISTRIBUTOR_ADDRESS": manifest["contracts"]["emission_distributor"]["address"],
            "NEXT_PUBLIC_SOTA_TOKEN_ADDRESS": manifest["contracts"]["sota_token"]["address"],
            "NEXT_PUBLIC_SOTA_READINESS_URL": _nonempty(args.readiness_url) or "",
            "NEXT_PUBLIC_SOTA_DEMO_ENABLED": "false",
        }
    )

    if args.claims_ui_url:
        manifest["evidence_links"]["claims_ui_smoke"] = args.claims_ui_url
    if args.monitoring_url:
        manifest["evidence_links"]["monitoring_dashboard"] = args.monitoring_url
    manifest["evidence_links"]["deployment_run"] = str(args.deployment)

    indexer = manifest["services"]["indexer_api"]
    indexer["sync_from_block"] = deployment_block
    manifest["notes"] = [
        note
        for note in manifest.get("notes", [])
        if "Template only" not in str(note)
    ]
    manifest["notes"].append(
        "Generated from the compact contract deployment manifest. Contains public addresses and service URLs only."
    )
    return manifest


def build_env(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, str]:
    contracts = manifest["contracts"]
    claims_api_url = _service_url_for(manifest, "indexer_api") or args.claims_api_url or ""
    autoresearch_url = _nonempty(args.autoresearch_api_url) or ""
    env = {
        "SOTA_ENVIRONMENT": "testnet",
        "SOTA_BASE_DEPLOYMENT_MANIFEST": str(args.manifest_out) if args.manifest_out else "",
        "SOTA_BASE_CHAIN_ID": str(BASE_SEPOLIA_CHAIN_ID),
        "SOTA_BASE_RPC_URL": args.public_rpc_url,
        "SOTA_BASE_SYNC_FROM_BLOCK": str(manifest["services"]["indexer_api"].get("sync_from_block") or ""),
        "SOTA_BASE_CONTRACTS_ABI_DIR": "/home/mekaneeky/repos/Pool/contracts/sota-base/abi",
        "SOTA_TOKEN_ADDRESS": contracts["sota_token"]["address"],
        "SOTA_VAULT_ADDRESS": contracts["vault"]["address"],
        "SOTA_ROOT_REGISTRY_ADDRESS": contracts["root_registry"]["address"],
        "SOTA_LANE_REGISTRY_ADDRESS": contracts["lane_registry"]["address"],
        "SOTA_GENESIS_DISTRIBUTOR_ADDRESS": contracts["genesis_distributor"]["address"],
        "SOTA_EMISSION_DISTRIBUTOR_ADDRESS": contracts["emission_distributor"]["address"],
        "SOTA_CLAIMS_API_URL": claims_api_url,
        "SOTA_TEST_WALLET_ADDRESS": args.test_wallet_address or "",
        "SOTA_TEST_OLD_COLDKEY": args.test_old_coldkey or "",
        "SOTA_TEST_EPOCH": str(args.test_epoch or "1"),
        "NEXT_PUBLIC_SOTA_ENVIRONMENT": "testnet",
        "NEXT_PUBLIC_SOTA_BASE_CHAIN_ID": str(BASE_SEPOLIA_CHAIN_ID),
        "NEXT_PUBLIC_SOTA_BASE_CHAIN_NAME": "Base Sepolia",
        "NEXT_PUBLIC_SOTA_BASE_RPC_URL": args.public_rpc_url,
        "NEXT_PUBLIC_SOTA_BASE_EXPLORER_URL": manifest["chain"]["block_explorer_url"],
        "NEXT_PUBLIC_SOTA_CLAIMS_API_URL": claims_api_url,
        "NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL": autoresearch_url,
        "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID": args.default_lane_id,
        "NEXT_PUBLIC_SOTA_CLAIMS_CONTRACT_ADDRESS": contracts["genesis_distributor"]["address"],
        "NEXT_PUBLIC_SOTA_GENESIS_DISTRIBUTOR_ADDRESS": contracts["genesis_distributor"]["address"],
        "NEXT_PUBLIC_SOTA_EMISSION_DISTRIBUTOR_ADDRESS": contracts["emission_distributor"]["address"],
        "NEXT_PUBLIC_SOTA_TOKEN_ADDRESS": contracts["sota_token"]["address"],
        "NEXT_PUBLIC_SOTA_READINESS_URL": args.readiness_url,
        "NEXT_PUBLIC_SOTA_DEMO_ENABLED": "false",
    }
    if args.test_wallet_address:
        env["SOTA_TEST_WALLET_ADDRESS"] = _require_address(args.test_wallet_address, "test_wallet_address")
    return env


def write_env(path: Path, env: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated Base SOTA Base Sepolia public/service env.",
        "# Public addresses and URLs only. Do not add private keys, mnemonics, admin tokens, or RPC tokens here.",
    ]
    for key in sorted(env):
        value = env[key]
        if key in PUBLIC_ENV_KEYS:
            continue
        lines.append(f"{key}={value}")
    lines.append("")
    lines.append("# Browser-public values for the claims website.")
    for key in PUBLIC_ENV_KEYS:
        lines.append(f"{key}={env[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Base Sepolia preflight manifest/env from a SOTA Base deployment.")
    parser.add_argument("--template", type=Path, required=True, help="Full Base Sepolia manifest template JSON")
    parser.add_argument("--deployment", type=Path, required=True, help="Compact output from Pool/scripts/deploy_sota_base.py")
    parser.add_argument("--manifest-out", type=Path, required=True, help="Path to write the filled full manifest")
    parser.add_argument("--env-out", type=Path, required=True, help="Path to write the public/service env file")
    parser.add_argument("--public-rpc-url", default="https://sepolia.base.org")
    parser.add_argument("--default-lane-id", default="base:sota-local")
    parser.add_argument("--claims-ui-url", default="")
    parser.add_argument("--claims-ui-health-url", default="")
    parser.add_argument("--indexer-api-url", default="")
    parser.add_argument("--indexer-api-health-url", default="")
    parser.add_argument("--root-publisher-url", default="")
    parser.add_argument("--root-publisher-health-url", default="")
    parser.add_argument("--attestation-builder-url", default="")
    parser.add_argument("--attestation-builder-health-url", default="")
    parser.add_argument("--monitoring-url", default="")
    parser.add_argument("--monitoring-alert-policy-url", default="")
    parser.add_argument("--monitoring-log-group-or-sink", default="")
    parser.add_argument("--claims-api-url", default="")
    parser.add_argument("--autoresearch-api-url", default="")
    parser.add_argument("--test-wallet-address", default="")
    parser.add_argument("--test-old-coldkey", default="")
    parser.add_argument("--test-epoch", default="1")
    parser.add_argument("--readiness-url", default="")
    parser.add_argument("--owner-address", default="")
    parser.add_argument("--root-publisher-address", default="")
    parser.add_argument("--pause-guardian-address", default="")
    parser.add_argument("--source-verification-base-url", default="https://sepolia.basescan.org")
    args = parser.parse_args(argv)

    manifest = build_manifest(args)
    env = build_env(args, manifest)
    _write_json(args.manifest_out, manifest)
    write_env(args.env_out, env)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(args.manifest_out),
                "env": str(args.env_out),
                "contracts": {key: manifest["contracts"][key]["address"] for key in CONTRACT_KEYS},
                "services": manifest["browser_safe"]["service_urls"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

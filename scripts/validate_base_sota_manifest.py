#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sota-base-sepolia-deployment-manifest/v1"
BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
REQUIRED_CONTRACTS = {
    "sota_token": "SOTAToken",
    "vault": "SOTAVault",
    "root_registry": "SOTARootRegistry",
    "lane_registry": "SOTALaneRegistry",
    "genesis_distributor": "GenesisClaimDistributor",
    "emission_distributor": "EmissionClaimDistributor",
}
REQUIRED_SERVICES = {
    "claims_ui",
    "indexer_api",
    "root_publisher",
    "attestation_builder",
}
REQUIRED_TOP_LEVEL = {
    "manifest_schema_version",
    "environment",
    "chain",
    "source",
    "abi_bundle",
    "deployer",
    "roles",
    "contracts",
    "services",
    "browser_safe",
    "secret_handles",
    "rollback",
    "evidence_links",
}
SENSITIVE_KEY_RE = re.compile(
    r"(private.?key|mnemonic|seed|rpc.?token|admin.?token|api.?key|password|suri|secret)",
    re.IGNORECASE,
)
EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
HANDLE_PREFIXES = (
    "TODO:secret-handle:",
    "aws-secretsmanager:",
    "gcp-secret-manager:",
    "vault:",
    "op://",
    "doppler://",
    "env:",
)


class ManifestError(ValueError):
    pass


def _walk(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    rows = [(path, value)]
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_walk(item, (*path, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_walk(item, (*path, str(index))))
    return rows


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _handle_like(value: str) -> bool:
    return value.startswith(HANDLE_PREFIXES)


def _assert_no_secret_values(manifest: dict[str, Any]) -> None:
    for path, value in _walk(manifest):
        if not isinstance(value, str):
            continue
        dotted = ".".join(path)
        lower = value.lower()
        _require("test test test test test test test test test test test junk" not in lower, f"demo mnemonic leaked at {dotted}")
        if "secret_handles" in path:
            _require(_handle_like(value), f"secret handle {dotted} must be a handle, not a raw value")
            continue
        if path and SENSITIVE_KEY_RE.search(path[-1]):
            _require(
                value == "" or _handle_like(value) or path[-1].endswith("_ref"),
                f"sensitive-looking field {dotted} must contain a handle/ref only",
            )


def _assert_browser_safe(manifest: dict[str, Any]) -> None:
    browser_safe = manifest.get("browser_safe")
    _require(isinstance(browser_safe, dict), "browser_safe must be an object")
    for path, value in _walk(browser_safe):
        dotted = ".".join(("browser_safe", *path))
        _require(not any(part == "secret_handles" for part in path), f"{dotted} must not contain secret_handles")
        if path and SENSITIVE_KEY_RE.search(path[-1]):
            raise ManifestError(f"{dotted} is not browser-safe")
        if isinstance(value, str):
            _require(not _handle_like(value), f"{dotted} must not contain a secret handle")


def _assert_chain(manifest: dict[str, Any]) -> None:
    _require(manifest.get("manifest_schema_version") == SCHEMA_VERSION, "unexpected manifest_schema_version")
    _require(manifest.get("environment") == "base-sepolia", "environment must be base-sepolia")
    chain = manifest.get("chain")
    _require(isinstance(chain, dict), "chain must be an object")
    _require(chain.get("chain_id") == BASE_SEPOLIA_CHAIN_ID, "chain.chain_id must be 84532")
    _require(chain.get("chain_name") == "base-sepolia", "chain.chain_name must be base-sepolia")
    _require(chain.get("chain_id") != BASE_MAINNET_CHAIN_ID, "Base mainnet chain id is not allowed")
    explorer = str(chain.get("block_explorer_url") or "")
    _require("sepolia.basescan.org" in explorer, "block explorer must be Base Sepolia")
    _require("https://basescan.org" not in explorer.rstrip("/"), "Base mainnet explorer is not allowed")
    _require(bool(chain.get("rpc_secret_handle_ref")), "chain.rpc_secret_handle_ref is required")


def _assert_contracts(manifest: dict[str, Any]) -> None:
    contracts = manifest.get("contracts")
    _require(isinstance(contracts, dict), "contracts must be an object")
    missing = sorted(set(REQUIRED_CONTRACTS) - set(contracts))
    _require(not missing, f"missing contracts: {', '.join(missing)}")
    for key, expected_name in REQUIRED_CONTRACTS.items():
        item = contracts.get(key)
        _require(isinstance(item, dict), f"contracts.{key} must be an object")
        _require(item.get("name") == expected_name, f"contracts.{key}.name must be {expected_name}")
        for field in ("address", "source_verification_url", "abi_path", "required_by"):
            _require(field in item, f"contracts.{key}.{field} is required")
        address = item.get("address")
        if address is not None:
            _require(isinstance(address, str) and EVM_ADDRESS_RE.fullmatch(address), f"contracts.{key}.address is invalid")


def _assert_services(manifest: dict[str, Any]) -> None:
    services = manifest.get("services")
    _require(isinstance(services, dict), "services must be an object")
    missing = sorted(REQUIRED_SERVICES - set(services))
    _require(not missing, f"missing services: {', '.join(missing)}")
    for key in REQUIRED_SERVICES:
        item = services.get(key)
        _require(isinstance(item, dict), f"services.{key} must be an object")
        _require("owner" in item, f"services.{key}.owner is required")
        _require("source_branch" in item, f"services.{key}.source_branch is required")
        _require("commit_sha" in item, f"services.{key}.commit_sha is required")


def _assert_source_and_ops(manifest: dict[str, Any]) -> None:
    source = manifest.get("source")
    _require(isinstance(source, dict), "source must be an object")
    for key in ("docs", "contracts", "claims_ui", "indexer_api", "attestation_tools"):
        _require(key in source, f"source.{key} is required")
        _require("branch" in source[key], f"source.{key}.branch is required")
        _require("commit_sha" in source[key], f"source.{key}.commit_sha is required")
    _require("version" in manifest.get("abi_bundle", {}), "abi_bundle.version is required")
    _require("address" in manifest.get("deployer", {}), "deployer.address is required")
    _require("owner" in manifest.get("rollback", {}), "rollback.owner is required")
    _require(isinstance(manifest.get("evidence_links"), dict), "evidence_links must be an object")


def validate_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON: {exc}") from exc
    _require(isinstance(manifest, dict), "manifest must be a JSON object")
    missing = sorted(REQUIRED_TOP_LEVEL - set(manifest))
    _require(not missing, f"missing top-level fields: {', '.join(missing)}")
    _assert_chain(manifest)
    _assert_source_and_ops(manifest)
    _assert_contracts(manifest)
    _assert_services(manifest)
    _assert_browser_safe(manifest)
    _assert_no_secret_values(manifest)
    return {
        "ok": True,
        "path": str(path),
        "environment": manifest["environment"],
        "chain_id": manifest["chain"]["chain_id"],
        "contracts": sorted(REQUIRED_CONTRACTS),
        "services": sorted(REQUIRED_SERVICES),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Base SOTA deployment manifest.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_manifest(args.manifest)
    except ManifestError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from validate_base_sota_manifest import ManifestError, validate_manifest


BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
REQUIRED_CONTRACT_KEYS = (
    "sota_token",
    "vault",
    "root_registry",
    "lane_registry",
    "genesis_distributor",
    "emission_distributor",
)
REQUIRED_SERVICE_KEYS = (
    "claims_ui",
    "indexer_api",
    "root_publisher",
    "attestation_builder",
)
REQUIRED_PUBLIC_ENV = {
    "NEXT_PUBLIC_SOTA_ENVIRONMENT": {"testnet", "base-sepolia", "sepolia"},
    "NEXT_PUBLIC_SOTA_BASE_CHAIN_ID": {"84532"},
    "NEXT_PUBLIC_SOTA_BASE_CHAIN_NAME": {"Base Sepolia", "base-sepolia"},
    "NEXT_PUBLIC_SOTA_BASE_RPC_URL": None,
    "NEXT_PUBLIC_SOTA_BASE_EXPLORER_URL": None,
    "NEXT_PUBLIC_SOTA_CLAIMS_API_URL": None,
    "NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL": None,
    "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID": None,
    "NEXT_PUBLIC_SOTA_GENESIS_DISTRIBUTOR_ADDRESS": None,
    "NEXT_PUBLIC_SOTA_EMISSION_DISTRIBUTOR_ADDRESS": None,
    "NEXT_PUBLIC_SOTA_TOKEN_ADDRESS": None,
}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    remediation: str = ""

    def as_dict(self) -> dict[str, str]:
        payload = {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }
        if self.remediation:
            payload["remediation"] = self.remediation
        return payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_env(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def _json_rpc(rpc_url: str, method: str, params: list[Any] | None = None, *, timeout: float) -> Any:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
    request = Request(
        rpc_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "sota-base-testnet-preflight/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if "error" in body:
        raise RuntimeError(str(body["error"]))
    return body.get("result")


def _http_status(url: str, *, timeout: float) -> tuple[int | None, str]:
    if not url:
        return None, "missing URL"
    request = Request(url, headers={"Accept": "application/json,text/html;q=0.9,*/*;q=0.8"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), "ok"
    except HTTPError as exc:
        return int(exc.code), str(exc)
    except (URLError, TimeoutError, OSError) as exc:
        return None, str(exc)


def _is_evm_address(value: Any) -> bool:
    return isinstance(value, str) and bool(EVM_ADDRESS_RE.fullmatch(value))


def _is_zero_or_missing(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, str) and value.lower() == ZERO_ADDRESS)


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(status, 2)


def _worst(checks: list[Check]) -> str:
    if not checks:
        return "green"
    return max((check.status for check in checks), key=_status_rank)


def _manifest_validation_check(manifest_path: Path) -> Check:
    try:
        validate_manifest(manifest_path)
    except (ManifestError, FileNotFoundError, json.JSONDecodeError) as exc:
        return Check(
            "manifest_schema",
            "red",
            f"{manifest_path}: {exc}",
            "Fix the Base Sepolia manifest shape before running service or browser checks.",
        )
    return Check("manifest_schema", "green", f"{manifest_path} matches the Base Sepolia manifest schema.")


def _chain_config_check(manifest: dict[str, Any], env: dict[str, str]) -> Check:
    chain = manifest.get("chain") or {}
    manifest_chain_id = chain.get("chain_id")
    env_chain_id = env.get("BASE_CHAIN_ID") or env.get("SOTA_BASE_CHAIN_ID") or env.get("NEXT_PUBLIC_SOTA_BASE_CHAIN_ID")
    if manifest_chain_id == BASE_MAINNET_CHAIN_ID or env_chain_id == str(BASE_MAINNET_CHAIN_ID):
        return Check("chain_config", "red", "Base mainnet chain id 8453 is present.", "Use Base Sepolia chain id 84532 only.")
    if manifest.get("environment") != "base-sepolia" or manifest_chain_id != BASE_SEPOLIA_CHAIN_ID:
        return Check("chain_config", "red", "Manifest is not pinned to Base Sepolia.", "Set environment=base-sepolia and chain.chain_id=84532.")
    if env_chain_id and env_chain_id != str(BASE_SEPOLIA_CHAIN_ID):
        return Check("chain_config", "red", f"Env chain id is {env_chain_id}, expected 84532.", "Fix the env file before building the UI or services.")
    return Check("chain_config", "green", "Manifest/env chain configuration is Base Sepolia only.")


def _rpc_check(rpc_url: str, *, timeout: float, offline: bool) -> Check:
    if not rpc_url:
        return Check("rpc_chain_id", "red", "No Base Sepolia RPC URL configured.", "Set chain.public_browser_rpc_url or BASE_RPC_URL.")
    if offline:
        return Check("rpc_chain_id", "yellow", f"Skipped network check for {rpc_url}.", "Run without --offline before opening testnet to users.")
    try:
        raw_chain_id = _json_rpc(rpc_url, "eth_chainId", timeout=timeout)
    except Exception as exc:
        return Check("rpc_chain_id", "red", f"RPC request failed for {rpc_url}: {exc}", "Fix the Base Sepolia RPC provider or secret handle.")
    try:
        chain_id = int(str(raw_chain_id), 16)
    except ValueError:
        return Check("rpc_chain_id", "red", f"RPC returned invalid chain id {raw_chain_id!r}.", "Use a standard EVM JSON-RPC endpoint.")
    if chain_id == BASE_MAINNET_CHAIN_ID:
        return Check("rpc_chain_id", "red", "RPC returned Base mainnet chain id 8453.", "Do not use this RPC for testnet.")
    if chain_id != BASE_SEPOLIA_CHAIN_ID:
        return Check("rpc_chain_id", "red", f"RPC returned chain id {chain_id}, expected 84532.", "Point the config at Base Sepolia.")
    return Check("rpc_chain_id", "green", f"RPC returned Base Sepolia chain id {chain_id}.")


def _contract_checks(manifest: dict[str, Any], rpc_url: str, *, timeout: float, offline: bool) -> list[Check]:
    checks: list[Check] = []
    contracts = manifest.get("contracts") or {}
    for key in REQUIRED_CONTRACT_KEYS:
        item = contracts.get(key) or {}
        address = item.get("address")
        if _is_zero_or_missing(address):
            checks.append(
                Check(
                    f"contract_{key}",
                    "red",
                    f"{key} address is missing or zero.",
                    "Deploy Base Sepolia contracts and write nonzero addresses into the manifest.",
                )
            )
            continue
        if not _is_evm_address(address):
            checks.append(Check(f"contract_{key}", "red", f"{key} address is invalid: {address!r}.", "Use a 20-byte EVM address."))
            continue
        if offline:
            checks.append(Check(f"contract_{key}", "yellow", f"{key} has address {address}; bytecode check skipped.", "Run without --offline before testnet review."))
            continue
        try:
            code = _json_rpc(rpc_url, "eth_getCode", [address, "latest"], timeout=timeout)
        except Exception as exc:
            checks.append(Check(f"contract_{key}", "red", f"Could not read bytecode for {key} at {address}: {exc}", "Fix RPC or address before browser smoke."))
            continue
        if not isinstance(code, str) or code == "0x":
            checks.append(Check(f"contract_{key}", "red", f"{key} at {address} has no bytecode on Base Sepolia.", "Deploy the contract or correct the manifest address."))
        else:
            checks.append(Check(f"contract_{key}", "green", f"{key} has bytecode at {address}."))
    return checks


def _service_checks(manifest: dict[str, Any], *, timeout: float, offline: bool) -> list[Check]:
    checks: list[Check] = []
    services = manifest.get("services") or {}
    for key in REQUIRED_SERVICE_KEYS:
        service = services.get(key) or {}
        url = (
            service.get("health_url")
            or service.get("public_url")
            or service.get("public_base_url")
            or service.get("service_url")
            or service.get("dashboard_url")
        )
        if not url:
            checks.append(
                Check(
                    f"service_{key}",
                    "red",
                    f"{key} has no public or health URL in the manifest.",
                    "Deploy the service or record the Base Sepolia health/public URL.",
                )
            )
            continue
        if offline:
            checks.append(Check(f"service_{key}", "yellow", f"{key} URL configured as {url}; HTTP check skipped.", "Run without --offline before inviting testers."))
            continue
        status, detail = _http_status(str(url), timeout=timeout)
        if status is not None and 200 <= status < 500:
            checks.append(Check(f"service_{key}", "green", f"{key} responded with HTTP {status} at {url}."))
        else:
            checks.append(Check(f"service_{key}", "red", f"{key} did not respond at {url}: {detail}", "Fix DNS/service health before browser smoke."))
    return checks


def _public_env_checks(env: dict[str, str]) -> list[Check]:
    checks: list[Check] = []
    if not env:
        return [
            Check(
                "public_env",
                "yellow",
                "No env file was supplied.",
                "Run with --env-file bitsota_website/docs/operations/base-sota.env.testnet.example or the real testnet env.",
            )
        ]
    for key, allowed_values in REQUIRED_PUBLIC_ENV.items():
        value = env.get(key)
        if not value:
            checks.append(Check(f"env_{key}", "red", f"{key} is missing.", "Add the public testnet env value before building the claims UI."))
            continue
        if allowed_values and value not in allowed_values:
            checks.append(Check(f"env_{key}", "red", f"{key}={value!r}, expected one of {sorted(allowed_values)}.", "Use Base Sepolia browser config."))
            continue
        if key.endswith("_ADDRESS") and (not _is_evm_address(value) or value.lower() == ZERO_ADDRESS):
            checks.append(Check(f"env_{key}", "red", f"{key} is zero or invalid.", "Populate nonzero deployed Base Sepolia contract addresses."))
            continue
        checks.append(Check(f"env_{key}", "green", f"{key} is configured."))
    return checks


def _wallet_check(env: dict[str, str], rpc_url: str, *, timeout: float, offline: bool) -> Check:
    wallet = env.get("SOTA_TEST_WALLET_ADDRESS") or env.get("TEST_WALLET_ADDRESS") or env.get("SOTA_BROWSER_SMOKE_WALLET_ADDRESS")
    if not wallet:
        return Check(
            "test_wallet",
            "yellow",
            "No public test wallet address configured.",
            "Set SOTA_TEST_WALLET_ADDRESS to a funded Base Sepolia wallet before browser smoke.",
        )
    if not _is_evm_address(wallet) or wallet.lower() == ZERO_ADDRESS:
        return Check("test_wallet", "red", f"Test wallet address is invalid: {wallet!r}.", "Use a public 20-byte EVM address only; never paste private keys.")
    if offline:
        return Check("test_wallet", "yellow", f"Wallet {wallet} configured; balance check skipped.", "Run without --offline before browser smoke.")
    try:
        raw_balance = _json_rpc(rpc_url, "eth_getBalance", [wallet, "latest"], timeout=timeout)
        balance_wei = int(str(raw_balance), 16)
    except Exception as exc:
        return Check("test_wallet", "red", f"Could not read test wallet balance for {wallet}: {exc}", "Fix RPC or wallet address before browser smoke.")
    if balance_wei <= 0:
        return Check("test_wallet", "red", f"Test wallet {wallet} has zero ETH on Base Sepolia.", "Fund it with Base Sepolia test ETH before browser smoke.")
    eth = balance_wei / 10**18
    return Check("test_wallet", "green", f"Test wallet {wallet} has {eth:.6f} ETH on Base Sepolia.")


def run_preflight(
    manifest_path: Path,
    *,
    env_file: Path | None = None,
    timeout: float = 10.0,
    offline: bool = False,
) -> dict[str, Any]:
    env = _load_env(env_file)
    checks = [_manifest_validation_check(manifest_path)]
    try:
        manifest = _load_json(manifest_path)
    except Exception:
        manifest = {}
    checks.append(_chain_config_check(manifest, env))
    chain = manifest.get("chain") or {}
    rpc_url = env.get("BASE_RPC_URL") or env.get("NEXT_PUBLIC_SOTA_BASE_RPC_URL") or str(chain.get("public_browser_rpc_url") or "")
    checks.append(_rpc_check(rpc_url, timeout=timeout, offline=offline))
    checks.extend(_contract_checks(manifest, rpc_url, timeout=timeout, offline=offline))
    checks.extend(_service_checks(manifest, timeout=timeout, offline=offline))
    checks.extend(_public_env_checks(env))
    checks.append(_wallet_check(env, rpc_url, timeout=timeout, offline=offline))
    status = _worst(checks)
    return {
        "ok": status == "green",
        "status": status,
        "manifest": str(manifest_path),
        "env_file": str(env_file) if env_file else None,
        "offline": offline,
        "checks": [check.as_dict() for check in checks],
        "summary": {
            "green": sum(1 for check in checks if check.status == "green"),
            "yellow": sum(1 for check in checks if check.status == "yellow"),
            "red": sum(1 for check in checks if check.status == "red"),
        },
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"Base SOTA testnet preflight: {report['status'].upper()}")
    print(f"Manifest: {report['manifest']}")
    if report["env_file"]:
        print(f"Env file: {report['env_file']}")
    print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
    for check in report["checks"]:
        print(f"- [{check['status']}] {check['name']}: {check['detail']}")
        if check.get("remediation"):
            print(f"  next: {check['remediation']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Base SOTA Base Sepolia preflight.")
    parser.add_argument("manifest", type=Path, help="Base Sepolia deployment manifest JSON")
    parser.add_argument("--env-file", type=Path, help="Optional Base SOTA testnet env file")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP/RPC timeout in seconds")
    parser.add_argument("--offline", action="store_true", help="Skip network checks")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when red checks remain")
    args = parser.parse_args(argv)
    report = run_preflight(args.manifest, env_file=args.env_file, timeout=args.timeout, offline=args.offline)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

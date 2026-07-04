#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
DEFAULT_RPC_URL = "https://sepolia.base.org"
DEFAULT_AWS_PROFILE = "moonrocklab-frankfurt"
DEFAULT_READINESS_URL = "https://claims-test.bitsota.com/base-sota-testnet-readiness.json"
DEFAULT_DEPLOYER_SECRET_ID = "base-sota/test/base-sepolia/deployer"
DEFAULT_ROOT_PUBLISHER_SECRET_ID = "base-sota/test/base-sepolia/root-publisher"
DEFAULT_FUNDING_REPORT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-funding.json"
DEFAULT_BLOCKER_REPORT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-blockers.json"
DEFAULT_SERVICE_HOSTS = {
    "claims_ui": "claims-test.bitsota.com",
    "claims_api": "claims-api-test.bitsota.com",
    "coordinator": "coordinator-test.bitsota.com",
}
EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
EVM_ADDRESS_ANY_RE = re.compile(r"0x[0-9a-fA-F]{40}")


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


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(status, 2)


def _worst(checks: list[Check]) -> str:
    if not checks:
        return "green"
    return max((check.status for check in checks), key=_status_rank)


def _host_from_value(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return parsed.hostname or value


def _aws_identity_payload(timeout: float, profile: str = "") -> dict[str, Any]:
    cmd = ["aws", "sts", "get-caller-identity", "--output", "json"]
    if profile:
        cmd.extend(["--profile", profile])
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(stderr)
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("aws sts returned non-object JSON")
    return payload


def _aws_identity_check(*, timeout: float, skip: bool, profile: str = "") -> Check:
    if skip:
        return Check(
            "aws_identity",
            "yellow",
            "AWS identity check skipped.",
            "Run without --skip-aws before assigning public testnet service deployment.",
        )
    try:
        payload = _aws_identity_payload(timeout, profile)
    except FileNotFoundError:
        return Check(
            "aws_identity",
            "red",
            "aws CLI is not installed or not on PATH.",
            "Install/configure AWS CLI for the operator account that deploys public testnet services.",
        )
    except Exception as exc:
        return Check(
            "aws_identity",
            "red",
            f"AWS identity unavailable: {exc}",
            "Run aws configure/sso login or export the approved temporary credentials before public testnet deployment.",
        )
    account = str(payload.get("Account") or "")
    arn = str(payload.get("Arn") or "")
    if not account or not arn:
        return Check(
            "aws_identity",
            "red",
            "AWS STS response did not include Account and Arn.",
            "Fix AWS authentication before public testnet deployment.",
        )
    suffix = f" using profile {profile!r}" if profile else ""
    return Check("aws_identity", "green", f"Authenticated to AWS account {account} as {arn}{suffix}.")


def _run_aws(args: list[str], *, profile: str, timeout: float) -> dict[str, Any]:
    cmd = ["aws", *args, "--output", "json"]
    if profile:
        cmd.extend(["--profile", profile])
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(stderr)
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("aws returned non-object JSON")
    return payload


def _secret_tag(secret_id: str, tag_key: str, *, profile: str, timeout: float) -> str:
    payload = _run_aws(["secretsmanager", "describe-secret", "--secret-id", secret_id], profile=profile, timeout=timeout)
    for item in payload.get("Tags") or []:
        if isinstance(item, dict) and str(item.get("Key") or "") == tag_key and item.get("Value"):
            return str(item["Value"]).strip()
    return ""


def _resolve_host(host: str, *, timeout: float) -> list[str]:
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        infos = socket.getaddrinfo(host, None)
    finally:
        socket.setdefaulttimeout(previous_timeout)
    addresses = sorted({item[4][0] for item in infos if item and item[4]})
    return addresses


def _dns_check(name: str, value: str, *, timeout: float) -> Check:
    host = _host_from_value(value)
    try:
        addresses = _resolve_host(host, timeout=timeout)
    except OSError as exc:
        return Check(
            f"dns_{name}",
            "red",
            f"{host} does not resolve: {exc}",
            "Create the public DNS record and point it at the deployed testnet service.",
        )
    if not addresses:
        return Check(
            f"dns_{name}",
            "red",
            f"{host} resolved with no addresses.",
            "Create the public DNS record and point it at the deployed testnet service.",
        )
    return Check(f"dns_{name}", "green", f"{host} resolves to {', '.join(addresses[:4])}.")


def _json_rpc(rpc_url: str, method: str, params: list[Any] | None = None, *, timeout: float) -> Any:
    request = Request(
        rpc_url,
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "sota-base-testnet-blockers/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload.get("result")


def _native_balance_wei(rpc_url: str, address: str, *, timeout: float) -> int:
    raw = _json_rpc(rpc_url, "eth_getBalance", [address, "latest"], timeout=timeout)
    return int(str(raw), 16)


def _is_evm_address(value: str) -> bool:
    return bool(EVM_ADDRESS_RE.fullmatch(value))


def _fallback_reports(args: argparse.Namespace) -> list[Path]:
    configured = getattr(args, "fallback_report", None)
    if configured is None:
        return [DEFAULT_FUNDING_REPORT, DEFAULT_BLOCKER_REPORT]
    return [Path(item) for item in configured]


def _fallback_gas_targets(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in _fallback_reports(args):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for target in payload.get("funding_targets") or []:
            if not isinstance(target, dict):
                continue
            label = str(target.get("label") or "").strip()
            address = str(target.get("address") or "").strip()
            if label and _is_evm_address(address):
                out.setdefault(label, {"address": address, "source": str(path)})
        for check in payload.get("checks") or []:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name") or "")
            if not name.startswith("gas_"):
                continue
            label = name.removeprefix("gas_")
            text = f"{check.get('detail') or ''} {check.get('remediation') or ''}"
            match = EVM_ADDRESS_ANY_RE.search(text)
            if label and match:
                out.setdefault(label, {"address": match.group(0), "source": str(path)})
    return out


def _gas_address_check(label: str, address: str, *, rpc_url: str, timeout: float) -> Check:
    if not address:
        return Check(
            f"gas_{label}",
            "yellow",
            f"No public address configured for {label}.",
            "Record the public testnet address so gas readiness can be checked before deployment/browser smoke.",
        )
    if not _is_evm_address(address):
        return Check(
            f"gas_{label}",
            "red",
            f"{label} address is not a valid EVM address: {address!r}.",
            "Use a 20-byte Base Sepolia EVM address.",
        )
    try:
        balance_wei = _native_balance_wei(rpc_url, address, timeout=timeout)
    except Exception as exc:
        return Check(
            f"gas_{label}",
            "red",
            f"Could not read Base Sepolia ETH balance for {label} {address}: {exc}",
            "Fix the Base Sepolia RPC endpoint before deployment/browser smoke.",
        )
    if balance_wei <= 0:
        return Check(
            f"gas_{label}",
            "red",
            f"{label} {address} has 0 ETH on Base Sepolia.",
            f"Fund {address} with Base Sepolia ETH before deployment/browser smoke.",
        )
    return Check(f"gas_{label}", "green", f"{label} {address} has {balance_wei / 10**18:.8f} ETH on Base Sepolia.")


def _cached_gas_check(
    label: str,
    secret_id: str,
    reason: str,
    *,
    rpc_url: str,
    timeout: float,
    fallback_targets: dict[str, dict[str, str]],
) -> Check | None:
    fallback = fallback_targets.get(label)
    if not fallback:
        return None
    address = str(fallback.get("address") or "")
    source = str(fallback.get("source") or "cached public report")
    check = _gas_address_check(label, address, rpc_url=rpc_url, timeout=timeout)
    detail = (
        f"{check.detail} (from cached public report {source}; "
        f"could not read secret tag {secret_id!r}: {reason})."
    )
    return Check(check.name, check.status, detail, check.remediation)


def _gas_secret_check(
    label: str,
    secret_id: str,
    *,
    rpc_url: str,
    profile: str,
    timeout: float,
    fallback_targets: dict[str, dict[str, str]] | None = None,
) -> Check:
    fallback_targets = fallback_targets or {}
    if not secret_id:
        return Check(
            f"gas_{label}",
            "yellow",
            f"No secret handle configured for {label}.",
            "Pass the approved secret handle so its public sota-address tag can be checked without reading secret values.",
        )
    try:
        address = _secret_tag(secret_id, "sota-address", profile=profile, timeout=timeout)
    except FileNotFoundError:
        cached = _cached_gas_check(
            label,
            secret_id,
            "aws CLI is not installed or not on PATH",
            rpc_url=rpc_url,
            timeout=timeout,
            fallback_targets=fallback_targets,
        )
        if cached:
            return cached
        return Check(
            f"gas_{label}",
            "red",
            "aws CLI is not installed or not on PATH.",
            "Install/configure AWS CLI before checking signer gas readiness.",
        )
    except Exception as exc:
        cached = _cached_gas_check(
            label,
            secret_id,
            str(exc),
            rpc_url=rpc_url,
            timeout=timeout,
            fallback_targets=fallback_targets,
        )
        if cached:
            return cached
        return Check(
            f"gas_{label}",
            "red",
            f"Could not read public sota-address tag from {secret_id!r}: {exc}",
            "Add a public sota-address tag to the approved testnet secret handle or fix AWS access.",
        )
    if not address:
        cached = _cached_gas_check(
            label,
            secret_id,
            "missing sota-address tag",
            rpc_url=rpc_url,
            timeout=timeout,
            fallback_targets=fallback_targets,
        )
        if cached:
            return cached
        return Check(
            f"gas_{label}",
            "yellow",
            f"{secret_id!r} has no public sota-address tag.",
            "Add a public sota-address tag to the secret handle so the operator can verify gas without reading secret values.",
        )
    check = _gas_address_check(label, address, rpc_url=rpc_url, timeout=timeout)
    if check.status != "green":
        return Check(check.name, check.status, f"{check.detail} (from secret tag {secret_id!r}).", check.remediation)
    return Check(check.name, check.status, f"{check.detail} (from secret tag {secret_id!r}).")


def _rpc_chain_id(rpc_url: str, *, timeout: float) -> int:
    raw = _json_rpc(rpc_url, "eth_chainId", timeout=timeout)
    return int(str(raw), 16)


def _rpc_check(rpc_url: str, *, timeout: float) -> Check:
    try:
        chain_id = _rpc_chain_id(rpc_url, timeout=timeout)
    except Exception as exc:
        return Check(
            "base_sepolia_rpc",
            "red",
            f"Could not read chain id from {rpc_url}: {exc}",
            "Use a reachable Base Sepolia JSON-RPC endpoint before testnet rehearsal.",
        )
    if chain_id == BASE_MAINNET_CHAIN_ID:
        return Check("base_sepolia_rpc", "red", "RPC returned Base mainnet chain id 8453.", "Never use Base mainnet for this testnet gate.")
    if chain_id != BASE_SEPOLIA_CHAIN_ID:
        return Check(
            "base_sepolia_rpc",
            "red",
            f"RPC returned chain id {chain_id}, expected 84532.",
            "Point the config at Base Sepolia.",
        )
    return Check("base_sepolia_rpc", "green", f"RPC returned Base Sepolia chain id {chain_id}.")


def _json_file_check(name: str, path: Path, *, required_schema: str | None = None, require_ok: bool = False) -> Check:
    if not path.exists():
        return Check(
            f"artifact_{name}",
            "red",
            f"Missing {path}.",
            "Run the Base Sepolia rehearsal/deployment path and preserve the generated artifact.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Check(
            f"artifact_{name}",
            "red",
            f"{path} is not valid JSON: {exc}",
            "Regenerate the artifact from the guarded testnet tooling.",
        )
    if not isinstance(payload, dict):
        return Check(
            f"artifact_{name}",
            "red",
            f"{path} does not contain a JSON object.",
            "Regenerate the artifact from the guarded testnet tooling.",
        )
    if required_schema and payload.get("schema") != required_schema:
        return Check(
            f"artifact_{name}",
            "red",
            f"{path} schema is {payload.get('schema')!r}, expected {required_schema!r}.",
            "Publish the readiness artifact generated by the Base Sepolia rehearsal.",
        )
    if require_ok and not payload.get("ok"):
        status = payload.get("status") or "unknown"
        return Check(
            f"artifact_{name}",
            "red",
            f"{path} exists but reports status {status!r}, ok=false.",
            "Clear the red preflight checks, regenerate the readiness artifact, and republish it.",
        )
    return Check(f"artifact_{name}", "green", f"{path} exists and is parseable.")


def _service_pack_check(path: Path) -> Check:
    if not path.exists():
        return Check(
            "artifact_service_pack",
            "red",
            f"Missing {path}.",
            "Generate the Base Sepolia service pack so infrastructure agents have service commands, DNS hosts, secret handles, and health URLs.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Check(
            "artifact_service_pack",
            "red",
            f"{path} is not valid JSON: {exc}",
            "Regenerate the service pack from scripts/sota_base_testnet_service_pack.py.",
        )
    if not isinstance(payload, dict):
        return Check(
            "artifact_service_pack",
            "red",
            f"{path} does not contain a JSON object.",
            "Regenerate the service pack from scripts/sota_base_testnet_service_pack.py.",
        )
    if payload.get("schema") != "sota-base-testnet-service-pack/v1":
        return Check(
            "artifact_service_pack",
            "red",
            f"{path} schema is {payload.get('schema')!r}, expected 'sota-base-testnet-service-pack/v1'.",
            "Regenerate the service pack from scripts/sota_base_testnet_service_pack.py.",
        )
    if not payload.get("deployment_ready"):
        status = str(payload.get("status") or "unknown")
        actions = [str(item) for item in payload.get("next_actions") or [] if str(item)]
        remediation = actions[0] if actions else "Clear service pack yellow/red checks before public testnet deployment."
        return Check(
            "artifact_service_pack",
            "red",
            f"{path} exists but deployment_ready=false, status={status!r}.",
            remediation,
        )
    return Check("artifact_service_pack", "green", f"{path} exists and is deployment-ready.")


def _container_pack_check(path: Path) -> Check:
    if not path.exists():
        return Check(
            "artifact_container_pack",
            "red",
            f"Missing {path}.",
            "Generate the container/App Runner image pack before creating public Base SOTA testnet services.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Check(
            "artifact_container_pack",
            "red",
            f"{path} is not valid JSON: {exc}",
            "Regenerate the container pack from scripts/sota_base_testnet_container_pack.py.",
        )
    if not isinstance(payload, dict):
        return Check(
            "artifact_container_pack",
            "red",
            f"{path} does not contain a JSON object.",
            "Regenerate the container pack from scripts/sota_base_testnet_container_pack.py.",
        )
    if payload.get("schema") != "sota-base-testnet-container-pack/v1":
        return Check(
            "artifact_container_pack",
            "red",
            f"{path} schema is {payload.get('schema')!r}, expected 'sota-base-testnet-container-pack/v1'.",
            "Regenerate the container pack from scripts/sota_base_testnet_container_pack.py.",
        )
    status = str(payload.get("status") or "unknown")
    if payload.get("ok"):
        return Check("artifact_container_pack", "green", f"{path} exists and is deployment-ready.")
    actions = [str(item) for item in payload.get("next_actions") or [] if str(item)]
    remediation = actions[0] if actions else "Clear container/App Runner pack checks before public service deployment."
    return Check(
        "artifact_container_pack",
        "yellow" if status == "yellow" else "red",
        f"{path} exists but is not deployment-ready, status={status!r}.",
        remediation,
    )


def _apprunner_source_pack_check(path: Path) -> Check:
    if not path.exists():
        return Check(
            "artifact_apprunner_source_pack",
            "red",
            f"Missing {path}.",
            "Generate the source-based App Runner pack before creating public Base SOTA testnet services.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Check(
            "artifact_apprunner_source_pack",
            "red",
            f"{path} is not valid JSON: {exc}",
            "Regenerate the source App Runner pack from scripts/sota_base_testnet_apprunner_source_pack.py.",
        )
    if not isinstance(payload, dict):
        return Check(
            "artifact_apprunner_source_pack",
            "red",
            f"{path} does not contain a JSON object.",
            "Regenerate the source App Runner pack from scripts/sota_base_testnet_apprunner_source_pack.py.",
        )
    if payload.get("schema") != "sota-base-testnet-apprunner-source-pack/v1":
        return Check(
            "artifact_apprunner_source_pack",
            "red",
            f"{path} schema is {payload.get('schema')!r}, expected 'sota-base-testnet-apprunner-source-pack/v1'.",
            "Regenerate the source App Runner pack from scripts/sota_base_testnet_apprunner_source_pack.py.",
        )
    status = str(payload.get("status") or "unknown")
    if payload.get("ok"):
        return Check("artifact_apprunner_source_pack", "green", f"{path} exists and is deployment-ready.")
    actions = [str(item) for item in payload.get("next_actions") or [] if str(item)]
    remediation = actions[0] if actions else "Clear source App Runner pack checks before public service deployment."
    return Check(
        "artifact_apprunner_source_pack",
        "yellow" if status == "yellow" else "red",
        f"{path} exists but is not deployment-ready, status={status!r}.",
        remediation,
    )


def _env_file_check(path: Path) -> Check:
    if not path.exists():
        return Check(
            "artifact_env",
            "red",
            f"Missing {path}.",
            "Generate base-sota.env.testnet from the Base Sepolia deployment manifest before building the claims UI.",
        )
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    chain_id = values.get("NEXT_PUBLIC_SOTA_BASE_CHAIN_ID") or values.get("BASE_CHAIN_ID")
    if chain_id == str(BASE_MAINNET_CHAIN_ID):
        return Check("artifact_env", "red", f"{path} points at Base mainnet chain id 8453.", "Regenerate the env file for Base Sepolia only.")
    if chain_id != str(BASE_SEPOLIA_CHAIN_ID):
        return Check("artifact_env", "red", f"{path} chain id is {chain_id!r}, expected 84532.", "Regenerate the env file for Base Sepolia.")
    return Check("artifact_env", "green", f"{path} exists and is pinned to Base Sepolia.")


def _http_status(url: str, *, timeout: float) -> tuple[int | None, str]:
    request = Request(url, headers={"Accept": "application/json,text/html;q=0.9,*/*;q=0.8"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), "ok"
    except HTTPError as exc:
        return int(exc.code), str(exc)
    except (URLError, TimeoutError, OSError) as exc:
        return None, str(exc)


def _readiness_url_check(url: str, *, timeout: float, skip: bool) -> Check:
    if skip:
        return Check(
            "public_readiness_url",
            "yellow",
            "Public readiness URL check skipped.",
            "Run without --skip-readiness-url before asking a browser tester to open claims.",
        )
    status, detail = _http_status(url, timeout=timeout)
    if status is None:
        return Check(
            "public_readiness_url",
            "red",
            f"{url} did not respond: {detail}",
            "Publish base-sota-testnet-readiness.json at the URL consumed by the claims website.",
        )
    if 200 <= status < 300:
        return Check("public_readiness_url", "green", f"{url} responded with HTTP {status}.")
    return Check(
        "public_readiness_url",
        "red",
        f"{url} responded with HTTP {status}.",
        "Publish a readable readiness artifact before public browser-wallet smoke.",
    )


def _next_actions(checks: list[Check]) -> list[str]:
    actions: list[str] = []
    for check in checks:
        if check.status == "green" or not check.remediation:
            continue
        if check.remediation not in actions:
            actions.append(check.remediation)
    return actions


def run_blocker_report(args: argparse.Namespace) -> dict[str, Any]:
    artifacts_dir = args.artifacts_dir
    hosts = dict(DEFAULT_SERVICE_HOSTS)
    for item in args.host:
        if "=" not in item:
            raise SystemExit(f"--host must be name=host_or_url, got {item!r}")
        key, value = item.split("=", 1)
        hosts[key.strip()] = value.strip()

    checks: list[Check] = []
    checks.append(_aws_identity_check(timeout=args.timeout, skip=args.skip_aws, profile=args.aws_profile))
    checks.append(_rpc_check(args.rpc_url, timeout=args.timeout))
    if not args.skip_gas:
        fallback_targets = _fallback_gas_targets(args)
        for label, secret_id in (
            ("deployer", args.deployer_secret_id),
            ("root_publisher", args.root_publisher_secret_id),
        ):
            checks.append(
                _gas_secret_check(
                    label,
                    secret_id,
                    rpc_url=args.rpc_url,
                    profile=args.aws_profile,
                    timeout=args.timeout,
                    fallback_targets=fallback_targets,
                )
            )
        for item in args.gas_address:
            if "=" not in item:
                raise SystemExit(f"--gas-address must be label=0xAddress, got {item!r}")
            label, address = item.split("=", 1)
            checks.append(
                _gas_address_check(
                    label.strip(),
                    address.strip(),
                    rpc_url=args.rpc_url,
                    timeout=args.timeout,
                )
            )
    for name, host in hosts.items():
        checks.append(_dns_check(name, host, timeout=args.timeout))
    checks.append(_readiness_url_check(args.readiness_url, timeout=args.timeout, skip=args.skip_readiness_url))
    checks.append(_json_file_check("compact_deployment", args.deployment))
    checks.append(_json_file_check("manifest", args.manifest))
    checks.append(_env_file_check(args.env_file))
    checks.append(_service_pack_check(args.service_pack))
    checks.append(_apprunner_source_pack_check(args.apprunner_source_pack))
    checks.append(
        _json_file_check(
            "readiness",
            args.readiness_file,
            required_schema="sota-base-testnet-readiness/v1",
            require_ok=True,
        )
    )

    status = _worst(checks)
    return {
        "schema": "sota-base-testnet-blockers/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": status == "green",
        "status": status,
        "message": (
            "Base Sepolia is clear for guarded rehearsal/browser smoke."
            if status == "green"
            else "Base Sepolia is still blocked for nontechnical browser-wallet testing."
        ),
        "read_only": True,
        "does_not": ["deploy", "sign", "broadcast_transactions", "touch_production_bittensor"],
        "rpc_url": args.rpc_url,
        "artifacts_dir": str(artifacts_dir),
        "checks": [check.as_dict() for check in checks],
        "summary": {
            "green": sum(1 for check in checks if check.status == "green"),
            "yellow": sum(1 for check in checks if check.status == "yellow"),
            "red": sum(1 for check in checks if check.status == "red"),
        },
        "next_actions": _next_actions(checks),
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"Base SOTA testnet blockers: {report['status'].upper()}")
    print(report["message"])
    print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
    for check in report["checks"]:
        print(f"- [{check['status']}] {check['name']}: {check['detail']}")
        if check.get("remediation"):
            print(f"  next: {check['remediation']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only blocker report for Base SOTA Base Sepolia readiness.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--readiness-file", type=Path)
    parser.add_argument("--service-pack", type=Path)
    parser.add_argument("--container-pack", type=Path)
    parser.add_argument("--apprunner-source-pack", type=Path)
    parser.add_argument("--readiness-url", default=DEFAULT_READINESS_URL)
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL)
    parser.add_argument("--host", action="append", default=[], help="Override/add DNS check as name=host_or_url. Repeatable.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--skip-aws", action="store_true", help="Skip AWS STS identity check.")
    parser.add_argument("--aws-profile", default=DEFAULT_AWS_PROFILE, help="Optional AWS CLI profile for the STS identity check.")
    parser.add_argument("--skip-readiness-url", action="store_true", help="Skip public readiness URL HTTP check.")
    parser.add_argument("--deployer-secret-id", default=DEFAULT_DEPLOYER_SECRET_ID, help="Approved testnet deployer secret handle; only the public sota-address tag is read.")
    parser.add_argument("--root-publisher-secret-id", default=DEFAULT_ROOT_PUBLISHER_SECRET_ID, help="Approved testnet root-publisher secret handle; only the public sota-address tag is read.")
    parser.add_argument("--gas-address", action="append", default=[], help="Check a public Base Sepolia gas balance as label=0xAddress. Repeatable.")
    parser.add_argument(
        "--fallback-report",
        action="append",
        type=Path,
        default=None,
        help="Existing public funding/blocker report to reuse only for cached public addresses when AWS tags are unavailable. Repeatable.",
    )
    parser.add_argument("--skip-gas", action="store_true", help="Skip signer/test-wallet gas balance checks.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when red checks remain.")
    args = parser.parse_args(argv)

    args.deployment = args.deployment or args.artifacts_dir / "base-sepolia-compact-deployment.json"
    args.manifest = args.manifest or args.artifacts_dir / "base-sepolia-deployment-manifest.json"
    args.env_file = args.env_file or args.artifacts_dir / "base-sota.env.testnet"
    args.readiness_file = args.readiness_file or args.artifacts_dir / "base-sota-testnet-readiness.json"
    args.service_pack = args.service_pack or args.artifacts_dir / "base-sota-testnet-service-pack.json"
    args.container_pack = args.container_pack or args.artifacts_dir / "base-sota-testnet-container-pack.json"
    args.apprunner_source_pack = args.apprunner_source_pack or args.artifacts_dir / "base-sota-testnet-apprunner-source-pack.json"

    report = run_blocker_report(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text, encoding="utf-8")
    if args.json:
        print(text, end="")
    else:
        _print_text(report)
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

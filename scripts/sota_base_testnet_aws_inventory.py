#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.parse import urlparse


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_REGION = "eu-central-1"
DEFAULT_REQUIRED_SECRET_NAMES = (
    "base-sota/test/base-sepolia/rpc-url",
    "base-sota/test/base-sepolia/deployer",
    "base-sota/test/base-sepolia/root-publisher",
    "base-sota/test/base-sepolia/indexer-admin-token",
    "base-sota/test/base-sepolia/autoresearch-database-url",
    "base-sota/test/base-sepolia/autoresearch-admin-token",
)
BASE_SERVICE_PATTERNS = {
    "claims_ui": re.compile(r"(base[-_]?sota|sota[-_]?base).*(claims|website|ui)|claims[-_]?test", re.I),
    "claims_api": re.compile(r"(base[-_]?sota|sota[-_]?base).*(claims|indexer|api)|claims[-_]?api[-_]?test", re.I),
    "coordinator": re.compile(r"(base[-_]?sota|sota[-_]?base).*coordinator|coordinator[-_]?test", re.I),
    "root_publisher": re.compile(r"(base[-_]?sota|sota[-_]?base).*root|root[-_]?publisher[-_]?test", re.I),
}
REQUIRED_PUBLIC_SERVICE_KEYS = tuple(BASE_SERVICE_PATTERNS)
BASE_SECRET_RE = re.compile(r"(base[-_/]?sepolia|base[-_/]?sota|sota[-_/]?base)", re.I)


def _host(value: str) -> str:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return (parsed.hostname or value).strip().lower()


def _is_bitsota_host(value: str) -> bool:
    host = _host(value)
    return host == "bitsota.com" or host.endswith(".bitsota.com")


def _has_direct_service_url_plan(service_urls: dict[str, str]) -> bool:
    configured = {key: str(service_urls.get(key) or "").strip() for key in REQUIRED_PUBLIC_SERVICE_KEYS}
    return all(configured.values()) and not any(_is_bitsota_host(value) for value in configured.values())


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    remediation: str = ""

    def as_dict(self) -> dict[str, str]:
        payload = {"name": self.name, "status": self.status, "detail": self.detail}
        if self.remediation:
            payload["remediation"] = self.remediation
        return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_aws(args: list[str], *, profile: str, region: str, timeout: float) -> dict[str, Any]:
    cmd = ["aws", *args, "--output", "json"]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"aws exited {result.returncode}")
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("aws returned non-object JSON")
    return payload


def _safe_call(args: list[str], *, profile: str, region: str, timeout: float) -> tuple[dict[str, Any], str | None]:
    try:
        return _run_aws(args, profile=profile, region=region, timeout=timeout), None
    except Exception as exc:
        return {}, str(exc)


def _summary(checks: list[Check]) -> dict[str, int]:
    return {
        "green": sum(1 for check in checks if check.status == "green"),
        "yellow": sum(1 for check in checks if check.status == "yellow"),
        "red": sum(1 for check in checks if check.status == "red"),
    }


def _worst(checks: list[Check]) -> str:
    rank = {"green": 0, "yellow": 1, "red": 2}
    return max((check.status for check in checks), key=lambda status: rank.get(status, 2), default="green")


def _service_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in payload.get("ServiceSummaryList") or []:
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "name": str(raw.get("ServiceName") or ""),
                "arn": str(raw.get("ServiceArn") or ""),
                "url": str(raw.get("ServiceUrl") or ""),
                "status": str(raw.get("Status") or ""),
            }
        )
    return rows


def _hosted_zone_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in payload.get("HostedZones") or []:
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "id": str(raw.get("Id") or ""),
                "name": str(raw.get("Name") or ""),
                "private": bool(dict(raw.get("Config") or {}).get("PrivateZone")),
                "record_count": int(raw.get("ResourceRecordSetCount") or 0),
            }
        )
    return rows


def _repo_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in payload.get("repositories") or []:
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "name": str(raw.get("repositoryName") or ""),
                "uri": str(raw.get("repositoryUri") or ""),
                "scan_on_push": bool(dict(raw.get("imageScanningConfiguration") or {}).get("scanOnPush")),
            }
        )
    return rows


def _secret_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in payload.get("SecretList") or []:
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "name": str(raw.get("Name") or ""),
                "arn": str(raw.get("ARN") or ""),
                "description": str(raw.get("Description") or ""),
            }
        )
    return rows


def _connection_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in payload.get("ConnectionSummaryList") or []:
        if not isinstance(raw, dict):
            continue
        rows.append(
            {
                "name": str(raw.get("ConnectionName") or ""),
                "arn": str(raw.get("ConnectionArn") or ""),
                "status": str(raw.get("Status") or ""),
                "provider_type": str(raw.get("ProviderType") or ""),
            }
        )
    return rows


def _inventory_checks(
    *,
    identity_error: str | None,
    identity: dict[str, Any],
    services: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    repos: list[dict[str, Any]],
    secrets: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    service_urls: dict[str, str],
    required_secret_names: list[str],
    external_dns_owner: str,
    collection_errors: dict[str, str | None],
) -> list[Check]:
    checks: list[Check] = []
    if identity_error:
        checks.append(
            Check(
                "aws_identity",
                "red",
                f"AWS identity unavailable: {identity_error}",
                "Authenticate with the approved testnet AWS profile before deployment.",
            )
        )
    else:
        checks.append(
            Check(
                "aws_identity",
                "green",
                f"Authenticated to AWS account {identity.get('Account')} as {identity.get('Arn')}.",
            )
        )

    collection_remediation = "Authenticate with the approved testnet AWS profile and verify read-only inventory permissions."

    bitsota_zone = next((zone for zone in zones if str(zone["name"]).lower() == "bitsota.com."), None)
    direct_service_url_plan = _has_direct_service_url_plan(service_urls)
    zones_error = collection_errors.get("zones")
    if zones_error and not external_dns_owner and not direct_service_url_plan:
        route53_status = "red"
        route53_detail = f"Route53 hosted zone inventory unavailable: {zones_error}"
        route53_remediation = collection_remediation
    elif bitsota_zone:
        route53_status = "green"
        route53_detail = f"Route53 hosted zone found for bitsota.com: {bitsota_zone['id']}."
        route53_remediation = ""
    elif external_dns_owner:
        route53_status = "green"
        route53_detail = f"No Route53 hosted zone for bitsota.com was found in this AWS account; external DNS owner documented: {external_dns_owner}."
        route53_remediation = ""
    elif direct_service_url_plan:
        route53_status = "green"
        route53_detail = "No Route53 hosted zone for bitsota.com was found in this AWS account; direct public service URLs are configured, so custom DNS is not required for this testnet run."
        route53_remediation = ""
    else:
        route53_status = "red"
        route53_detail = "No Route53 hosted zone for bitsota.com was found in this AWS account."
        route53_remediation = "Create DNS records wherever bitsota.com is hosted, document the external DNS owner, or pass direct public service URLs for testnet domains."
    checks.append(
        Check(
            "route53_bitsota_zone",
            route53_status,
            route53_detail,
            route53_remediation,
        )
    )

    services_error = collection_errors.get("services")
    if services_error:
        checks.append(
            Check(
                "base_sota_apprunner_services",
                "red",
                f"App Runner service inventory unavailable: {services_error}",
                collection_remediation,
            )
        )
    else:
        service_names = [str(service["name"]) for service in services]
        service_hosts = {_host(str(service.get("url") or "")): str(service.get("name") or "") for service in services if service.get("url")}
        configured_service_matches = {
            key: service_hosts[_host(value)]
            for key, value in service_urls.items()
            if value and _host(value) in service_hosts
        }
        missing_service_keys = [
            key
            for key, pattern in BASE_SERVICE_PATTERNS.items()
            if key not in configured_service_matches and not any(pattern.search(name) for name in service_names)
        ]
        service_detail = (
            "Base SOTA-specific or explicitly configured App Runner service URLs are present: "
            + ", ".join(f"{key}={value}" for key, value in sorted(configured_service_matches.items()))
            if configured_service_matches
            else "Base SOTA-specific App Runner service names are present."
        )
        checks.append(
            Check(
                "base_sota_apprunner_services",
                "green" if not missing_service_keys else "red",
                service_detail
                if not missing_service_keys
                else f"Missing Base SOTA-specific App Runner services for: {', '.join(missing_service_keys)}. Existing services: {', '.join(service_names) or 'none'}.",
                "" if not missing_service_keys else "Create, configure, or pass direct public App Runner URLs for Base SOTA testnet services before public browser testing.",
            )
        )

    connections_error = collection_errors.get("connections")
    if connections_error:
        checks.append(
            Check(
                "apprunner_github_connection",
                "red",
                f"App Runner GitHub connection inventory unavailable: {connections_error}",
                collection_remediation,
            )
        )
    else:
        github_connections = [
            connection
            for connection in connections
            if str(connection.get("provider_type")).upper() == "GITHUB"
            and str(connection.get("status")).upper() == "AVAILABLE"
        ]
        checks.append(
            Check(
                "apprunner_github_connection",
                "green" if github_connections else "red",
                f"Available App Runner GitHub connections found: {', '.join(connection['name'] for connection in github_connections)}."
                if github_connections
                else "No AVAILABLE App Runner GitHub connection was found.",
                "" if github_connections else "Create or approve an App Runner GitHub connection before source-based service deployment.",
            )
        )

    repos_error = collection_errors.get("repos")
    if repos_error:
        checks.append(
            Check(
                "base_sota_ecr_repos",
                "yellow",
                f"ECR repository inventory unavailable: {repos_error}",
                "Fix AWS inventory access if the deployment plan uses ECR images; source-based App Runner deploys do not require this.",
            )
        )
    else:
        base_repos = [repo for repo in repos if re.search(r"(base[-_]?sota|sota[-_]?base|claims)", str(repo["name"]), re.I)]
        checks.append(
            Check(
                "base_sota_ecr_repos",
                "green" if base_repos else "yellow",
                f"Base SOTA/claims ECR repositories found: {', '.join(repo['name'] for repo in base_repos)}."
                if base_repos
                else "No Base SOTA/claims ECR repository found. This is acceptable only if App Runner deploys from source.",
                "" if base_repos else "Create ECR repositories only if the service deployment plan uses container images.",
            )
        )

    existing_secret_names = {str(secret["name"]) for secret in secrets}
    required_secret_names = [name for name in required_secret_names if name]
    missing_required_secrets = sorted(set(required_secret_names) - existing_secret_names)
    base_secrets = [secret for secret in secrets if BASE_SECRET_RE.search(str(secret["name"]))]
    secrets_error = collection_errors.get("secrets")
    if secrets_error:
        checks.append(
            Check(
                "base_sepolia_secret_handles",
                "red",
                f"Secrets Manager handle inventory unavailable: {secrets_error}",
                collection_remediation,
            )
        )
        return checks
    if required_secret_names:
        secret_ok = not missing_required_secrets
        secret_detail = (
            "All required Base Sepolia/Base SOTA secret handles exist: "
            + ", ".join(required_secret_names)
            if secret_ok
            else "Missing required Base Sepolia/Base SOTA secret handles: "
            + ", ".join(missing_required_secrets)
        )
    else:
        secret_ok = bool(base_secrets)
        secret_detail = (
            f"Base Sepolia/Base SOTA secret handles found: {', '.join(secret['name'] for secret in base_secrets)}."
            if base_secrets
            else "No Base Sepolia/Base SOTA secret handles were found by name. Existing secret names were inventoried without values."
        )
    checks.append(
        Check(
            "base_sepolia_secret_handles",
            "green" if secret_ok else "red",
            secret_detail,
            "" if secret_ok else "Create approved testnet secret handles for RPC, deployer, root publisher, indexer admin, and coordinator DB/admin.",
        )
    )
    return checks


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    service_urls: dict[str, str] = {}
    for item in args.service_url:
        if "=" not in item:
            raise SystemExit(f"--service-url must be name=url, got {item!r}")
        key, value = item.split("=", 1)
        service_urls[key.strip()] = value.strip()
    required_secret_names = [str(item).strip() for item in args.required_secret if str(item).strip()]
    external_dns_owner = str(getattr(args, "external_dns_owner", "") or "").strip()

    identity, identity_error = _safe_call(["sts", "get-caller-identity"], profile=args.aws_profile, region="", timeout=args.timeout)
    services_payload, services_error = _safe_call(["apprunner", "list-services"], profile=args.aws_profile, region=args.region, timeout=args.timeout)
    zones_payload, zones_error = _safe_call(["route53", "list-hosted-zones"], profile=args.aws_profile, region="", timeout=args.timeout)
    repos_payload, repos_error = _safe_call(["ecr", "describe-repositories"], profile=args.aws_profile, region=args.region, timeout=args.timeout)
    secrets_payload, secrets_error = _safe_call(["secretsmanager", "list-secrets", "--max-results", "100"], profile=args.aws_profile, region=args.region, timeout=args.timeout)
    connections_payload, connections_error = _safe_call(["apprunner", "list-connections"], profile=args.aws_profile, region=args.region, timeout=args.timeout)

    services = _service_rows(services_payload)
    zones = _hosted_zone_rows(zones_payload)
    repos = _repo_rows(repos_payload)
    secrets = _secret_rows(secrets_payload)
    connections = _connection_rows(connections_payload)
    checks = _inventory_checks(
        identity_error=identity_error,
        identity=identity,
        services=services,
        zones=zones,
        repos=repos,
        secrets=secrets,
        connections=connections,
        service_urls=service_urls,
        required_secret_names=required_secret_names,
        external_dns_owner=external_dns_owner,
        collection_errors={
            "services": services_error,
            "zones": zones_error,
            "repos": repos_error,
            "secrets": secrets_error,
            "connections": connections_error,
        },
    )
    for name, error in (
        ("apprunner_list_services", services_error),
        ("route53_list_hosted_zones", zones_error),
        ("ecr_describe_repositories", repos_error),
        ("secretsmanager_list_secrets", secrets_error),
        ("apprunner_list_connections", connections_error),
    ):
        if error:
            checks.append(Check(name, "red", error, "Fix AWS permissions for read-only testnet inventory."))
    status = _worst(checks)
    return {
        "schema": "sota-base-testnet-aws-inventory/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": status == "green",
        "status": status,
        "read_only": True,
        "does_not": ["create", "update", "delete", "deploy", "read_secret_values", "touch_production_bittensor", "touch_base_mainnet"],
        "aws": {
            "profile": args.aws_profile,
            "region": args.region,
            "account": identity.get("Account"),
            "arn": identity.get("Arn"),
        },
        "inventory": {
            "app_runner_services": services,
            "hosted_zones": zones,
            "ecr_repositories": repos,
            "secret_handles": secrets,
            "required_secret_handles": required_secret_names,
            "app_runner_connections": connections,
            "configured_service_urls": service_urls,
            "external_dns_owner": external_dns_owner,
        },
        "checks": [check.as_dict() for check in checks],
        "summary": _summary(checks),
        "next_actions": [check.remediation for check in checks if check.status != "green" and check.remediation],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only AWS inventory for Base SOTA Base Sepolia testnet readiness.")
    parser.add_argument("--aws-profile", default="")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--service-url", action="append", default=[], help="Configured public service URL as name=url. Repeatable.")
    parser.add_argument("--external-dns-owner", default="", help="Optional note naming the system or owner that manages bitsota.com DNS outside this AWS account.")
    parser.add_argument("--required-secret", action="append", default=list(DEFAULT_REQUIRED_SECRET_NAMES), help="Required AWS Secrets Manager secret name. Repeatable.")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-aws-inventory.json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)

    report = build_inventory(args)
    _write_json(args.out, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Base SOTA AWS inventory: {report['status'].upper()}")
        print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
        print(f"Report: {args.out}")
        for action in report["next_actions"][:8]:
            print(f"- next: {action}")
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

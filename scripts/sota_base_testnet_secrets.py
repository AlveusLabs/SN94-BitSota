#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
from typing import Any

from eth_account import Account
from sqlalchemy.engine import make_url


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_REGION = "eu-central-1"
DEFAULT_PROFILE = "moonrocklab-frankfurt"
DEFAULT_RPC_URL = "https://sepolia.base.org"
SECRET_PREFIX = "base-sota/test/base-sepolia/"
DEFAULT_SOURCE_AUTORESEARCH_DB_SECRET_ID = "bitsota/test/db"
DEFAULT_AUTORESEARCH_DATABASE_NAME = "base_sota_testnet_autoresearch"
SAFE_TEST_DATABASE_RE = re.compile(r"^base_sota_testnet_[a-z0-9_]+$")


@dataclass(frozen=True)
class SecretSpec:
    name: str
    purpose: str
    kind: str
    json_key: str
    managed: bool
    required: bool = True


SECRET_SPECS = (
    SecretSpec(f"{SECRET_PREFIX}rpc-url", "Base Sepolia RPC URL", "rpc_url", "rpc_url", True),
    SecretSpec(f"{SECRET_PREFIX}deployer", "Base Sepolia contract deployer", "evm_private_key", "private_key", True),
    SecretSpec(
        f"{SECRET_PREFIX}root-publisher",
        "Base Sepolia root publisher",
        "evm_private_key",
        "root_publisher_private_key",
        True,
    ),
    SecretSpec(f"{SECRET_PREFIX}indexer-admin-token", "Claims indexer admin token", "admin_token", "admin_token", True),
    SecretSpec(
        f"{SECRET_PREFIX}autoresearch-admin-token",
        "Autoresearch coordinator admin token",
        "admin_token",
        "admin_token",
        True,
    ),
    SecretSpec(
        f"{SECRET_PREFIX}autoresearch-database-url",
        "Autoresearch database URL",
        "external_secret",
        "database_url",
        False,
    ),
    SecretSpec(f"{SECRET_PREFIX}monitoring-api-key", "Monitoring provider API key", "external_secret", "api_key", False, False),
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_aws_json(args: list[str], *, profile: str, region: str, timeout: float) -> dict[str, Any]:
    cmd = ["aws", *args, "--output", "json"]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"aws exited {result.returncode}")
    if not result.stdout.strip():
        return {}
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("aws returned non-object JSON")
    return payload


def _aws_secret_string(secret_id: str, *, profile: str, region: str, timeout: float) -> str:
    payload = _run_aws_json(
        ["secretsmanager", "get-secret-value", "--secret-id", secret_id],
        profile=profile,
        region=region,
        timeout=timeout,
    )
    value = str(payload.get("SecretString") or "").strip()
    if not value:
        raise RuntimeError(f"secret {secret_id!r} has no SecretString")
    return value


def _safe_describe_secret(name: str, *, profile: str, region: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return (
            _run_aws_json(
                ["secretsmanager", "describe-secret", "--secret-id", name],
                profile=profile,
                region=region,
                timeout=timeout,
            ),
            None,
        )
    except Exception as exc:
        text = str(exc)
        if "ResourceNotFoundException" in text or "can't find the specified secret" in text:
            return None, None
        return None, text


def _address_from_tags(secret: dict[str, Any] | None) -> str | None:
    for tag in (secret or {}).get("Tags") or []:
        if not isinstance(tag, dict):
            continue
        if str(tag.get("Key") or "") == "sota-address" and tag.get("Value"):
            return str(tag["Value"])
    return None


def _secret_payload(spec: SecretSpec, *, rpc_url: str) -> tuple[dict[str, Any], str | None]:
    base = {
        "environment": "base-sepolia",
        "purpose": spec.purpose,
        "created_by": "sota_base_testnet_secrets.py",
    }
    if spec.kind == "rpc_url":
        return {**base, spec.json_key: rpc_url}, None
    if spec.kind == "evm_private_key":
        account = Account.create()
        private_key = account.key.hex()
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key
        return {**base, spec.json_key: private_key, "address": account.address}, account.address
    if spec.kind == "admin_token":
        return {**base, spec.json_key: secrets.token_urlsafe(32)}, None
    raise ValueError(f"{spec.name} is not managed by this bootstrapper")


def _safe_secret_name(name: str) -> None:
    if not name.startswith(SECRET_PREFIX):
        raise SystemExit(f"refusing non-testnet secret name: {name}")


def _database_url_from_secret_string(secret_string: str) -> str:
    text = secret_string.strip()
    if not text:
        raise ValueError("source database secret is empty")
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("JSON database secret must be an object")
        for key in ("DATABASE_URL", "database_url", "url", "uri"):
            if payload.get(key):
                return str(payload[key]).strip()
        raise ValueError("JSON database secret did not include a database URL field")
    return text


def _guard_source_database_secret(secret_id: str, database_name: str) -> None:
    lowered = secret_id.lower()
    if "/prod/" in lowered or "prod" in lowered:
        raise SystemExit(f"refusing production-looking source database secret: {secret_id}")
    if "/test/" not in lowered and "test" not in lowered:
        raise SystemExit(f"source database secret must be explicitly test-scoped: {secret_id}")
    if not SAFE_TEST_DATABASE_RE.fullmatch(database_name):
        raise SystemExit(f"refusing unsafe test database name: {database_name}")


def _redact_secret_text(value: object) -> str:
    text = str(value)
    text = re.sub(r"postgres(?:ql)?://[^\s\"']+", "postgresql://<redacted>", text, flags=re.IGNORECASE)
    text = re.sub(r"(password=)[^\s;]+", r"\1<redacted>", text, flags=re.IGNORECASE)
    text = re.sub(r"([\"']?(?:password|database_url|DATABASE_URL)[\"']?\s*[:=]\s*[\"']?)[^\"'\s,}]+", r"\1<redacted>", text)
    return text


def _ensure_postgres_database(source_database_url: str, database_name: str) -> dict[str, Any]:
    import psycopg2
    from psycopg2 import sql

    source_url = make_url(source_database_url)
    if not source_url.drivername.startswith("postgresql"):
        raise ValueError(f"source database driver must be postgresql, got {source_url.drivername}")
    connection = psycopg2.connect(source_url.render_as_string(hide_password=False))
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
            exists = cursor.fetchone() is not None
            if not exists:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
    finally:
        connection.close()
    target_url = source_url.set(database=database_name)
    return {
        "database": database_name,
        "host": source_url.host,
        "created_database": not exists,
        "database_url": target_url.render_as_string(hide_password=False),
    }


def _create_secret(
    spec: SecretSpec,
    payload: dict[str, Any],
    *,
    address: str | None,
    profile: str,
    region: str,
    timeout: float,
) -> dict[str, Any]:
    _safe_secret_name(spec.name)
    tags = [
        {"Key": "project", "Value": "base-sota"},
        {"Key": "environment", "Value": "base-sepolia"},
        {"Key": "purpose", "Value": spec.purpose},
    ]
    if address:
        tags.append({"Key": "sota-address", "Value": address})
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(json.dumps(payload, sort_keys=True))
        temp_path.chmod(0o600)
        created = _run_aws_json(
            [
                "secretsmanager",
                "create-secret",
                "--name",
                spec.name,
                "--description",
                f"SOTA Base Sepolia testnet: {spec.purpose}",
                "--secret-string",
                f"file://{temp_path}",
                "--tags",
                json.dumps(tags, separators=(",", ":")),
            ],
            profile=profile,
            region=region,
            timeout=timeout,
        )
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
    return created


def _create_autoresearch_database_secret(
    spec: SecretSpec,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _guard_source_database_secret(args.source_autoresearch_db_secret_id, args.autoresearch_database_name)
    source_secret = _aws_secret_string(
        args.source_autoresearch_db_secret_id,
        profile=args.aws_profile,
        region=args.region,
        timeout=args.timeout,
    )
    database_url = _database_url_from_secret_string(source_secret)
    database = _ensure_postgres_database(database_url, args.autoresearch_database_name)
    payload = {
        "environment": "base-sepolia",
        "purpose": spec.purpose,
        "created_by": "sota_base_testnet_secrets.py",
        "source_secret_id": args.source_autoresearch_db_secret_id,
        spec.json_key: database["database_url"],
    }
    created = _create_secret(
        spec,
        payload,
        address=None,
        profile=args.aws_profile,
        region=args.region,
        timeout=args.timeout,
    )
    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "managed": False,
        "required": spec.required,
        "status": "green",
        "action": "database_created_secret_created" if database["created_database"] else "database_exists_secret_created",
        "arn": created.get("ARN"),
        "database": database["database"],
        "host": database["host"],
        "source_secret_id": args.source_autoresearch_db_secret_id,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    checks: list[dict[str, str]] = []
    for spec in SECRET_SPECS:
        existing, error = _safe_describe_secret(
            spec.name,
            profile=args.aws_profile,
            region=args.region,
            timeout=args.timeout,
        )
        if error:
            actions.append(
                {
                    "name": spec.name,
                    "purpose": spec.purpose,
                    "managed": spec.managed,
                    "status": "red",
                    "detail": f"Could not describe secret handle: {error}",
                }
            )
            continue
        if existing:
            actions.append(
                {
                    "name": spec.name,
                    "purpose": spec.purpose,
                    "managed": spec.managed,
                    "required": spec.required,
                    "status": "green",
                    "action": "exists",
                    "arn": existing.get("ARN"),
                    "address": _address_from_tags(existing),
                }
            )
            continue
        if (
            not spec.managed
            and args.command == "create"
            and args.create_autoresearch_database
            and spec.name == f"{SECRET_PREFIX}autoresearch-database-url"
        ):
            try:
                actions.append(_create_autoresearch_database_secret(spec, args=args))
            except Exception as exc:
                actions.append(
                    {
                        "name": spec.name,
                        "purpose": spec.purpose,
                        "managed": False,
                        "required": spec.required,
                        "status": "red",
                        "action": "database_secret_create_failed",
                        "detail": _redact_secret_text(exc),
                    }
                )
            continue
        if not spec.managed and not spec.required:
            actions.append(
                {
                    "name": spec.name,
                    "purpose": spec.purpose,
                    "managed": False,
                    "required": False,
                    "status": "yellow",
                    "action": "optional_external_missing",
                    "detail": "Optional telemetry secret is not required for claim, mining, or self-validation testing.",
                }
            )
            continue
        if not spec.managed:
            actions.append(
                {
                    "name": spec.name,
                    "purpose": spec.purpose,
                    "managed": False,
                    "required": spec.required,
                    "status": "red",
                    "action": "external_required",
                    "detail": "This value must come from the real database provider; no placeholder was created.",
                }
            )
            continue
        if args.command != "create":
            actions.append(
                {
                    "name": spec.name,
                    "purpose": spec.purpose,
                    "managed": True,
                    "required": spec.required,
                    "status": "yellow",
                    "action": "would_create",
                }
            )
            continue
        payload, address = _secret_payload(spec, rpc_url=args.rpc_url)
        try:
            created = _create_secret(
                spec,
                payload,
                address=address,
                profile=args.aws_profile,
                region=args.region,
                timeout=args.timeout,
            )
        except Exception as exc:
            actions.append(
                {
                    "name": spec.name,
                    "purpose": spec.purpose,
                    "managed": True,
                    "required": spec.required,
                    "status": "red",
                    "action": "create_failed",
                    "detail": str(exc),
                }
            )
            continue
        actions.append(
            {
                "name": spec.name,
                "purpose": spec.purpose,
                "managed": True,
                "required": spec.required,
                "status": "green",
                "action": "created",
                "arn": created.get("ARN"),
                "address": address,
            }
        )

    managed_missing = [item["name"] for item in actions if item.get("required", True) and item["managed"] and item["status"] != "green"]
    external_missing = [item["name"] for item in actions if item.get("required", True) and not item["managed"] and item["status"] != "green"]
    if managed_missing:
        checks.append(
            {
                "name": "managed_secret_handles",
                "status": "yellow" if args.command != "create" else "red",
                "detail": "Managed testnet handles are missing: " + ", ".join(managed_missing),
                "remediation": "Run this script with create after confirming the AWS profile/region.",
            }
        )
    else:
        checks.append({"name": "managed_secret_handles", "status": "green", "detail": "All managed testnet handles exist."})
    if external_missing:
        checks.append(
            {
                "name": "external_secret_handles",
                "status": "red",
                "detail": "External secret handles still need real values: " + ", ".join(external_missing),
                "remediation": "Create the real database URL secret; do not use placeholders.",
            }
        )
    else:
        checks.append({"name": "external_secret_handles", "status": "green", "detail": "All external secret handles exist."})

    rank = {"green": 0, "yellow": 1, "red": 2}
    status = max((check["status"] for check in checks), key=lambda value: rank[value], default="green")
    return {
        "schema": "sota-base-testnet-secret-bootstrap/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "ok": status == "green",
        "status": status,
        "aws": {"profile": args.aws_profile, "region": args.region},
        "read_secret_values": bool(args.command == "create" and args.create_autoresearch_database),
        "prints_secret_values": False,
        "does_not": ["touch_production_bittensor", "touch_base_mainnet", "create_placeholders_for_external_secrets", "print_secret_values"],
        "secret_handles": actions,
        "checks": checks,
        "summary": {
            "green": sum(1 for check in checks if check["status"] == "green"),
            "yellow": sum(1 for check in checks if check["status"] == "yellow"),
            "red": sum(1 for check in checks if check["status"] == "red"),
        },
        "next_actions": [check["remediation"] for check in checks if check["status"] != "green" and check.get("remediation")],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap approved AWS secret handles for Base SOTA Base Sepolia.")
    parser.add_argument("command", choices=("plan", "create"))
    parser.add_argument("--aws-profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-secret-handles.json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    parser.add_argument(
        "--create-autoresearch-database",
        action="store_true",
        help="Create an isolated Base SOTA test database from a test-scoped source DB secret and store its URL.",
    )
    parser.add_argument("--source-autoresearch-db-secret-id", default=DEFAULT_SOURCE_AUTORESEARCH_DB_SECRET_ID)
    parser.add_argument("--autoresearch-database-name", default=DEFAULT_AUTORESEARCH_DATABASE_NAME)
    args = parser.parse_args(argv)

    report = build_report(args)
    _write_json(args.out, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Base SOTA testnet secrets: {str(report['status']).upper()}")
        print(f"Report: {args.out}")
        for item in report["secret_handles"]:
            address = f" address={item['address']}" if item.get("address") else ""
            print(f"- {item['status']} {item['name']} action={item.get('action')}{address}")
        for action in report["next_actions"]:
            print(f"- next: {action}")
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

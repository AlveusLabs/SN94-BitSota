#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen


REPOS = Path("/home/mekaneeky/repos")
DOCS_REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_RPC_URL = "https://sepolia.base.org"
DEFAULT_AWS_REGION = "eu-central-1"
DEFAULT_DEPLOYER_SECRET_ID = "base-sota/test/base-sepolia/deployer"
DEFAULT_ROOT_PUBLISHER_SECRET_ID = "base-sota/test/base-sepolia/root-publisher"
DEFAULT_LOCAL_STATE = REPOS / ".sota-base-local" / "state.json"
DEFAULT_TEMPLATE = DOCS_REPO / "docs" / "base" / "manifests" / "base-sepolia-deployment-manifest.template.json"
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


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str
    detail: str
    remediation: str = ""
    artifacts: dict[str, str] | None = None
    command: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }
        if self.remediation:
            payload["remediation"] = self.remediation
        if self.artifacts:
            payload["artifacts"] = self.artifacts
        if self.command:
            payload["command"] = [_redact_arg(item) for item in self.command]
        return payload


def _redact_arg(value: str) -> str:
    text = str(value)
    if "PRIVATE_KEY" in text or "MNEMONIC" in text or "SECRET" in text:
        return "<secret-env-ref>"
    return text


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(status, 2)


def _worst(steps: list[StepResult]) -> str:
    if not steps:
        return "green"
    return max((step.status for step in steps), key=_status_rank)


def _summary(steps: list[StepResult]) -> dict[str, int]:
    return {
        "green": sum(1 for step in steps if step.status == "green"),
        "yellow": sum(1 for step in steps if step.status == "yellow"),
        "red": sum(1 for step in steps if step.status == "red"),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _fill_seed_inputs_from_local_state(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, str]:
    state_path = Path(getattr(args, "local_state", DEFAULT_LOCAL_STATE))
    if not state_path.exists():
        return {}
    try:
        state = _load_json(state_path)
    except Exception:
        return {}
    filled: dict[str, str] = {}
    accounts = dict(state.get("accounts") or {})
    genesis = dict(state.get("genesis") or {})
    autoresearch = dict(state.get("autoresearch") or {})
    evidence = dict(autoresearch.get("evidence") or {})
    if not args.test_wallet_address and accounts.get("alice_reward"):
        args.test_wallet_address = str(accounts["alice_reward"])
        filled["test_wallet_address"] = str(state_path)
    if not args.test_old_coldkey and genesis.get("old_coldkey"):
        args.test_old_coldkey = str(genesis["old_coldkey"])
        filled["test_old_coldkey"] = str(state_path)
    if not args.emission_evidence and evidence:
        evidence_path = paths["artifacts_dir"] / "base-sota-testnet-emission-evidence-from-local.json"
        _write_json(evidence_path, evidence)
        args.emission_evidence = evidence_path
        filled["emission_evidence"] = str(evidence_path)
    return filled


def _load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(_redact_arg(item)) for item in cmd)


def _run_command(
    cmd: list[str],
    *,
    cwd: Path = DOCS_REPO,
    timeout: float = 120.0,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout) or "",
            "stderr": f"command timed out after {timeout} seconds",
            "command": cmd,
            "command_text": _command_text(cmd),
        }
    return {
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": cmd,
        "command_text": _command_text(cmd),
    }


def _aws_secret_string(secret_id: str, *, profile: str, region: str, timeout: float) -> str:
    cmd = [
        "aws",
        "secretsmanager",
        "get-secret-value",
        "--secret-id",
        secret_id,
        "--query",
        "SecretString",
        "--output",
        "text",
    ]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"aws exited {result.returncode}")
    value = result.stdout.strip()
    if not value or value == "None":
        raise RuntimeError(f"secret {secret_id!r} has no SecretString")
    return value


def _aws_secret_tag(secret_id: str, tag_key: str, *, profile: str, region: str, timeout: float) -> str:
    cmd = [
        "aws",
        "secretsmanager",
        "describe-secret",
        "--secret-id",
        secret_id,
        "--output",
        "json",
    ]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"aws exited {result.returncode}")
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError("describe-secret returned non-object JSON")
    for tag in payload.get("Tags") or []:
        if isinstance(tag, dict) and str(tag.get("Key") or "") == tag_key and tag.get("Value"):
            return str(tag["Value"]).strip()
    return ""


def _fill_root_publisher_address_from_secret(args: argparse.Namespace) -> str:
    if not args.deploy or args.root_publisher_address or not args.root_publisher_private_key_secret_id:
        return ""
    try:
        address = _aws_secret_tag(
            args.root_publisher_private_key_secret_id,
            "sota-address",
            profile=args.aws_profile,
            region=args.aws_region,
            timeout=args.timeout,
        )
    except Exception:
        return ""
    if address:
        args.root_publisher_address = address
    return address


def _extract_secret_value(secret_string: str, *, json_key: str, env_name: str) -> str:
    text = secret_string.strip()
    if not text:
        raise ValueError("secret string is empty")
    if not text.startswith("{"):
        return text
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("JSON secret must be an object")
    candidate_keys = [json_key] if json_key else []
    candidate_keys.extend(
        [
            env_name,
            "private_key",
            "deployer_private_key",
            "sota_deployer_private_key",
            "root_publisher_private_key",
            "sota_root_publisher_private_key",
        ]
    )
    for key in candidate_keys:
        if key and payload.get(key):
            return str(payload[key]).strip()
    raise ValueError(
        "JSON secret did not contain the configured key or any supported private key field"
    )


def _deployment_env_overrides(args: argparse.Namespace) -> tuple[dict[str, str], str]:
    if not args.deploy:
        return {}, ""
    if os.environ.get(args.private_key_env):
        return {}, ""
    if not args.private_key_secret_id:
        return {}, ""
    try:
        secret_string = _aws_secret_string(
            args.private_key_secret_id,
            profile=args.aws_profile,
            region=args.aws_region,
            timeout=args.timeout,
        )
        secret_value = _extract_secret_value(
            secret_string,
            json_key=args.private_key_secret_json_key,
            env_name=args.private_key_env,
        )
    except Exception as exc:
        return {}, f"Could not load {args.private_key_env} from AWS secret handle {args.private_key_secret_id!r}: {exc}"
    return {args.private_key_env: secret_value}, ""


def _root_publisher_env_overrides(args: argparse.Namespace) -> tuple[dict[str, str], str]:
    env_name = "SOTA_ROOT_PUBLISHER_PRIVATE_KEY"
    if not args.broadcast_roots:
        return {}, ""
    if os.environ.get(env_name):
        return {}, ""
    if not args.root_publisher_private_key_secret_id:
        return {}, ""
    try:
        secret_string = _aws_secret_string(
            args.root_publisher_private_key_secret_id,
            profile=args.aws_profile,
            region=args.aws_region,
            timeout=args.timeout,
        )
        secret_value = _extract_secret_value(
            secret_string,
            json_key=args.root_publisher_private_key_secret_json_key,
            env_name=env_name,
        )
    except Exception as exc:
        return {}, f"Could not load {env_name} from AWS secret handle {args.root_publisher_private_key_secret_id!r}: {exc}"
    return {env_name: secret_value}, ""


def _step_from_result(name: str, result: dict[str, Any], *, success_detail: str, failure_remediation: str, artifacts: dict[str, str] | None = None) -> StepResult:
    if int(result["returncode"]) == 0:
        return StepResult(name, "green", success_detail, artifacts=artifacts, command=list(result.get("command") or []))
    detail = (str(result.get("stderr") or "").strip() or str(result.get("stdout") or "").strip() or f"exit {result['returncode']}")
    if "has no native gas balance" in detail:
        failure_remediation = "Fund the listed deployer address with Base Sepolia ETH, then rerun the operator with --deploy."
    return StepResult(name, "red", detail[:1000], failure_remediation, artifacts=artifacts, command=list(result.get("command") or []))


def _step_from_json_report(
    name: str,
    result: dict[str, Any],
    *,
    report_path: Path,
    expected_schema: str | None,
    success_detail: str,
    failure_remediation: str,
    artifacts: dict[str, str] | None = None,
) -> StepResult:
    if int(result["returncode"]) != 0 and not report_path.exists():
        return _step_from_result(
            name,
            result,
            success_detail=success_detail,
            failure_remediation=failure_remediation,
            artifacts=artifacts,
        )
    try:
        payload = _load_json(report_path)
    except Exception as exc:
        return StepResult(
            name,
            "red",
            f"{report_path} could not be read after command execution: {exc}",
            failure_remediation,
            artifacts=artifacts,
            command=list(result.get("command") or []),
        )
    schema = str(payload.get("schema") or "")
    if expected_schema and schema != expected_schema:
        return StepResult(
            name,
            "red",
            f"{report_path} schema is {schema or 'missing'}, expected {expected_schema}.",
            failure_remediation,
            artifacts=artifacts,
            command=list(result.get("command") or []),
        )
    status = str(payload.get("status") or ("green" if payload.get("ok") else "red"))
    if status not in {"green", "yellow", "red"}:
        status = "green" if payload.get("ok") else "red"
    if status == "green" and not payload.get("ok", status == "green"):
        status = "red"
    detail = success_detail if status == "green" else str(payload.get("message") or f"{report_path} status is {status}.")
    remediation = failure_remediation
    next_actions = payload.get("next_actions")
    if status != "green" and isinstance(next_actions, list) and next_actions:
        remediation = str(next_actions[0])
    return StepResult(
        name,
        status,
        detail,
        "" if status == "green" else remediation,
        artifacts=artifacts,
        command=list(result.get("command") or []),
    )


def _service_pack_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        "scripts/sota_base_testnet_service_pack.py",
        "--manifest",
        str(paths["manifest"]),
        "--env-file",
        str(paths["env"]),
        "--claims-ui",
        args.claims_ui_url,
        "--claims-api",
        args.claims_api_url,
        "--coordinator",
        args.coordinator_url,
        "--attestation",
        args.attestation_url,
        "--root-publisher",
        args.root_publisher_url,
        "--claim-artifacts",
        args.claim_artifacts_url,
        "--monitoring",
        args.monitoring_url,
        "--readiness-url",
        args.readiness_url,
        "--json-out",
        str(paths["service_pack_json"]),
        "--markdown-out",
        str(paths["service_pack_md"]),
        "--html-out",
        str(paths["service_pack_html"]),
        "--apprunner-out-dir",
        str(paths["apprunner_dir"]),
    ]


def _apprunner_source_pack_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        "scripts/sota_base_testnet_apprunner_source_pack.py",
        "--service-pack",
        str(paths["service_pack_json"]),
        "--apprunner-input-dir",
        str(paths["apprunner_dir"]),
        "--out-dir",
        str(paths["apprunner_source_dir"]),
        "--report-out",
        str(paths["apprunner_source_pack"]),
        "--aws-profile",
        args.aws_profile,
        "--region",
        args.aws_region,
        "--allow-blocked",
    ]


def _blockers_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/sota_base_testnet_blockers.py",
        "--artifacts-dir",
        str(args.artifacts_dir),
        "--report-out",
        str(paths["blockers"]),
        "--rpc-url",
        args.rpc_url,
        "--aws-profile",
        args.aws_profile,
        "--readiness-url",
        args.readiness_url,
        "--allow-blocked",
    ]
    for name, value in (
        ("claims_ui", args.claims_ui_url),
        ("claims_api", args.claims_api_url),
        ("coordinator", args.coordinator_url),
        ("attestation", args.attestation_url),
        ("root_publisher", args.root_publisher_url),
    ):
        cmd.extend(["--host", f"{name}={value}"])
    if args.test_wallet_address:
        cmd.extend(["--gas-address", f"test_wallet={args.test_wallet_address}"])
    return cmd


def _aws_inventory_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/sota_base_testnet_aws_inventory.py",
        "--aws-profile",
        args.aws_profile,
        "--region",
        args.aws_region,
        "--timeout",
        str(args.timeout),
        "--out",
        str(paths["aws_inventory"]),
        "--allow-blocked",
    ]
    for name, value in (
        ("claims_ui", args.claims_ui_url),
        ("claims_api", args.claims_api_url),
        ("coordinator", args.coordinator_url),
        ("root_publisher", args.root_publisher_url),
    ):
        cmd.extend(["--service-url", f"{name}={value}"])
    return cmd


def _rehearsal_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/sota_base_testnet_rehearsal.py",
        "--artifacts-dir",
        str(args.artifacts_dir),
        "--template",
        str(args.template),
        "--rpc-url",
        args.rpc_url,
        "--default-lane-id",
        args.default_lane_id,
        "--claims-ui-url",
        args.claims_ui_url,
        "--claims-ui-health-url",
        args.claims_ui_health_url,
        "--indexer-api-url",
        args.claims_api_url,
        "--indexer-api-health-url",
        args.claims_api_health_url,
        "--root-publisher-url",
        args.root_publisher_url,
        "--root-publisher-health-url",
        args.root_publisher_health_url,
        "--attestation-builder-url",
        args.attestation_url,
        "--attestation-builder-health-url",
        args.attestation_health_url,
        "--monitoring-url",
        args.monitoring_url,
        "--monitoring-alert-policy-url",
        args.monitoring_alert_policy_url,
        "--monitoring-log-group-or-sink",
        args.monitoring_log_group_or_sink,
        "--autoresearch-api-url",
        args.coordinator_url,
        "--test-wallet-address",
        args.test_wallet_address,
        "--test-old-coldkey",
        args.test_old_coldkey,
        "--test-epoch",
        str(args.test_epoch),
        "--readiness-url",
        args.readiness_url,
        "--readiness-out",
        str(paths["readiness"]),
        "--report-out",
        str(paths["rehearsal_report"]),
        "--timeout",
        str(args.timeout),
        "--allow-blocked",
    ]
    if args.deploy:
        cmd.append("--deploy")
        cmd.extend(["--private-key-env", args.private_key_env])
        cmd.extend(["--initial-vault-supply-sota", args.initial_vault_supply_sota])
        if args.owner_address:
            cmd.extend(["--owner-address", args.owner_address])
        if args.supply_authority_address:
            cmd.extend(["--supply-authority-address", args.supply_authority_address])
        if args.emission_authority_address:
            cmd.extend(["--emission-authority-address", args.emission_authority_address])
        if args.root_publisher_address:
            cmd.extend(["--root-publisher-address", args.root_publisher_address])
    else:
        cmd.extend(["--deployment", str(args.deployment)])
    if args.build_website:
        cmd.append("--build-website")
    return cmd


def _seed_build_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        "scripts/sota_base_testnet_seed_artifacts.py",
        "build",
        "--manifest",
        str(paths["manifest"]),
        "--emission-evidence",
        str(args.emission_evidence),
        "--test-wallet-address",
        args.test_wallet_address,
        "--test-old-coldkey",
        args.test_old_coldkey,
        "--lane-id",
        args.default_lane_id,
        "--min-accepted-count",
        str(args.min_accepted_count),
        "--min-committee-count",
        str(args.min_committee_count),
        "--out-dir",
        str(args.artifacts_dir),
    ]


def _publish_cmd(paths: dict[str, Path], *, kind: str, broadcast: bool) -> list[str]:
    artifact_key = "genesis_root_artifact" if kind == "genesis" else "emission_root_artifact"
    out_key = "genesis_publish_result" if kind == "genesis" else "emission_publish_result"
    cmd = [
        sys.executable,
        "scripts/sota_base_publish_root.py",
        "--manifest",
        str(paths["manifest"]),
        "--root-artifact",
        str(paths[artifact_key]),
        "--kind",
        kind,
        "--out",
        str(paths[out_key]),
    ]
    if broadcast:
        cmd.append("--broadcast")
    return cmd


def _finalize_cmd(paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        "scripts/sota_base_testnet_seed_artifacts.py",
        "finalize",
        "--build-report",
        str(paths["seed_report"]),
        "--genesis-publish-result",
        str(paths["genesis_publish_result"]),
        "--emission-publish-result",
        str(paths["emission_publish_result"]),
        "--out-dir",
        str(paths["artifacts_dir"]),
    ]


def _browser_smoke_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        "scripts/sota_base_testnet_browser_smoke.py",
        "--artifacts-dir",
        str(args.artifacts_dir),
        "--report-out",
        str(paths["browser_smoke"]),
        "--claims-url",
        args.claims_ui_url,
        "--claims-api-url",
        args.claims_api_url,
        "--autoresearch-url",
        args.coordinator_url,
        "--readiness-url",
        args.readiness_url,
        "--test-wallet-address",
        args.test_wallet_address,
        "--test-old-coldkey",
        args.test_old_coldkey,
        "--lane-id",
        args.default_lane_id,
        "--epoch",
        str(args.test_epoch),
        "--timeout",
        str(args.timeout),
        "--allow-blocked",
    ]


def _release_status_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        "scripts/sota_base_release_status.py",
        "--testnet-artifacts-dir",
        str(args.artifacts_dir),
        "--report-out",
        str(paths["release_status"]),
        "--allow-blocked",
    ]


def _paths(artifacts_dir: Path) -> dict[str, Path]:
    return {
        "artifacts_dir": artifacts_dir,
        "compact_deployment": artifacts_dir / "base-sepolia-compact-deployment.json",
        "manifest": artifacts_dir / "base-sepolia-deployment-manifest.json",
        "env": artifacts_dir / "base-sota.env.testnet",
        "readiness": artifacts_dir / "base-sota-testnet-readiness.json",
        "service_pack_json": artifacts_dir / "base-sota-testnet-service-pack.json",
        "service_pack_md": artifacts_dir / "base-sota-testnet-service-pack.md",
        "service_pack_html": artifacts_dir / "base-sota-testnet-service-pack.html",
        "apprunner_dir": artifacts_dir / "apprunner",
        "apprunner_source_dir": artifacts_dir / "apprunner-source",
        "apprunner_source_pack": artifacts_dir / "base-sota-testnet-apprunner-source-pack.json",
        "blockers": artifacts_dir / "base-sota-testnet-blockers.json",
        "aws_inventory": artifacts_dir / "base-sota-testnet-aws-inventory.json",
        "rehearsal_report": artifacts_dir / "base-sota-testnet-rehearsal.json",
        "seed_report": artifacts_dir / "base-sota-testnet-seed-artifacts.json",
        "seed_finalized_report": artifacts_dir / "base-sota-testnet-seed-artifacts-finalized.json",
        "genesis_root_artifact": artifacts_dir / "base-sota-testnet-genesis-root-artifact.json",
        "emission_root_artifact": artifacts_dir / "base-sota-testnet-emission-root-artifact.json",
        "genesis_publish_result": artifacts_dir / "base-sota-testnet-genesis-root-publish-result.json",
        "emission_publish_result": artifacts_dir / "base-sota-testnet-emission-root-publish-result.json",
        "genesis_claim_artifact": artifacts_dir / "base-sota-testnet-genesis-claim-artifact.json",
        "emission_claim_artifact": artifacts_dir / "base-sota-testnet-emission-claim-artifact.json",
        "browser_smoke": artifacts_dir / "base-sota-testnet-browser-smoke.json",
        "release_status": artifacts_dir / "base-sota-release-status.json",
        "operator_report": artifacts_dir / "base-sota-testnet-operator-run.json",
    }


def _claims_api_url(args: argparse.Namespace, paths: dict[str, Path]) -> str:
    env = _load_env(paths["env"])
    return args.claims_api_url or env.get("SOTA_CLAIMS_API_URL") or env.get("NEXT_PUBLIC_SOTA_CLAIMS_API_URL") or DEFAULT_URLS["claims_api"]


def _post_json(url: str, payload: dict[str, Any], *, token_env: str, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.environ.get(token_env, "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body) if body.strip() else {}
    return dict(parsed or {})


def _import_claim_artifacts(args: argparse.Namespace, paths: dict[str, Path]) -> StepResult:
    missing = [path for path in (paths["genesis_claim_artifact"], paths["emission_claim_artifact"]) if not path.exists()]
    if missing:
        return StepResult(
            "import_claim_artifacts",
            "red",
            "Finalized claim artifact is missing: " + ", ".join(str(path) for path in missing),
            "Broadcast both roots, finalize claim artifacts, then import them into the testnet indexer.",
        )
    base = _claims_api_url(args, paths)
    try:
        genesis = _post_json(
            _url(base, "/api/v1/base/index/artifact"),
            _load_json(paths["genesis_claim_artifact"]),
            token_env=args.indexer_admin_token_env,
            timeout=args.timeout,
        )
        emission = _post_json(
            _url(base, "/api/v1/base/index/artifact"),
            _load_json(paths["emission_claim_artifact"]),
            token_env=args.indexer_admin_token_env,
            timeout=args.timeout,
        )
    except Exception as exc:
        return StepResult(
            "import_claim_artifacts",
            "red",
            f"Indexer import failed for {base}: {exc}",
            "Fix the public claims API/indexer service, then rerun with --import-artifacts.",
        )
    return StepResult(
        "import_claim_artifacts",
        "green",
        f"Imported finalized genesis and emission claim artifacts into {base}.",
        artifacts={"genesis_response": json.dumps(genesis, sort_keys=True), "emission_response": json.dumps(emission, sort_keys=True)},
    )


def _publish_step(
    paths: dict[str, Path],
    *,
    kind: str,
    broadcast: bool,
    timeout: float,
    env_overrides: dict[str, str] | None = None,
) -> StepResult:
    cmd = _publish_cmd(paths, kind=kind, broadcast=broadcast)
    result = _run_command(cmd, timeout=timeout, env_overrides=env_overrides)
    out_key = "genesis_publish_result" if kind == "genesis" else "emission_publish_result"
    if int(result["returncode"]) != 0:
        return _step_from_result(
            f"publish_{kind}_root",
            result,
            success_detail="",
            failure_remediation="Fix the root artifact, manifest, signer, or Base Sepolia transaction failure.",
            artifacts={out_key: str(paths[out_key])},
        )
    if not broadcast:
        return StepResult(
            f"publish_{kind}_root",
            "yellow",
            f"Built dry-run {kind} root publish transaction; root was not broadcast.",
            "Rerun with --broadcast-roots after the root publisher signer is funded and approved.",
            artifacts={out_key: str(paths[out_key])},
            command=cmd,
        )
    payload = _load_json(paths[out_key])
    if payload.get("status") == "broadcasted" and payload.get("root_id"):
        return StepResult(
            f"publish_{kind}_root",
            "green",
            f"Broadcast {kind} root and recorded emitted root_id {payload['root_id']}.",
            artifacts={out_key: str(paths[out_key])},
            command=cmd,
        )
    return StepResult(
        f"publish_{kind}_root",
        "red",
        f"Broadcast result did not include status=broadcasted and root_id: {paths[out_key]}",
        "Inspect the root publish result before finalizing claim artifacts.",
        artifacts={out_key: str(paths[out_key])},
        command=cmd,
    )


def run_operator(args: argparse.Namespace) -> dict[str, Any]:
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(args.artifacts_dir)
    resolved_seed_inputs = _fill_seed_inputs_from_local_state(args, paths)
    resolved_root_publisher_address = _fill_root_publisher_address_from_secret(args)
    steps: list[StepResult] = []

    def build_report(current_steps: list[StepResult]) -> dict[str, Any]:
        status = _worst(current_steps)
        return {
            "schema": "sota-base-testnet-operator-run/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ok": status == "green",
            "status": status,
            "message": (
                "Base Sepolia operator run completed all requested steps."
                if status == "green"
                else "Base Sepolia operator run is not ready for a nontechnical tester yet."
            ),
            "environment": "base-sepolia",
            "read_only_default": not args.broadcast_roots and not args.import_artifacts and not args.deploy,
            "does_not": [
                "touch_production_bittensor",
                "touch_base_mainnet",
                "print_private_keys",
                "use_mock_claims",
            ],
            "artifacts_dir": str(args.artifacts_dir),
            "paths": {key: str(value) for key, value in paths.items()},
            "resolved_addresses": {
                "root_publisher": args.root_publisher_address or resolved_root_publisher_address,
            },
            "resolved_seed_inputs": resolved_seed_inputs,
            "steps": [step.as_dict() for step in current_steps],
            "summary": _summary(current_steps),
            "next_actions": [
                step.remediation
                for step in current_steps
                if step.status != "green" and step.remediation
            ],
        }

    service_pack = _run_command(_service_pack_cmd(args, paths), timeout=args.timeout)
    steps.append(
        _step_from_json_report(
            "service_pack",
            service_pack,
            report_path=paths["service_pack_json"],
            expected_schema="sota-base-testnet-service-pack/v1",
            success_detail="Generated the Base Sepolia service pack.",
            failure_remediation="Fix service pack generation before continuing testnet deployment.",
            artifacts={
                "json": str(paths["service_pack_json"]),
                "markdown": str(paths["service_pack_md"]),
                "html": str(paths["service_pack_html"]),
            },
        )
    )

    source_pack = _run_command(_apprunner_source_pack_cmd(args, paths), timeout=args.timeout)
    steps.append(
        _step_from_json_report(
            "apprunner_source_pack",
            source_pack,
            report_path=paths["apprunner_source_pack"],
            expected_schema="sota-base-testnet-apprunner-source-pack/v1",
            success_detail="Rendered source-based App Runner service inputs.",
            failure_remediation="Commit and push Base SOTA service branches and resolve App Runner connection/runtime roles before public service creation.",
            artifacts={
                "report": str(paths["apprunner_source_pack"]),
                "rendered_inputs": str(paths["apprunner_source_dir"]),
            },
        )
    )

    blockers = _run_command(_blockers_cmd(args, paths), timeout=args.timeout)
    steps.append(
        _step_from_json_report(
            "blocker_gate",
            blockers,
            report_path=paths["blockers"],
            expected_schema="sota-base-testnet-blockers/v1",
            success_detail="Regenerated the read-only Base Sepolia blocker report.",
            failure_remediation="Clear AWS/DNS/artifact blockers before inviting a nontechnical tester.",
            artifacts={"report": str(paths["blockers"])},
        )
    )

    aws_inventory = _run_command(_aws_inventory_cmd(args, paths), timeout=args.timeout)
    steps.append(
        _step_from_json_report(
            "aws_inventory",
            aws_inventory,
            report_path=paths["aws_inventory"],
            expected_schema="sota-base-testnet-aws-inventory/v1",
            success_detail="Inventoried AWS Base SOTA testnet services, DNS, ECR repositories, and secret handles.",
            failure_remediation="Create or document the missing Base SOTA testnet AWS resources before inviting a nontechnical tester.",
            artifacts={"report": str(paths["aws_inventory"])},
        )
    )

    if args.deploy or args.deployment:
        deploy_env, deploy_env_error = _deployment_env_overrides(args)
        if deploy_env_error:
            steps.append(
                StepResult(
                    "rehearsal",
                    "red",
                    deploy_env_error,
                    "Create the approved Base Sepolia deployer secret handle or export SOTA_DEPLOYER_PRIVATE_KEY in the operator process.",
                    artifacts={
                        "compact_deployment": str(paths["compact_deployment"] if args.deploy else args.deployment),
                        "manifest": str(paths["manifest"]),
                        "env": str(paths["env"]),
                        "readiness": str(paths["readiness"]),
                        "report": str(paths["rehearsal_report"]),
                    },
                    command=_rehearsal_cmd(args, paths),
                )
            )
        else:
            rehearsal = _run_command(_rehearsal_cmd(args, paths), timeout=args.command_timeout, env_overrides=deploy_env)
            steps.append(
                _step_from_json_report(
                    "rehearsal",
                    rehearsal,
                    report_path=paths["rehearsal_report"],
                    expected_schema=None,
                    success_detail="Ran guarded Base Sepolia contract/manifest/env/preflight rehearsal.",
                    failure_remediation="Fix Base Sepolia deployment/rehearsal before building claim artifacts.",
                    artifacts={
                        "compact_deployment": str(paths["compact_deployment"] if args.deploy else args.deployment),
                        "manifest": str(paths["manifest"]),
                        "env": str(paths["env"]),
                        "readiness": str(paths["readiness"]),
                        "report": str(paths["rehearsal_report"]),
                    },
                )
            )
    else:
        steps.append(
            StepResult(
                "rehearsal",
                "red",
                "No compact deployment was provided and --deploy was not requested.",
                "Provide --deployment <base-sepolia-compact-deployment.json> or rerun with --deploy and an approved funded deployer.",
            )
        )

    if args.emission_evidence and paths["manifest"].exists():
        seed = _run_command(_seed_build_cmd(args, paths), timeout=args.timeout)
        steps.append(
            _step_from_result(
                "seed_artifacts",
                seed,
                success_detail="Built publish-ready genesis/emission root artifacts from real autoresearch evidence.",
                failure_remediation="Provide accepted self-validation emission evidence and a filled Base Sepolia manifest.",
                artifacts={
                    "report": str(paths["seed_report"]),
                    "genesis_root": str(paths["genesis_root_artifact"]),
                    "emission_root": str(paths["emission_root_artifact"]),
                },
            )
        )
    else:
        missing = []
        if not args.emission_evidence:
            missing.append("--emission-evidence")
        if not paths["manifest"].exists():
            missing.append(str(paths["manifest"]))
        if not paths["manifest"].exists() and args.emission_evidence:
            seed_remediation = "Run rehearsal/deployment to create the Base Sepolia deployment manifest, then rebuild seed artifacts."
        elif not args.emission_evidence:
            seed_remediation = "Run the local demo first or provide accepted self-validation emission evidence, then rerun the operator."
        else:
            seed_remediation = "Run rehearsal to create the manifest and provide accepted autoresearch emission evidence."
        steps.append(
            StepResult(
                "seed_artifacts",
                "red",
                "Seed artifact build inputs are missing: " + ", ".join(missing),
                seed_remediation,
            )
        )

    if paths["genesis_root_artifact"].exists() and paths["emission_root_artifact"].exists():
        publish_env, publish_env_error = _root_publisher_env_overrides(args)
        if publish_env_error:
            for kind in ("genesis", "emission"):
                steps.append(
                    StepResult(
                        f"publish_{kind}_root",
                        "red",
                        publish_env_error,
                        "Create the approved Base Sepolia root publisher secret handle or export SOTA_ROOT_PUBLISHER_PRIVATE_KEY in the operator process.",
                        artifacts={
                            "genesis_publish_result" if kind == "genesis" else "emission_publish_result": str(
                                paths["genesis_publish_result"] if kind == "genesis" else paths["emission_publish_result"]
                            )
                        },
                        command=_publish_cmd(paths, kind=kind, broadcast=args.broadcast_roots),
                    )
                )
        else:
            steps.append(_publish_step(paths, kind="genesis", broadcast=args.broadcast_roots, timeout=args.command_timeout, env_overrides=publish_env))
            steps.append(_publish_step(paths, kind="emission", broadcast=args.broadcast_roots, timeout=args.command_timeout, env_overrides=publish_env))
    else:
        steps.append(
            StepResult(
                "publish_genesis_root",
                "red",
                "Genesis root artifact is missing.",
                "Build seed artifacts before publishing roots.",
            )
        )
        steps.append(
            StepResult(
                "publish_emission_root",
                "red",
                "Emission root artifact is missing.",
                "Build seed artifacts before publishing roots.",
            )
        )

    if args.broadcast_roots:
        finalize = _run_command(_finalize_cmd(paths), timeout=args.timeout)
        steps.append(
            _step_from_result(
                "finalize_claim_artifacts",
                finalize,
                success_detail="Finalized claim artifacts with emitted on-chain root IDs.",
                failure_remediation="Fix root publish results, then rerun finalize.",
                artifacts={
                    "report": str(paths["seed_finalized_report"]),
                    "genesis_claim": str(paths["genesis_claim_artifact"]),
                    "emission_claim": str(paths["emission_claim_artifact"]),
                },
            )
        )
    else:
        steps.append(
            StepResult(
                "finalize_claim_artifacts",
                "yellow",
                "Skipped finalization because roots were dry-run only.",
                "Rerun with --broadcast-roots to record emitted root IDs, then finalize claim artifacts.",
            )
        )

    if args.import_artifacts:
        steps.append(_import_claim_artifacts(args, paths))
    else:
        steps.append(
            StepResult(
                "import_claim_artifacts",
                "yellow",
                "Skipped indexer import.",
                "Rerun with --import-artifacts after finalized claim artifacts exist and the claims API is deployed.",
            )
        )

    if not args.skip_browser_smoke:
        browser = _run_command(_browser_smoke_cmd(args, paths), timeout=args.command_timeout)
        steps.append(
            _step_from_json_report(
                "browser_smoke",
                browser,
                report_path=paths["browser_smoke"],
                expected_schema="sota-base-testnet-browser-smoke/v1",
                success_detail="Regenerated public Base Sepolia browser smoke report.",
                failure_remediation="Deploy public services/artifacts and rerun browser smoke before inviting a nontechnical tester.",
                artifacts={"report": str(paths["browser_smoke"])},
            )
        )
    else:
        steps.append(
            StepResult(
                "browser_smoke",
                "yellow",
                "Browser smoke skipped by operator flag.",
                "Run without --skip-browser-smoke before giving the testnet to a nontechnical tester.",
            )
        )

    _write_json(paths["operator_report"], build_report(steps))
    release = _run_command(_release_status_cmd(args, paths), timeout=args.timeout)
    steps.append(
        _step_from_json_report(
            "release_status",
            release,
            report_path=paths["release_status"],
            expected_schema="sota-base-release-status/v1",
            success_detail="Regenerated aggregate Base SOTA release status.",
            failure_remediation="Clear all local and Base Sepolia release gates.",
            artifacts={"report": str(paths["release_status"])},
        )
    )

    report = build_report(steps)
    _write_json(paths["operator_report"], report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the guarded end-to-end Base SOTA Base Sepolia operator path.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL)
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""))
    parser.add_argument("--aws-region", default=os.environ.get("AWS_REGION", DEFAULT_AWS_REGION))
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--deploy", action="store_true", help="Broadcast Base Sepolia contract deployment transactions")
    parser.add_argument("--private-key-env", default="SOTA_DEPLOYER_PRIVATE_KEY")
    parser.add_argument("--private-key-secret-id", default=os.environ.get("SOTA_DEPLOYER_PRIVATE_KEY_SECRET_ID", DEFAULT_DEPLOYER_SECRET_ID))
    parser.add_argument("--private-key-secret-json-key", default=os.environ.get("SOTA_DEPLOYER_PRIVATE_KEY_SECRET_JSON_KEY", ""))
    parser.add_argument("--initial-vault-supply-sota", default=os.environ.get("SOTA_INITIAL_VAULT_SUPPLY", "1000000"))
    parser.add_argument("--owner-address", default=os.environ.get("SOTA_OWNER_ADDRESS", ""))
    parser.add_argument("--supply-authority-address", default=os.environ.get("SOTA_SUPPLY_AUTHORITY_ADDRESS", ""))
    parser.add_argument("--emission-authority-address", default=os.environ.get("SOTA_EMISSION_AUTHORITY_ADDRESS", ""))
    parser.add_argument("--root-publisher-address", default=os.environ.get("SOTA_ROOT_PUBLISHER_ADDRESS", ""))
    parser.add_argument("--default-lane-id", default=os.environ.get("NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID", "base:sota-local"))
    parser.add_argument("--emission-evidence", type=Path)
    parser.add_argument("--local-state", type=Path, default=DEFAULT_LOCAL_STATE)
    parser.add_argument("--test-wallet-address", default=os.environ.get("SOTA_TEST_WALLET_ADDRESS", ""))
    parser.add_argument("--test-old-coldkey", default=os.environ.get("SOTA_TEST_OLD_COLDKEY", ""))
    parser.add_argument("--test-epoch", default=os.environ.get("SOTA_TEST_EPOCH", "1"))
    parser.add_argument("--min-accepted-count", type=int, default=3)
    parser.add_argument("--min-committee-count", type=int, default=3)
    parser.add_argument("--claims-ui-url", default=os.environ.get("SOTA_CLAIMS_UI_URL", DEFAULT_URLS["claims_ui"]))
    parser.add_argument("--claims-ui-health-url", default=os.environ.get("SOTA_CLAIMS_UI_HEALTH_URL", ""))
    parser.add_argument("--claims-api-url", default=os.environ.get("SOTA_CLAIMS_API_URL", DEFAULT_URLS["claims_api"]))
    parser.add_argument("--claims-api-health-url", default=os.environ.get("SOTA_INDEXER_HEALTHCHECK_URL", ""))
    parser.add_argument("--coordinator-url", default=os.environ.get("SOTA_COORDINATOR_URL", DEFAULT_URLS["coordinator"]))
    parser.add_argument("--attestation-url", default=os.environ.get("SOTA_ATTESTATION_SERVICE_URL", DEFAULT_URLS["attestation"]))
    parser.add_argument("--attestation-health-url", default=os.environ.get("SOTA_ATTESTATION_HEALTHCHECK_URL", ""))
    parser.add_argument("--root-publisher-url", default=os.environ.get("SOTA_ROOT_PUBLISHER_URL", DEFAULT_URLS["root_publisher"]))
    parser.add_argument("--root-publisher-health-url", default=os.environ.get("SOTA_ROOT_PUBLISHER_HEALTH_URL", ""))
    parser.add_argument("--claim-artifacts-url", default=os.environ.get("SOTA_CLAIM_ARTIFACTS_URL", DEFAULT_URLS["claim_artifacts"]))
    parser.add_argument("--monitoring-url", default=os.environ.get("SOTA_MONITORING_URL", DEFAULT_URLS["monitoring"]))
    parser.add_argument("--monitoring-alert-policy-url", default=os.environ.get("SOTA_MONITORING_ALERT_POLICY_URL", ""))
    parser.add_argument("--monitoring-log-group-or-sink", default=os.environ.get("SOTA_MONITORING_LOG_GROUP_OR_SINK", ""))
    parser.add_argument("--readiness-url", default=os.environ.get("NEXT_PUBLIC_SOTA_READINESS_URL", DEFAULT_URLS["readiness"]))
    parser.add_argument("--build-website", action="store_true")
    parser.add_argument("--broadcast-roots", action="store_true", help="Broadcast root publication transactions with SOTA_ROOT_PUBLISHER_PRIVATE_KEY")
    parser.add_argument("--root-publisher-private-key-secret-id", default=os.environ.get("SOTA_ROOT_PUBLISHER_PRIVATE_KEY_SECRET_ID", DEFAULT_ROOT_PUBLISHER_SECRET_ID))
    parser.add_argument("--root-publisher-private-key-secret-json-key", default=os.environ.get("SOTA_ROOT_PUBLISHER_PRIVATE_KEY_SECRET_JSON_KEY", ""))
    parser.add_argument("--import-artifacts", action="store_true", help="POST finalized claim artifacts into the configured claims API")
    parser.add_argument("--indexer-admin-token-env", default="SOTA_INDEXER_ADMIN_TOKEN")
    parser.add_argument("--skip-browser-smoke", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--command-timeout", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when operator report remains red/yellow.")
    args = parser.parse_args(argv)

    report = run_operator(args)
    if args.report_out is not None:
        _write_json(args.report_out, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Base SOTA testnet operator run: {str(report['status']).upper()}")
        print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
        print(f"Report: {report['paths']['operator_report']}")
        for action in report["next_actions"][:8]:
            print(f"- next: {action}")
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

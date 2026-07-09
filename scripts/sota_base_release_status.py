#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


REPOS = Path("/home/mekaneeky/repos")
LOCAL_RUN_DIR = REPOS / ".sota-base-local"
TESTNET_RUN_DIR = REPOS / ".sota-base-testnet"
DEFAULT_SNAPSHOT_DIR = Path("/mnt/4tb/tao_fork_snapshot")
PUBLISHER_TIMERS = {
    "genesis": ("base-sota-genesis-publisher.timer", "base-sota-genesis-publisher.service"),
    "emission": ("base-sota-emission-publisher.timer", "base-sota-emission-publisher.service"),
}


@dataclass(frozen=True)
class GateSpec:
    name: str
    phase: str
    path: Path
    expected_schema: str
    required: bool
    next_action: str


def _load_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(status, 2)


def _worst(statuses: list[str]) -> str:
    if not statuses:
        return "green"
    return max(statuses, key=_status_rank)


def _gate_status(spec: GateSpec) -> dict[str, Any]:
    try:
        report = _load_report(spec.path)
    except Exception as exc:
        return {
            "name": spec.name,
            "phase": spec.phase,
            "required": spec.required,
            "path": str(spec.path),
            "expected_schema": spec.expected_schema,
            "schema": None,
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "message": f"Could not read report: {exc}",
            "next_action": spec.next_action,
        }
    if report is None:
        return {
            "name": spec.name,
            "phase": spec.phase,
            "required": spec.required,
            "path": str(spec.path),
            "expected_schema": spec.expected_schema,
            "schema": None,
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "message": "Report is missing.",
            "next_action": spec.next_action,
        }
    schema = str(report.get("schema") or "")
    schema_ok = schema == spec.expected_schema
    ok = bool(report.get("ok")) and schema_ok
    status = str(report.get("status") or ("green" if ok else "red"))
    if not schema_ok:
        status = "red"
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    next_action = spec.next_action
    report_next_actions = report.get("next_actions")
    if isinstance(report_next_actions, list):
        for item in report_next_actions:
            candidate = str(item or "").strip()
            if candidate:
                next_action = candidate
                break
    return {
        "name": spec.name,
        "phase": spec.phase,
        "required": spec.required,
        "path": str(spec.path),
        "expected_schema": spec.expected_schema,
        "schema": schema or None,
        "ok": ok,
        "status": status,
        "summary": {
            "green": int(summary.get("green") or 0),
            "yellow": int(summary.get("yellow") or 0),
            "red": int(summary.get("red") or 0),
        },
        "message": str(report.get("message") or ""),
        "next_action": "" if ok else next_action,
    }


def _normalize_address(value: Any) -> str:
    return str(value or "").strip().lower()


def _current_claim_seed(testnet_dir: Path) -> dict[str, Any]:
    try:
        report = _load_report(testnet_dir / "base-sota-testnet-seed-artifacts-finalized.json")
    except Exception:
        return {}
    return dict(dict(report or {}).get("seeded_claims") or {})


def _artifact_expectation(path: Path) -> dict[str, set[str]]:
    try:
        artifact = _load_report(path)
    except Exception:
        return {"wallets": set(), "amounts": set()}
    if not artifact:
        return {"wallets": set(), "amounts": set()}
    wallets: set[str] = set()
    amounts: set[str] = set()
    allocations = artifact.get("allocations") if isinstance(artifact.get("allocations"), list) else []
    for allocation in allocations:
        if not isinstance(allocation, dict):
            continue
        wallet = _normalize_address(allocation.get("reward_address") or allocation.get("account"))
        amount = str(allocation.get("amount_units") or allocation.get("amount") or "").strip()
        if wallet:
            wallets.add(wallet)
        if amount:
            amounts.add(amount)
    root = dict(artifact.get("root") or {})
    root_amount = str(root.get("total_amount_units") or root.get("budget") or "").strip()
    if root_amount and not amounts:
        amounts.add(root_amount)
    return {"wallets": wallets, "amounts": amounts}


def _current_claim_expectations(testnet_dir: Path) -> dict[str, set[str]]:
    seed = _current_claim_seed(testnet_dir)
    genesis = _artifact_expectation(testnet_dir / "base-sota-testnet-genesis-claim-artifact.json")
    emission = _artifact_expectation(testnet_dir / "base-sota-testnet-emission-claim-artifact.json")
    wallets = set(genesis["wallets"]) | set(emission["wallets"])
    seed_wallet = _normalize_address(seed.get("test_wallet_address"))
    if seed_wallet and not wallets:
        wallets.add(seed_wallet)
    genesis_amounts = set(genesis["amounts"])
    emission_amounts = set(emission["amounts"])
    seed_genesis = str(seed.get("genesis_total_units") or "").strip()
    seed_emission = str(seed.get("emission_total_units") or "").strip()
    if not genesis_amounts and seed_genesis:
        genesis_amounts.add(seed_genesis)
    if not emission_amounts and seed_emission:
        emission_amounts.add(seed_emission)
    return {
        "wallets": wallets,
        "genesis_amounts": genesis_amounts,
        "emission_amounts": emission_amounts,
    }


def _snapshot_block(snapshot_dir: Path) -> dict[str, str]:
    path = snapshot_dir / "genesis_snapshot_block.json"
    if not path.exists():
        return {}
    try:
        payload = _load_report(path)
    except Exception:
        return {}
    if not payload:
        return {}
    return {
        "number": str(payload.get("bittensor_block_number") or "").strip(),
        "hash": str(payload.get("bittensor_block_hash") or "").strip().lower(),
    }


def _snapshot_alpha_row_count(snapshot_dir: Path) -> int:
    path = snapshot_dir / "alpha_exposures.csv"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _token_from_env(name: str) -> str:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict):
        for key in (name, "admin_token", "token", "SOTA_BASE_INDEXER_ADMIN_TOKEN", "SOTA_INDEXER_ADMIN_TOKEN"):
            value = str(parsed.get(key) or "").strip()
            if value:
                return value
    return raw


def _public_binding_export_evidence(url: str, *, token_env: str, timeout: float) -> dict[str, Any]:
    url = str(url or "").strip()
    if not url:
        return {"status": "not_configured"}
    headers = {"Accept": "application/json"}
    token = _token_from_env(token_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        request = Request(url, headers=headers, method="GET")
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(body) if body.strip() else {}
        bindings = payload.get("bindings")
        count = payload.get("count")
        if isinstance(bindings, list):
            count = len(bindings)
        return {
            "status": "green",
            "url": url,
            "schema": str(payload.get("schema") or ""),
            "accepted_signed_binding_count": int(count or 0),
            "token_env": token_env,
            "used_auth_header": bool(token),
        }
    except HTTPError as exc:
        status = "auth_required" if exc.code in {401, 403} else "red"
        return {
            "status": status,
            "url": url,
            "http_status": exc.code,
            "error": str(exc),
            "token_env": token_env,
            "used_auth_header": bool(token),
        }
    except Exception as exc:
        return {
            "status": "red",
            "url": url,
            "error": str(exc),
            "token_env": token_env,
            "used_auth_header": bool(token),
        }


def _snapshot_binding_evidence(
    testnet_dir: Path,
    *,
    snapshot_claim_bindings_url: str = "",
    indexer_admin_token_env: str = "SOTA_BASE_INDEXER_ADMIN_TOKEN",
    timeout: float = 10.0,
) -> dict[str, Any]:
    claim_dir = testnet_dir / "snapshot-claims"
    accepted_files: list[str] = []
    invalid_files: list[str] = []
    if claim_dir.exists():
        for path in sorted(claim_dir.glob("api-binding-*.json")):
            try:
                payload = _load_report(path) or {}
            except Exception:
                invalid_files.append(str(path))
                continue
            if isinstance(payload.get("message"), dict) and str(payload.get("signature") or "").strip():
                accepted_files.append(str(path))
            else:
                invalid_files.append(str(path))
    pending_requests: list[str] = []
    for path in sorted(testnet_dir.glob("*binding-request*.json")):
        try:
            payload = _load_report(path) or {}
        except Exception:
            continue
        if str(payload.get("schema") or "") == "sota-snapshot-binding-message/v1" and not str(payload.get("signature") or "").strip():
            pending_requests.append(str(path))
    return {
        "snapshot_claim_dir": str(claim_dir),
        "accepted_signed_binding_count": len(accepted_files),
        "accepted_signed_binding_files": accepted_files,
        "invalid_binding_file_count": len(invalid_files),
        "pending_unsigned_binding_request_count": len(pending_requests),
        "pending_unsigned_binding_request_files": pending_requests,
        "public_binding_export": _public_binding_export_evidence(
            snapshot_claim_bindings_url,
            token_env=indexer_admin_token_env,
            timeout=timeout,
        ),
    }


def _browser_smoke_check_statuses(testnet_dir: Path) -> dict[str, str]:
    try:
        report = _load_report(testnet_dir / "base-sota-testnet-browser-smoke.json")
    except Exception:
        return {}
    if not isinstance(report, dict):
        return {}
    return {
        str(check.get("name") or ""): str(check.get("status") or "")
        for check in (report.get("checks") or [])
        if isinstance(check, dict)
    }


def _publisher_report(path: Path, *, expected_schema: str) -> dict[str, Any]:
    try:
        report = _load_report(path)
    except Exception as exc:
        return {"ok": False, "status": "red", "path": str(path), "message": f"Could not read report: {exc}"}
    if report is None:
        return {"ok": False, "status": "red", "path": str(path), "message": "Report is missing."}
    schema = str(report.get("schema") or "")
    ok = bool(report.get("ok")) and schema == expected_schema and str(report.get("status") or "") in {"idle", "published"}
    return {
        "ok": ok,
        "schema": schema or None,
        "status": str(report.get("status") or "red"),
        "path": str(path),
        "message": str(report.get("message") or ""),
    }


def _systemctl_show(unit: str, properties: list[str], *, timeout: float) -> dict[str, str]:
    cmd = ["systemctl", "--user", "show", unit, "--no-pager"]
    for prop in properties:
        cmd.extend(["-p", prop])
    values: dict[str, str] = {}
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)
    except Exception as exc:
        values["__error__"] = str(exc)
        return values
    if result.returncode != 0:
        values["__error__"] = (result.stderr or result.stdout or f"systemctl exited {result.returncode}").strip()
        return values
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _publisher_timers_gate(*, timeout: float) -> dict[str, Any]:
    timer_statuses: dict[str, Any] = {}
    reasons: list[str] = []
    for label, (timer_unit, service_unit) in PUBLISHER_TIMERS.items():
        timer = _systemctl_show(
            timer_unit,
            ["ActiveState", "SubState", "UnitFileState"],
            timeout=timeout,
        )
        service = _systemctl_show(
            service_unit,
            ["ActiveState", "SubState", "Result", "ExecMainStatus"],
            timeout=timeout,
        )
        active = timer.get("ActiveState") == "active"
        enabled = timer.get("UnitFileState") == "enabled"
        service_success = service.get("Result") in {"success", ""} and service.get("ExecMainStatus") in {"0", ""}
        ok = active and enabled and service_success and "__error__" not in timer and "__error__" not in service
        if not ok:
            detail_parts = []
            if timer.get("__error__"):
                detail_parts.append(f"timer error: {timer['__error__']}")
            if service.get("__error__"):
                detail_parts.append(f"service error: {service['__error__']}")
            if not active:
                detail_parts.append(f"timer ActiveState={timer.get('ActiveState') or 'missing'}")
            if not enabled:
                detail_parts.append(f"timer UnitFileState={timer.get('UnitFileState') or 'missing'}")
            if not service_success:
                detail_parts.append(
                    f"service Result={service.get('Result') or 'missing'} ExecMainStatus={service.get('ExecMainStatus') or 'missing'}"
                )
            reasons.append(f"{label}: " + "; ".join(detail_parts))
        timer_statuses[label] = {
            "ok": ok,
            "timer_unit": timer_unit,
            "service_unit": service_unit,
            "timer": timer,
            "service": service,
        }
    ok = not reasons
    return {
        "name": "testnet_publisher_timers",
        "phase": "base_sepolia",
        "required": True,
        "path": "systemctl --user",
        "expected_schema": "systemd-user-timers",
        "schema": "systemd-user-timers",
        "ok": ok,
        "status": "green" if ok else "red",
        "summary": {"green": 1 if ok else 0, "yellow": 0, "red": 0 if ok else 1},
        "message": (
            "Genesis and emission publisher timers are active, enabled, and last service result is success."
            if ok
            else "; ".join(reasons)
        ),
        "next_action": (
            ""
            if ok
            else "Install/enable ops/systemd/base-sota-*-publisher.timer under the user systemd instance and confirm the last service result is success."
        ),
        "timers": timer_statuses,
    }


def _deferred_holder_readiness(
    *,
    testnet_dir: Path,
    block: dict[str, str],
    alpha_rows: int,
    binding_evidence: dict[str, Any],
) -> dict[str, Any]:
    public_export = dict(binding_evidence.get("public_binding_export") or {})
    browser_checks = _browser_smoke_check_statuses(testnet_dir)
    genesis_publisher = _publisher_report(
        testnet_dir / "base-sota-genesis-batch-publisher.json",
        expected_schema="sota-base-genesis-batch-publisher/v1",
    )
    required_browser = {
        "claims_binding_frontend",
        "genesis_binding_message",
        "genesis_binding_submit_route",
    }
    reasons: list[str] = []
    if not block.get("number") or not block.get("hash"):
        reasons.append("snapshot block lock is missing")
    if alpha_rows <= 0:
        reasons.append("alpha_exposures.csv is missing or empty")
    missing_browser = sorted(name for name in required_browser if browser_checks.get(name) != "green")
    if missing_browser:
        reasons.append("browser smoke does not prove holder binding controls: " + ", ".join(missing_browser))
    if public_export.get("status") not in {"green", "not_configured"}:
        reasons.append("public binding export is not readable")
    if not bool(genesis_publisher.get("ok")):
        reasons.append("genesis batch publisher is not ready: " + str(genesis_publisher.get("message") or genesis_publisher.get("status")))
    return {
        "ok": not reasons,
        "status": "green" if not reasons else "red",
        "message": "; ".join(reasons),
        "browser_binding_checks": {name: browser_checks.get(name) for name in sorted(required_browser)},
        "public_binding_export": public_export,
        "genesis_batch_publisher": genesis_publisher,
    }


def _snapshot_genesis_gate(testnet_dir: Path, snapshot_dir: Path, args: argparse.Namespace | None = None) -> dict[str, Any]:
    artifact_path = testnet_dir / "base-sota-testnet-genesis-claim-artifact.json"
    binding_evidence = _snapshot_binding_evidence(
        testnet_dir,
        snapshot_claim_bindings_url=str(getattr(args, "snapshot_claim_bindings_url", "") or ""),
        indexer_admin_token_env=str(getattr(args, "indexer_admin_token_env", "SOTA_BASE_INDEXER_ADMIN_TOKEN") or "SOTA_BASE_INDEXER_ADMIN_TOKEN"),
        timeout=float(getattr(args, "timeout", 10.0) or 10.0),
    )
    base = {
        "name": "testnet_snapshot_genesis",
        "phase": "base_sepolia",
        "required": True,
        "path": str(artifact_path),
        "expected_schema": "sota-base-claim-artifact/v1",
        "next_action": (
            "Have a snapshot holder submit a signed coldkey binding through the claims UI Genesis binding panel "
            "or scripts/sota_sign_snapshot_binding.py, then rerun "
            "scripts/sota_base_testnet_operator.py "
            f"with --snapshot-dir {snapshot_dir} and --snapshot-claim-bindings-url "
            '"$SOTA_CLAIMS_API_URL/api/v1/base/genesis/bindings"; alternatively pass '
            "--snapshot-claim-binding <binding>."
        ),
    }
    try:
        artifact = _load_report(artifact_path)
    except Exception as exc:
        return {
            **base,
            "schema": None,
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "message": f"Could not read genesis claim artifact: {exc}",
            "snapshot_binding_evidence": binding_evidence,
        }
    if artifact is None:
        block = _snapshot_block(snapshot_dir)
        alpha_rows = _snapshot_alpha_row_count(snapshot_dir)
        if bool(getattr(args, "defer_real_holder_test", False)):
            public_export = dict(binding_evidence.get("public_binding_export") or {})
            public_count = int(public_export.get("accepted_signed_binding_count") or 0) if public_export.get("status") == "green" else 0
            accepted_count = int(binding_evidence["accepted_signed_binding_count"]) + public_count
            deferred = _deferred_holder_readiness(
                testnet_dir=testnet_dir,
                block=block,
                alpha_rows=alpha_rows,
                binding_evidence=binding_evidence,
            )
            if accepted_count == 0 and deferred["ok"]:
                return {
                    **base,
                    "schema": "sota-base-claim-artifact/v1",
                    "ok": True,
                    "status": "green",
                    "summary": {"green": 1, "yellow": 0, "red": 0},
                    "message": (
                        "Real holder claim is deferred by operator request. No snapshot genesis claim artifact exists yet "
                        "because no holder binding has been accepted, but the public binding path and genesis publisher are ready."
                    ),
                    "next_action": "",
                    "holder_test_deferred": True,
                    "strict_reasons_without_defer": ["Genesis claim artifact is missing."],
                    "snapshot_source": {
                        "path": str(snapshot_dir),
                        "block_number": block["number"],
                        "block_hash": block["hash"],
                        "alpha_rows": alpha_rows,
                    },
                    "snapshot_binding_evidence": binding_evidence,
                    "deferred_holder_readiness": deferred,
                }
        return {
            **base,
            "schema": None,
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "message": "Genesis claim artifact is missing.",
            "snapshot_binding_evidence": binding_evidence,
        }
    schema = str(artifact.get("schema") or "")
    snapshot = dict(artifact.get("snapshot") or {})
    root = dict(artifact.get("root") or {})
    allocations = artifact.get("allocations") if isinstance(artifact.get("allocations"), list) else []
    block = _snapshot_block(snapshot_dir)
    alpha_rows = _snapshot_alpha_row_count(snapshot_dir)
    reasons: list[str] = []
    if schema != "sota-base-claim-artifact/v1":
        reasons.append(f"schema is {schema or 'missing'}")
    if not root.get("root_id"):
        reasons.append("root_id is missing")
    if str(root.get("subnet_id") or "") != "genesis":
        reasons.append("root subnet_id is not genesis")
    if not snapshot:
        reasons.append("snapshot metadata is missing")
    if block:
        observed_number = str(snapshot.get("bittensor_block_number") or "").strip()
        observed_hash = str(snapshot.get("bittensor_block_hash") or "").strip().lower()
        if observed_number != block["number"] or observed_hash != block["hash"]:
            reasons.append(f"snapshot block does not match {snapshot_dir}")
    else:
        reasons.append("snapshot block lock is missing")
    if alpha_rows <= 0:
        reasons.append("alpha_exposures.csv is missing or empty")
    if not allocations:
        reasons.append("allocations are missing")
    missing_credit_fields = [
        index
        for index, allocation in enumerate(allocations)
        if not isinstance(allocation, dict)
        or "tao_credit_rao" not in allocation
        or "alpha_synthetic_credit_rao" not in allocation
        or "alpha_credit_rao_by_netuid" not in allocation
    ]
    if missing_credit_fields:
        reasons.append("allocations are missing TAO/alpha rao credit fields from the snapshot bridge")
    if reasons and int(binding_evidence["accepted_signed_binding_count"]) <= 0:
        suffix = ""
        if int(binding_evidence["pending_unsigned_binding_request_count"]) > 0:
            suffix = "; pending unsigned binding request exists, but it is not an accepted signed binding"
        public_export = dict(binding_evidence.get("public_binding_export") or {})
        public_count = public_export.get("accepted_signed_binding_count")
        if public_export.get("status") == "green":
            suffix += f"; public claims API accepted binding count is {int(public_count or 0)}"
        elif public_export.get("status") == "auth_required":
            suffix += f"; claims API binding export requires {public_export.get('token_env') or 'admin token'}"
        elif public_export.get("status") == "red":
            suffix += "; public claims API binding count could not be read"
        reasons.insert(0, f"accepted signed snapshot binding count is 0{suffix}")
    if reasons and bool(getattr(args, "defer_real_holder_test", False)):
        public_export = dict(binding_evidence.get("public_binding_export") or {})
        public_count = int(public_export.get("accepted_signed_binding_count") or 0) if public_export.get("status") == "green" else 0
        accepted_count = int(binding_evidence["accepted_signed_binding_count"]) + public_count
        if accepted_count == 0:
            deferred = _deferred_holder_readiness(
                testnet_dir=testnet_dir,
                block=block,
                alpha_rows=alpha_rows,
                binding_evidence=binding_evidence,
            )
            if deferred["ok"]:
                return {
                    **base,
                    "schema": schema or "sota-base-claim-artifact/v1",
                    "ok": True,
                    "status": "green",
                    "summary": {"green": 1, "yellow": 0, "red": 0},
                    "message": (
                        "Real holder claim is deferred by operator request. The public claims UI/API can build "
                        "snapshot binding payloads from the locked TAO plus alpha snapshot, rejects invalid "
                        "signatures, and the genesis batch publisher is ready to publish the first accepted binding."
                    ),
                    "next_action": "",
                    "holder_test_deferred": True,
                    "strict_reasons_without_defer": reasons,
                    "snapshot_source": {
                        "path": str(snapshot_dir),
                        "block_number": block["number"],
                        "block_hash": block["hash"],
                        "alpha_rows": alpha_rows,
                    },
                    "snapshot_binding_evidence": binding_evidence,
                    "deferred_holder_readiness": deferred,
                }
    try:
        allocation_total = sum(int(dict(row).get("amount_units") or dict(row).get("amount") or 0) for row in allocations)
        root_total = int(root.get("total_amount_units") or root.get("budget") or 0)
        if allocation_total <= 0 or root_total != allocation_total:
            reasons.append("root total does not equal allocation total")
    except Exception:
        reasons.append("allocation totals could not be verified")
    if reasons:
        return {
            **base,
            "schema": schema or None,
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "message": "; ".join(reasons),
            "snapshot_source": {
                "path": str(snapshot_dir),
                "block_number": block.get("number"),
                "block_hash": block.get("hash"),
                "alpha_rows": alpha_rows,
            },
            "snapshot_binding_evidence": binding_evidence,
        }
    return {
        **base,
        "schema": schema,
        "ok": True,
        "status": "green",
        "summary": {"green": 1, "yellow": 0, "red": 0},
        "message": (
            "Base Sepolia genesis claim artifact is built from the locked TAO plus alpha snapshot "
            f"at block {block['number']} with {alpha_rows} alpha exposure rows."
        ),
        "next_action": "",
        "snapshot_source": {
            "path": str(snapshot_dir),
            "block_number": block["number"],
            "block_hash": block["hash"],
            "alpha_rows": alpha_rows,
        },
        "snapshot_binding_evidence": binding_evidence,
    }


def _operator_gate_with_deferred_holder_test(
    testnet_dir: Path,
    gate: dict[str, Any],
    gate_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    if gate.get("name") != "testnet_operator_run" or gate.get("status") != "red":
        return gate
    snapshot_gate = next((item for item in gate_reports if item.get("name") == "testnet_snapshot_genesis"), {})
    if not snapshot_gate.get("holder_test_deferred"):
        return gate
    green_names = {
        item.get("name")
        for item in gate_reports
        if item.get("status") == "green"
    }
    if "testnet_blockers" not in green_names or "testnet_browser_smoke" not in green_names:
        return gate
    timer_gate = next((item for item in gate_reports if item.get("name") == "testnet_publisher_timers"), None)
    if timer_gate is not None and not bool(timer_gate.get("ok")):
        return gate
    try:
        report = _load_report(testnet_dir / "base-sota-testnet-operator-run.json") or {}
    except Exception:
        return gate
    steps = [dict(item) for item in (report.get("steps") or report.get("checks") or []) if isinstance(item, dict)]
    allowed_red = {"snapshot_binding_export", "snapshot_genesis_artifacts", "publish_genesis_root", "browser_smoke"}
    allowed_yellow = {"finalize_claim_artifacts", "import_claim_artifacts"}
    unexpected = [
        str(step.get("name") or "")
        for step in steps
        if (
            (step.get("status") == "red" and str(step.get("name") or "") not in allowed_red)
            or (step.get("status") == "yellow" and str(step.get("name") or "") not in allowed_yellow)
        )
    ]
    if unexpected:
        return gate
    genesis_publisher = _publisher_report(
        testnet_dir / "base-sota-genesis-batch-publisher.json",
        expected_schema="sota-base-genesis-batch-publisher/v1",
    )
    emission_publisher = _publisher_report(
        testnet_dir / "base-sota-emission-batch-publisher.json",
        expected_schema="sota-base-emission-batch-publisher/v1",
    )
    if not genesis_publisher.get("ok") or not emission_publisher.get("ok"):
        return gate
    updated = dict(gate)
    updated.update(
        {
            "ok": True,
            "status": "green",
            "summary": {"green": 1, "yellow": 0, "red": 0},
            "message": (
                "Strict operator run is red only because no real holder binding was submitted. "
                "Real holder testing is deferred; scheduled genesis and emission publishers are active and idle/ready."
            ),
            "next_action": "",
            "holder_test_deferred": True,
            "deferred_publishers": {
                "genesis": genesis_publisher,
                "emission": emission_publisher,
            },
        }
    )
    return updated


def _claim_tx_evidence_current_gate(testnet_dir: Path, gate: dict[str, Any]) -> dict[str, Any]:
    if gate.get("name") != "claim_tx_evidence" or gate.get("status") != "green":
        return gate
    expected = _current_claim_expectations(testnet_dir)
    expected_wallets = set(expected["wallets"])
    expected_genesis_amounts = set(expected["genesis_amounts"])
    expected_emission_amounts = set(expected["emission_amounts"])
    try:
        report = _load_report(Path(str(gate.get("path") or "")))
    except Exception:
        report = None
    config = dict(dict(report or {}).get("config") or {})
    transactions = dict(dict(report or {}).get("transactions") or {})
    genesis = dict(transactions.get("genesis") or {})
    emission = dict(transactions.get("emission") or {})
    observed_wallets = {
        _normalize_address(config.get("wallet_address")),
        _normalize_address(genesis.get("from")),
        _normalize_address(emission.get("from")),
    }
    observed_wallets.discard("")
    reasons: list[str] = []
    if expected_wallets and observed_wallets and not observed_wallets.issubset(expected_wallets):
        reasons.append(
            "claim transaction evidence is for "
            + ", ".join(sorted(observed_wallets))
            + ", but current claim artifacts allow "
            + ", ".join(sorted(expected_wallets))
        )
    if expected_genesis_amounts and str(genesis.get("claim_amount_raw") or "").strip() not in expected_genesis_amounts:
        reasons.append("genesis claim amount does not match the current finalized genesis artifact")
    if expected_emission_amounts and str(emission.get("claim_amount_raw") or "").strip() not in expected_emission_amounts:
        reasons.append("emission claim amount does not match the current finalized emission artifact")
    if not reasons:
        return gate
    updated = dict(gate)
    updated.update(
        {
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "message": "; ".join(reasons),
            "next_action": "Submit both claims from the current Base Sepolia tester wallet, then rerun scripts/sota_base_claim_tx_evidence.py with the new tx hashes.",
        }
    )
    return updated


def _browser_smoke_current_gate(testnet_dir: Path, gate: dict[str, Any]) -> dict[str, Any]:
    if gate.get("name") != "testnet_browser_smoke" or gate.get("status") == "red":
        return gate
    path = testnet_dir / "base-sota-testnet-browser-smoke.json"
    try:
        report = _load_report(path)
    except Exception:
        return gate
    checks = {
        str(check.get("name") or ""): str(check.get("status") or "")
        for check in (dict(report or {}).get("checks") or [])
        if isinstance(check, dict)
    }
    required = {
        "claims_page_text",
        "genesis_binding_message",
        "genesis_binding_submit_route",
        "genesis_lookup",
        "emission_lookup",
        "genesis_calldata",
        "emission_calldata",
        "self_validation_evidence",
    }
    missing = sorted(required - set(checks))
    failed = sorted(name for name in required if checks.get(name) not in {"green", "yellow"} and name not in missing)
    if not missing and not failed:
        return gate
    reasons: list[str] = []
    if missing:
        reasons.append("missing current browser-smoke checks: " + ", ".join(missing))
    if failed:
        reasons.append("failed current browser-smoke checks: " + ", ".join(failed))
    out = dict(gate)
    out["ok"] = False
    out["status"] = "red"
    out["summary"] = {"green": 0, "yellow": 0, "red": 1}
    out["message"] = "; ".join(reasons)
    out["next_action"] = "Deploy current claims UI/API and rerun scripts/sota_base_testnet_browser_smoke.py."
    return out


def _local_wallet_status(local_report: Path) -> dict[str, Any]:
    try:
        report = _load_report(local_report)
    except Exception as exc:
        return {
            "ok": False,
            "status": "red",
            "message": f"Could not read local UI smoke report: {exc}",
            "next_action": "Run ./scripts/sota_local_demo.py launch and regenerate the local UI smoke report.",
        }
    if report is None:
        return {
            "ok": False,
            "status": "red",
            "message": "Local UI smoke report is missing.",
            "next_action": "Run ./scripts/sota_local_demo.py launch and regenerate the local UI smoke report.",
        }
    for check in report.get("checks") or []:
        if not isinstance(check, dict) or check.get("name") != "tester_wallet_rpc":
            continue
        status = str(check.get("status") or "red")
        return {
            "ok": status == "green",
            "status": status,
            "message": str(check.get("detail") or ""),
            "next_action": "" if status == "green" else str(check.get("remediation") or "Relaunch with --share-mode tailscale-https for remote MetaMask testing."),
        }
    return {
        "ok": False,
        "status": "red",
        "message": "Local UI smoke did not report tester wallet RPC readiness.",
        "next_action": "Rerun ./scripts/sota_local_demo.py ui-smoke so the tester_wallet_rpc check is present.",
    }


def _local_miner_swarm_gate(path: Path, *, min_miners: int) -> dict[str, Any]:
    base = {
        "name": "local_miner_swarm",
        "phase": "local",
        "required": True,
        "path": str(path),
        "expected_schema": "sota-local-multi-miner/v1",
        "next_action": f"Run ./scripts/sota_local_demo.py swarm-smoke --count {min_miners}.",
    }
    try:
        report = _load_report(path)
    except Exception as exc:
        return {
            **base,
            "schema": None,
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "message": f"Could not read local miner swarm report: {exc}",
        }
    if report is None:
        return {
            **base,
            "schema": None,
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "message": "Local miner swarm report is missing.",
        }
    schema = str(report.get("schema") or "")
    checks = dict(report.get("checks") or {})
    miners = report.get("miners") if isinstance(report.get("miners"), list) else []
    claim_transactions = report.get("claim_transactions") if isinstance(report.get("claim_transactions"), list) else []
    reasons: list[str] = []
    miner_count = int(report.get("miner_count") or 0)
    accepted_count = int(report.get("accepted_count") or 0)
    matching_claim_count = int(report.get("matching_claim_count") or 0)
    if schema != "sota-local-multi-miner/v1":
        reasons.append(f"schema is {schema or 'missing'}")
    if not bool(report.get("ok")):
        reasons.append("report ok is false")
    if miner_count < min_miners:
        reasons.append(f"miner_count {miner_count} is below required {min_miners}")
    if len(miners) < min_miners:
        reasons.append(f"miners list has {len(miners)} entries")
    if accepted_count < min_miners:
        reasons.append(f"accepted_count {accepted_count} is below required {min_miners}")
    if matching_claim_count < min_miners:
        reasons.append(f"matching_claim_count {matching_claim_count} is below required {min_miners}")
    if len(claim_transactions) < min_miners:
        reasons.append(f"claim_transactions has {len(claim_transactions)} entries")
    required_checks = {
        "distinct_hotkeys",
        "distinct_miner_addresses",
        "distinct_reward_addresses",
        "all_processes_exited_zero",
        "all_self_validation_accepted",
        "all_claims_submitted",
    }
    failed_checks = sorted(name for name in required_checks if not bool(checks.get(name)))
    if failed_checks:
        reasons.append("failed checks: " + ", ".join(failed_checks))
    if reasons:
        return {
            **base,
            "schema": schema or None,
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "message": "; ".join(reasons),
        }
    return {
        **base,
        "schema": schema,
        "ok": True,
        "status": "green",
        "summary": {"green": 1, "yellow": 0, "red": 0},
        "message": (
            f"Local miner swarm ran {miner_count} distinct miners, accepted {accepted_count} self-validated submissions, "
            f"published {matching_claim_count} matching leaves, and submitted {len(claim_transactions)} claim transactions."
        ),
        "next_action": "",
    }


def _local_remote_wallet_status(tailscale_preflight: dict[str, Any]) -> dict[str, Any]:
    status = str(tailscale_preflight.get("status") or "missing")
    ok = bool(tailscale_preflight.get("ok"))
    next_actions = [str(item) for item in tailscale_preflight.get("next_actions") or [] if str(item)]
    if ok:
        return {
            "ok": True,
            "status": "green",
            "message": "Tailscale HTTPS sharing is ready for remote MetaMask testing.",
            "next_action": "",
        }
    return {
        "ok": False,
        "status": "red" if status in {"missing", "red"} else status,
        "message": str(tailscale_preflight.get("message") or "Tailscale HTTPS sharing is not ready for remote MetaMask testing."),
        "next_action": next_actions[0] if next_actions else "Enable Tailscale Serve/HTTPS, then relaunch with --share-mode tailscale-https.",
    }


def _optional_report_status(path: Path, *, expected_schema: str) -> dict[str, Any]:
    try:
        report = _load_report(path)
    except Exception as exc:
        return {
            "path": str(path),
            "schema": None,
            "ok": False,
            "status": "red",
            "message": f"Could not read report: {exc}",
            "summary": {"green": 0, "yellow": 0, "red": 1},
        }
    if report is None:
        return {
            "path": str(path),
            "schema": None,
            "ok": False,
            "status": "missing",
            "message": "Report is missing.",
            "summary": {"green": 0, "yellow": 0, "red": 0},
        }
    schema = str(report.get("schema") or "")
    schema_ok = schema == expected_schema
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    return {
        "path": str(path),
        "schema": schema or None,
        "ok": bool(report.get("ok")) and schema_ok,
        "status": "red" if not schema_ok else str(report.get("status") or "red"),
        "message": str(report.get("message") or ""),
        "summary": {
            "green": int(summary.get("green") or 0),
            "yellow": int(summary.get("yellow") or 0),
            "red": int(summary.get("red") or 0),
        },
        "next_actions": report.get("next_actions") if isinstance(report.get("next_actions"), list) else [],
    }


def default_gates(*, local_report: Path, local_claim_proof: Path, testnet_dir: Path, include_testnet: bool) -> list[GateSpec]:
    gates = [
        GateSpec(
            name="local_demo",
            phase="local",
            path=local_report,
            expected_schema="sota-local-claims-ui-smoke/v1",
            required=True,
            next_action="Run ./scripts/sota_local_demo.py launch, then ./scripts/sota_local_demo.py ui-smoke --skip-screenshot.",
        ),
        GateSpec(
            name="local_claim_proof",
            phase="local",
            path=local_claim_proof,
            expected_schema="sota-local-claim-proof/v1",
            required=True,
            next_action="Run ./scripts/sota_local_demo.py launch so it records the local claim proof and resets the stack for the tester.",
        ),
    ]
    if include_testnet:
        gates.extend(
            [
                GateSpec(
                    name="testnet_operator_run",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-testnet-operator-run.json",
                    expected_schema="sota-base-testnet-operator-run/v1",
                    required=True,
                    next_action="Run scripts/sota_base_testnet_operator.py after Base Sepolia deployment inputs and evidence are available.",
                ),
                GateSpec(
                    name="testnet_blockers",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-testnet-blockers.json",
                    expected_schema="sota-base-testnet-blockers/v1",
                    required=True,
                    next_action="Clear AWS/DNS/artifact blockers, then rerun scripts/sota_base_testnet_blockers.py.",
                ),
                GateSpec(
                    name="testnet_emission_lane_sync",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-testnet-emission-lane-sync.json",
                    expected_schema="sota-base-lane-sync/v1",
                    required=True,
                    next_action="Run scripts/sota_base_sync_lane.py and broadcast with the deployer before public emission claims.",
                ),
                GateSpec(
                    name="testnet_aws_inventory",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-testnet-aws-inventory.json",
                    expected_schema="sota-base-testnet-aws-inventory/v1",
                    required=True,
                    next_action="Create or document Base SOTA testnet AWS resources, then rerun scripts/sota_base_testnet_aws_inventory.py.",
                ),
                GateSpec(
                    name="testnet_funding",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-testnet-funding.json",
                    expected_schema="sota-base-testnet-funding/v1",
                    required=True,
                    next_action="Fund the deployer, root publisher, and test wallet with Base Sepolia ETH, then rerun scripts/sota_base_testnet_funding.py.",
                ),
                GateSpec(
                    name="testnet_secret_handles",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-testnet-secret-handles.json",
                    expected_schema="sota-base-testnet-secret-bootstrap/v1",
                    required=True,
                    next_action="Run scripts/sota_base_testnet_secrets.py create, then add the real autoresearch database secret handle.",
                ),
                GateSpec(
                    name="testnet_apprunner_source_pack",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-testnet-apprunner-source-pack.json",
                    expected_schema="sota-base-testnet-apprunner-source-pack/v1",
                    required=True,
                    next_action="Run scripts/sota_base_testnet_apprunner_source_pack.py after the Base SOTA service branches are pushed and App Runner connection/instance roles are resolved.",
                ),
                GateSpec(
                    name="testnet_container_pack",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-testnet-container-pack.json",
                    expected_schema="sota-base-testnet-container-pack/v1",
                    required=False,
                    next_action="Optional ECR path: rerun scripts/sota_base_testnet_container_pack.py with an App Runner ECR access role ARN.",
                ),
                GateSpec(
                    name="testnet_browser_smoke",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-testnet-browser-smoke.json",
                    expected_schema="sota-base-testnet-browser-smoke/v1",
                    required=True,
                    next_action="Deploy public testnet services/artifacts, then rerun scripts/sota_base_testnet_browser_smoke.py.",
                ),
                GateSpec(
                    name="testnet_fresh_emission_tester",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-fresh-emission-tester.json",
                    expected_schema="sota-base-fresh-emission-tester/v1",
                    required=True,
                    next_action="Run scripts/sota_prepare_fresh_testnet_emission_tester.py so a fresh funded wallet has a real self-validated emission claim and unsigned MetaMask calldata.",
                ),
                GateSpec(
                    name="claim_tx_evidence",
                    phase="base_sepolia",
                    path=testnet_dir / "base-sota-claim-tx-evidence.json",
                    expected_schema="sota-base-claim-tx-evidence/v1",
                    required=True,
                    next_action="After browser-smoke is green, submit both MetaMask claims and rerun scripts/sota_base_claim_tx_evidence.py with tx hashes.",
                ),
            ]
        )
    return gates


def run_status(args: argparse.Namespace) -> dict[str, Any]:
    gates = default_gates(
        local_report=args.local_report,
        local_claim_proof=args.local_claim_proof,
        testnet_dir=args.testnet_artifacts_dir,
        include_testnet=not args.local_only,
    )
    gate_reports = [_gate_status(gate) for gate in gates]
    local_insert_at = next(
        (
            index + 1
            for index, gate in enumerate(gate_reports)
            if gate["name"] == "local_claim_proof"
        ),
        len(gate_reports),
    )
    gate_reports.insert(
        local_insert_at,
        _local_miner_swarm_gate(args.local_miner_swarm, min_miners=args.min_local_miners),
    )
    local_tailscale_preflight = _optional_report_status(
        args.local_tailscale_preflight,
        expected_schema="sota-local-tailscale-preflight/v1",
    )
    local_wallet = _local_wallet_status(args.local_report)
    local_remote_wallet = _local_remote_wallet_status(local_tailscale_preflight)
    if not args.local_only:
        testnet_insert_at = next(
            (
                index + 1
                for index, gate in enumerate(gate_reports)
                if gate["name"] == "testnet_operator_run"
            ),
            len(gate_reports),
        )
        gate_reports.insert(
            testnet_insert_at,
            _snapshot_genesis_gate(args.testnet_artifacts_dir, args.snapshot_dir, args),
        )
        if bool(getattr(args, "check_publisher_timers", False)):
            timer_insert_at = next(
                (
                    index + 1
                    for index, gate in enumerate(gate_reports)
                    if gate["name"] == "testnet_snapshot_genesis"
                ),
                len(gate_reports),
            )
            gate_reports.insert(
                timer_insert_at,
                _publisher_timers_gate(timeout=float(getattr(args, "timer_check_timeout", 5.0) or 5.0)),
            )
        gate_reports = [
            _claim_tx_evidence_current_gate(args.testnet_artifacts_dir, gate)
            for gate in gate_reports
        ]
        gate_reports = [
            _browser_smoke_current_gate(args.testnet_artifacts_dir, gate)
            for gate in gate_reports
        ]
        if bool(getattr(args, "defer_real_holder_test", False)):
            gate_reports = [
                _operator_gate_with_deferred_holder_test(args.testnet_artifacts_dir, gate, gate_reports)
                for gate in gate_reports
            ]
    local_wallet_gate = {
        "name": "local_wallet",
        "phase": "local",
        "required": True,
        "path": str(args.local_report),
        "expected_schema": "sota-local-claims-ui-smoke/v1",
        "schema": "sota-local-claims-ui-smoke/v1",
        "ok": bool(local_wallet.get("ok")),
        "status": str(local_wallet.get("status") or "red"),
        "summary": {
            "green": 1 if local_wallet.get("status") == "green" else 0,
            "yellow": 1 if local_wallet.get("status") == "yellow" else 0,
            "red": 1 if local_wallet.get("status") not in {"green", "yellow"} else 0,
        },
        "message": str(local_wallet.get("message") or ""),
        "next_action": str(local_wallet.get("next_action") or ""),
    }
    insert_at = next((index for index, gate in enumerate(gate_reports) if gate["phase"] != "local"), len(gate_reports))
    gate_reports.insert(insert_at, local_wallet_gate)
    required = [gate for gate in gate_reports if gate["required"]]
    ok = all(bool(gate["ok"]) for gate in required)
    status = _worst([str(gate["status"]) for gate in required])
    blocked = [gate for gate in required if not gate["ok"]]
    local_stack_ok = all(
        bool(gate["ok"])
        for gate in gate_reports
        if gate["phase"] == "local" and gate["required"] and gate["name"] != "local_wallet"
    )
    local_ok = local_stack_ok and bool(local_wallet.get("ok"))
    testnet_ok = (
        all(bool(gate["ok"]) for gate in gate_reports if gate["phase"] == "base_sepolia" and gate["required"])
        if not args.local_only
        else None
    )
    return {
        "schema": "sota-base-release-status/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "status": status,
        "local_stack_ok": local_stack_ok,
        "local_ok": local_ok,
        "local_wallet_ok": bool(local_wallet.get("ok")),
        "local_wallet": local_wallet,
        "local_remote_wallet_ok": bool(local_remote_wallet.get("ok")),
        "local_remote_wallet": local_remote_wallet,
        "local_tailscale_preflight": local_tailscale_preflight,
        "testnet_ok": testnet_ok,
        "real_holder_test_deferred": bool(getattr(args, "defer_real_holder_test", False)),
        "message": (
            "Local and Base Sepolia infrastructure gates are green; real holder claim testing is deferred."
            if ok and bool(getattr(args, "defer_real_holder_test", False)) and not args.local_only
            else
            "Local and Base Sepolia gates are green."
            if ok
            else "Local and/or Base Sepolia gates are not complete."
        ),
        "gates": gate_reports,
        "blocked_gates": [
            {
                "name": gate["name"],
                "phase": gate["phase"],
                "status": gate["status"],
                "path": gate["path"],
                "message": gate.get("message") or "",
                "next_action": gate["next_action"],
            }
            for gate in blocked
        ],
        "summary": {
            "green": sum(1 for gate in gate_reports if gate["status"] == "green"),
            "yellow": sum(1 for gate in gate_reports if gate["status"] == "yellow"),
            "red": sum(1 for gate in gate_reports if gate["status"] == "red"),
        },
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"SOTA Base release status: {report['status'].upper()}")
    print(report["message"])
    remote = dict(report.get("local_remote_wallet") or {})
    wallet = dict(report.get("local_wallet") or {})
    if wallet:
        print(
            "Local MetaMask: "
            f"{wallet.get('status', 'unknown')} - {wallet.get('message', '')}"
        )
    if remote:
        print(
            "Remote Tailscale MetaMask: "
            f"{remote.get('status', 'unknown')} - {remote.get('message', '')}"
        )
    print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
    for gate in report["gates"]:
        summary = gate.get("summary") if isinstance(gate.get("summary"), dict) else {}
        counts = f"{summary.get('green', 0)} green, {summary.get('yellow', 0)} yellow, {summary.get('red', 0)} red"
        print(f"- [{gate['status']}] {gate['name']} ({gate['phase']}): {counts}")
        if gate.get("message"):
            print(f"  {gate['message']}")
        if gate.get("next_action"):
            print(f"  next: {gate['next_action']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Base SOTA local and Base Sepolia release-gate reports.")
    parser.add_argument("--local-report", type=Path, default=LOCAL_RUN_DIR / "ui-smoke" / "report.json")
    parser.add_argument("--local-claim-proof", type=Path, default=LOCAL_RUN_DIR / "claim-proof" / "latest.json")
    parser.add_argument("--local-miner-swarm", type=Path, default=LOCAL_RUN_DIR / "miner-swarm" / "latest.json")
    parser.add_argument("--min-local-miners", type=int, default=3)
    parser.add_argument("--local-tailscale-preflight", type=Path, default=LOCAL_RUN_DIR / "tailscale-preflight.json")
    parser.add_argument("--testnet-artifacts-dir", type=Path, default=TESTNET_RUN_DIR)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--snapshot-claim-bindings-url", default="", help="Optional claims API export URL for accepted signed snapshot bindings.")
    parser.add_argument("--indexer-admin-token-env", default="SOTA_BASE_INDEXER_ADMIN_TOKEN", help="Environment variable containing the claims API admin token or JSON secret payload.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--timer-check-timeout", type=float, default=5.0)
    parser.add_argument("--local-only", action="store_true", help="Only require the local demo gate.")
    parser.add_argument(
        "--skip-publisher-timer-check",
        dest="check_publisher_timers",
        action="store_false",
        help="Do not require live user-systemd publisher timers in the aggregate testnet status.",
    )
    parser.set_defaults(check_publisher_timers=True)
    parser.add_argument(
        "--defer-real-holder-test",
        action="store_true",
        help="Do not require an accepted real snapshot holder binding/root; require the public self-serve holder binding path and scheduled publishers instead.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when required gates are red.")
    args = parser.parse_args(argv)
    report = run_status(args)
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


REPOS = Path("/home/mekaneeky/repos")
LOCAL_RUN_DIR = REPOS / ".sota-base-local"
TESTNET_RUN_DIR = REPOS / ".sota-base-testnet"


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
        "next_action": "" if ok else spec.next_action,
    }


def _local_remote_wallet_status(local_report: Path) -> dict[str, Any]:
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
    required = [gate for gate in gate_reports if gate["required"]]
    ok = all(bool(gate["ok"]) for gate in required)
    status = _worst([str(gate["status"]) for gate in required])
    blocked = [gate for gate in required if not gate["ok"]]
    local_ok = all(bool(gate["ok"]) for gate in gate_reports if gate["phase"] == "local" and gate["required"])
    local_remote_wallet = _local_remote_wallet_status(args.local_report)
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
        "local_ok": local_ok,
        "local_remote_wallet_ok": bool(local_remote_wallet.get("ok")),
        "local_remote_wallet": local_remote_wallet,
        "testnet_ok": testnet_ok,
        "message": (
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
    if remote:
        print(
            "Local remote MetaMask: "
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
    parser.add_argument("--testnet-artifacts-dir", type=Path, default=TESTNET_RUN_DIR)
    parser.add_argument("--local-only", action="store_true", help="Only require the local demo gate.")
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

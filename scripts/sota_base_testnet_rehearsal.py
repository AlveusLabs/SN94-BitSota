#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.request import Request, urlopen


REPOS = Path("/home/mekaneeky/repos")
DOCS_REPO = Path(__file__).resolve().parents[1]
POOL_REPO = REPOS / "Pool"
WEBSITE_REPO = REPOS / "bitsota_website"
BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
DEFAULT_RPC_URL = "https://sepolia.base.org"
DEFAULT_TEMPLATE = DOCS_REPO / "docs" / "base" / "manifests" / "base-sepolia-deployment-manifest.template.json"
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"


def _json_rpc(rpc_url: str, method: str, params: list[Any] | None = None, *, timeout: float = 10.0) -> Any:
    request = Request(
        rpc_url,
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "sota-base-testnet-rehearsal/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload.get("result")


def _chain_id(rpc_url: str, *, timeout: float) -> int:
    raw = _json_rpc(rpc_url, "eth_chainId", timeout=timeout)
    return int(str(raw), 16)


def assert_base_sepolia_rpc(rpc_url: str, *, timeout: float) -> int:
    chain_id = _chain_id(rpc_url, timeout=timeout)
    if chain_id == BASE_MAINNET_CHAIN_ID:
        raise SystemExit("refusing Base mainnet RPC chain id 8453")
    if chain_id != BASE_SEPOLIA_CHAIN_ID:
        raise SystemExit(f"expected Base Sepolia chain id 84532, got {chain_id}")
    return chain_id


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=str(cwd), env=env, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"command exited {result.returncode}"
        raise SystemExit(detail)
    return result


def _require_existing(path: Path, label: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def deploy_contracts(args: argparse.Namespace, compact_out: Path) -> dict[str, Any]:
    if not os.environ.get(args.private_key_env, "").strip():
        raise SystemExit(f"{args.private_key_env} must be set in the process environment for --deploy")
    cmd = [
        sys.executable,
        "scripts/deploy_sota_base.py",
        "--rpc-url",
        args.rpc_url,
        "--environment",
        "base-sepolia",
        "--private-key-env",
        args.private_key_env,
        "--initial-vault-supply-sota",
        args.initial_vault_supply_sota,
        "--output",
        str(compact_out),
    ]
    if args.owner_address:
        cmd.extend(["--owner-address", args.owner_address])
    if args.supply_authority_address:
        cmd.extend(["--supply-authority-address", args.supply_authority_address])
    if args.emission_authority_address:
        cmd.extend(["--emission-authority-address", args.emission_authority_address])
    if args.root_publisher_address:
        cmd.extend(["--root-publisher-address", args.root_publisher_address])
    result = _run(cmd, cwd=POOL_REPO)
    return {"command": _redacted_command(cmd), "stdout_bytes": len(result.stdout)}


def _redacted_command(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_value = False
    for item in cmd:
        if skip_value:
            redacted.append("<redacted>")
            skip_value = False
            continue
        redacted.append(item)
        if item == "--private-key-env":
            skip_value = True
    return redacted


def generate_manifest_and_env(args: argparse.Namespace, compact_path: Path, manifest_out: Path, env_out: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/sota_base_testnet_manifest.py",
        "--template",
        str(args.template),
        "--deployment",
        str(compact_path),
        "--manifest-out",
        str(manifest_out),
        "--env-out",
        str(env_out),
        "--public-rpc-url",
        args.rpc_url,
        "--default-lane-id",
        args.default_lane_id,
        "--claims-ui-url",
        args.claims_ui_url,
        "--claims-ui-health-url",
        args.claims_ui_health_url,
        "--indexer-api-url",
        args.indexer_api_url,
        "--indexer-api-health-url",
        args.indexer_api_health_url,
        "--root-publisher-url",
        args.root_publisher_url,
        "--root-publisher-health-url",
        args.root_publisher_health_url,
        "--attestation-builder-url",
        args.attestation_builder_url,
        "--attestation-builder-health-url",
        args.attestation_builder_health_url,
        "--monitoring-url",
        args.monitoring_url,
        "--monitoring-alert-policy-url",
        args.monitoring_alert_policy_url,
        "--monitoring-log-group-or-sink",
        args.monitoring_log_group_or_sink,
        "--autoresearch-api-url",
        args.autoresearch_api_url,
        "--test-wallet-address",
        args.test_wallet_address,
        "--test-old-coldkey",
        args.test_old_coldkey,
        "--test-epoch",
        args.test_epoch,
        "--readiness-url",
        args.readiness_url,
    ]
    if args.source_verification_base_url:
        cmd.extend(["--source-verification-base-url", args.source_verification_base_url])
    result = _run(cmd, cwd=DOCS_REPO)
    return json.loads(result.stdout)


def run_preflight(args: argparse.Namespace, manifest_out: Path, env_out: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/sota_base_testnet_preflight.py",
        str(manifest_out),
        "--env-file",
        str(env_out),
        "--timeout",
        str(args.timeout),
        "--json",
    ]
    if args.offline:
        cmd.append("--offline")
    if args.allow_blocked:
        cmd.append("--allow-blocked")
    result = _run(cmd, cwd=DOCS_REPO)
    return json.loads(result.stdout)


def build_website(env_out: Path) -> dict[str, Any]:
    build_env = os.environ.copy()
    build_env.update(_load_env_file(env_out))
    result = _run(["corepack", "pnpm", "build"], cwd=WEBSITE_REPO, env=build_env)
    return {"ok": True, "stdout_bytes": len(result.stdout)}


def public_readiness_report(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    preflight = report.get("preflight") if isinstance(report.get("preflight"), dict) else {}
    checks = preflight.get("checks") if isinstance(preflight.get("checks"), list) else []
    public_checks = []
    for item in checks:
        if not isinstance(item, dict):
            continue
        public_checks.append(
            {
                "name": str(item.get("name") or ""),
                "status": str(item.get("status") or "unknown"),
                "detail": str(item.get("detail") or ""),
                "remediation": str(item.get("remediation") or ""),
            }
        )
    return {
        "schema": "sota-base-testnet-readiness/v1",
        "generated_at": report.get("generated_at"),
        "environment": "base-sepolia",
        "chain": {
            "chain_id": BASE_SEPOLIA_CHAIN_ID,
            "chain_name": "Base Sepolia",
            "rpc_url": args.rpc_url,
            "explorer_url": "https://sepolia.basescan.org",
        },
        "status": str(preflight.get("status") or "unknown"),
        "ok": bool(preflight.get("ok")),
        "summary": preflight.get("summary") if isinstance(preflight.get("summary"), dict) else {},
        "tester_message": (
            "Base Sepolia testnet claims are ready for browser-wallet smoke."
            if preflight.get("ok")
            else "Base Sepolia testnet claims are not open for nontechnical testers yet."
        ),
        "claims_ui_url": args.claims_ui_url,
        "claims_api_url": args.indexer_api_url,
        "autoresearch_url": args.autoresearch_api_url,
        "checks": public_checks,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    _require_existing(args.template, "manifest template")
    assert_base_sepolia_rpc(args.rpc_url, timeout=args.timeout)

    compact_out = args.artifacts_dir / "base-sepolia-compact-deployment.json"
    manifest_out = args.artifacts_dir / "base-sepolia-deployment-manifest.json"
    env_out = args.artifacts_dir / "base-sota.env.testnet"
    readiness_out = args.readiness_out or (args.artifacts_dir / "base-sota-testnet-readiness.json")
    report: dict[str, Any] = {
        "ok": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts_dir": str(args.artifacts_dir),
        "rpc_url": args.rpc_url,
        "deployed": False,
        "compact_deployment": str(compact_out),
        "manifest": str(manifest_out),
        "env": str(env_out),
        "readiness": str(readiness_out),
    }

    if args.deploy:
        report["deploy"] = deploy_contracts(args, compact_out)
        report["deployed"] = True
    else:
        compact_source = _require_existing(args.deployment, "compact deployment")
        compact_out = compact_source
        report["compact_deployment"] = str(compact_out)

    report["manifest_generation"] = generate_manifest_and_env(args, compact_out, manifest_out, env_out)
    report["preflight"] = run_preflight(args, manifest_out, env_out)
    if args.build_website:
        report["website_build"] = build_website(env_out)
    report["ok"] = bool(report["preflight"].get("ok")) and (not args.build_website or report.get("website_build", {}).get("ok"))
    readiness = public_readiness_report(args, report)
    readiness_out.parent.mkdir(parents=True, exist_ok=True)
    readiness_out.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the guarded Base SOTA Base Sepolia rehearsal path.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--deploy", action="store_true", help="Broadcast Base Sepolia contract deployment transactions")
    parser.add_argument("--private-key-env", default="SOTA_DEPLOYER_PRIVATE_KEY")
    parser.add_argument("--initial-vault-supply-sota", default=os.environ.get("SOTA_INITIAL_VAULT_SUPPLY", "1000000"))
    parser.add_argument("--owner-address", default=os.environ.get("SOTA_OWNER_ADDRESS", ""))
    parser.add_argument("--supply-authority-address", default=os.environ.get("SOTA_SUPPLY_AUTHORITY_ADDRESS", ""))
    parser.add_argument("--emission-authority-address", default=os.environ.get("SOTA_EMISSION_AUTHORITY_ADDRESS", ""))
    parser.add_argument("--root-publisher-address", default=os.environ.get("SOTA_ROOT_PUBLISHER_ADDRESS", ""))
    parser.add_argument("--default-lane-id", default=os.environ.get("NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID", "base:sota-local"))
    parser.add_argument("--claims-ui-url", default=os.environ.get("SOTA_CLAIMS_UI_URL", ""))
    parser.add_argument("--claims-ui-health-url", default=os.environ.get("SOTA_CLAIMS_UI_HEALTH_URL", ""))
    parser.add_argument("--indexer-api-url", default=os.environ.get("SOTA_CLAIMS_API_URL", ""))
    parser.add_argument("--indexer-api-health-url", default=os.environ.get("SOTA_INDEXER_HEALTHCHECK_URL", ""))
    parser.add_argument("--root-publisher-url", default=os.environ.get("SOTA_ROOT_PUBLISHER_URL", ""))
    parser.add_argument("--root-publisher-health-url", default=os.environ.get("SOTA_ROOT_PUBLISHER_HEALTH_URL", ""))
    parser.add_argument("--attestation-builder-url", default=os.environ.get("SOTA_ATTESTATION_SERVICE_URL", ""))
    parser.add_argument("--attestation-builder-health-url", default=os.environ.get("SOTA_ATTESTATION_HEALTHCHECK_URL", ""))
    parser.add_argument("--monitoring-url", default=os.environ.get("SOTA_MONITORING_URL", ""))
    parser.add_argument("--monitoring-alert-policy-url", default=os.environ.get("SOTA_MONITORING_ALERT_POLICY_URL", ""))
    parser.add_argument("--monitoring-log-group-or-sink", default=os.environ.get("SOTA_MONITORING_LOG_GROUP_OR_SINK", ""))
    parser.add_argument("--autoresearch-api-url", default=os.environ.get("SOTA_COORDINATOR_URL", ""))
    parser.add_argument("--test-wallet-address", default=os.environ.get("SOTA_TEST_WALLET_ADDRESS", ""))
    parser.add_argument("--test-old-coldkey", default=os.environ.get("SOTA_TEST_OLD_COLDKEY", ""))
    parser.add_argument("--test-epoch", default=os.environ.get("SOTA_TEST_EPOCH", "1"))
    parser.add_argument("--readiness-url", default=os.environ.get("NEXT_PUBLIC_SOTA_READINESS_URL", ""))
    parser.add_argument("--readiness-out", type=Path)
    parser.add_argument("--source-verification-base-url", default="https://sepolia.basescan.org")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--offline", action="store_true", help="Skip live service/bytecode/balance preflight checks")
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 and write report even if preflight is not green")
    parser.add_argument("--build-website", action="store_true", help="Run the claims website production build using generated env")
    parser.add_argument("--report-out", type=Path, default=Path(""))
    args = parser.parse_args(argv)

    if not args.deploy and args.deployment is None:
        raise SystemExit("provide --deployment <compact-deployment.json> or pass --deploy")

    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

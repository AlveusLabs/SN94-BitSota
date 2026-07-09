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
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sota_prepare_fresh_testnet_tester import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_AWS_PROFILE,
    DEFAULT_AWS_REGION,
    DEFAULT_CLAIMS_API_URL,
    DEFAULT_COORDINATOR_URL,
    DEFAULT_INDEXER_ADMIN_SECRET_ID,
    DEFAULT_RPC_URL,
    DEFAULT_SPONSOR_KEY_FILE,
    _assert_ok,
    _load_indexer_admin_token,
    _run_command,
    _top_up_if_needed,
)


DOCS_REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_PUBLISHER_SECRET_ID = "base-sota/test/base-sepolia/root-publisher"
DEFAULT_DEPLOYER_SECRET_ID = "base-sota/test/base-sepolia/deployer"
DEFAULT_LANE_ID = "base:sota-local"
DEFAULT_DEPLOYMENT_MANIFEST = DEFAULT_ARTIFACTS_DIR / "base-sepolia-deployment-manifest.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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


def _secret_private_key(secret_id: str, *, fields: tuple[str, ...], args: argparse.Namespace) -> str:
    text = _aws_secret_string(
        secret_id,
        profile=args.aws_profile,
        region=args.aws_region,
        timeout=args.timeout,
    )
    candidates: list[str] = [text]
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise RuntimeError(f"secret {secret_id!r} JSON must be an object")
        candidates = [str(payload.get(field) or "").strip() for field in fields]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        if not value.startswith("0x"):
            value = "0x" + value
        if len(value) != 66:
            raise RuntimeError(f"secret {secret_id!r} did not contain a 32-byte private key")
        return value
    raise RuntimeError(f"secret {secret_id!r} is missing one of: {', '.join(fields)}")


def _http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        decoded = response.read().decode("utf-8")
    loaded = json.loads(decoded)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    return loaded


def _eligibility(args: argparse.Namespace, *, reward_address: str, lane_id: str) -> dict[str, Any]:
    query = urlencode({"evm_address": reward_address, "subnet_id": lane_id})
    return _http_json(
        "GET",
        f"{args.claims_api_url.rstrip('/')}/api/v1/base/eligibility/{quote(reward_address)}?{query}",
        timeout=args.timeout,
    )


def _claim_transaction(args: argparse.Namespace, *, reward_address: str, lane_id: str) -> dict[str, Any]:
    return _http_json(
        "POST",
        f"{args.claims_api_url.rstrip('/')}/api/v1/base/claims/transaction",
        payload={"program": "emission", "evmAddress": reward_address, "laneId": lane_id},
        timeout=args.timeout,
    )


def _raw_credit(payload: dict[str, Any], key: str) -> str:
    credits = dict(payload.get("credits") or {})
    value = credits.get(key)
    if isinstance(value, dict):
        return str(value.get("raw") or "")
    return str(value or "")


def _is_claimable(payload: dict[str, Any]) -> bool:
    state = dict(payload.get("claim_state") or {})
    try:
        total = int(_raw_credit(payload, "total_sota") or "0")
        unclaimed = int(_raw_credit(payload, "unclaimed_sota") or "0")
    except ValueError:
        return False
    return bool(payload.get("eligible")) and bool(state.get("claimable")) and total > 0 and unclaimed > 0


def _publisher_saw_epoch(report: dict[str, Any], *, epoch: int) -> bool:
    for row in report.get("published") or []:
        if isinstance(row, dict) and int(row.get("epoch") or 0) == int(epoch):
            return True
    for row in report.get("skipped") or []:
        if isinstance(row, dict) and int(row.get("epoch") or 0) == int(epoch) and row.get("reason") == "already_indexed":
            return True
    return False


def prepare_emission_tester(args: argparse.Namespace) -> dict[str, Any]:
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    reward_key_file = args.reward_key_file or args.artifacts_dir / f"fresh-emission-wallet-{stamp}.json"
    evidence_out = args.evidence_out or args.artifacts_dir / f"base-sota-testnet-emission-evidence-fresh-emission-{stamp}.json"
    seed_report_out = args.seed_report_out or args.artifacts_dir / f"base-sota-testnet-autoresearch-seed-emission-{stamp}.json"
    publisher_report_out = args.publisher_report_out or args.artifacts_dir / "base-sota-emission-batch-publisher.json"

    if args.reuse_seed_report:
        if not seed_report_out.exists():
            raise RuntimeError(f"--reuse-seed-report was set, but {seed_report_out} does not exist")
    else:
        seed_cmd = [
            sys.executable,
            "scripts/sota_seed_testnet_autoresearch.py",
            "--coordinator-url",
            args.coordinator_url,
            "--reward-key-file",
            str(reward_key_file),
            "--evidence-out",
            str(evidence_out),
            "--report-out",
            str(seed_report_out),
            "--aws-profile",
            args.aws_profile,
            "--aws-region",
            args.aws_region,
            "--timeout",
            str(args.timeout),
            "--require-single-claim",
        ]
        if args.metric_value is not None:
            seed_cmd.extend(["--metric-value", str(args.metric_value)])
        if args.epoch is not None:
            seed_cmd.extend(["--epoch", str(args.epoch)])
        seed_result = _run_command(seed_cmd, timeout=args.command_timeout)
        _assert_ok(seed_result, label="public autoresearch emission seed")
    seed_report = _load_json(seed_report_out)
    reward_address = str(seed_report.get("reward_address") or "").strip()
    epoch = int(seed_report.get("epoch") or 0)
    if not reward_address or epoch <= 0:
        raise RuntimeError("seed report is missing reward_address or epoch")

    funding = _top_up_if_needed(args, reward_key_file=reward_key_file)
    env = os.environ.copy()
    env["SOTA_BASE_INDEXER_ADMIN_TOKEN"] = _load_indexer_admin_token(args)
    env["SOTA_ROOT_PUBLISHER_PRIVATE_KEY"] = _secret_private_key(
        args.root_publisher_secret_id,
        fields=("root_publisher_private_key", "private_key", "SOTA_ROOT_PUBLISHER_PRIVATE_KEY"),
        args=args,
    )
    env["SOTA_DEPLOYER_PRIVATE_KEY"] = _secret_private_key(
        args.deployer_secret_id,
        fields=("deployer_private_key", "sota_deployer_private_key", "private_key", "SOTA_DEPLOYER_PRIVATE_KEY"),
        args=args,
    )
    publisher_cmd = [
        sys.executable,
        "scripts/sota_base_emission_batch_publisher.py",
        "--coordinator-url",
        args.coordinator_url,
        "--claims-api-url",
        args.claims_api_url,
        "--lane-id",
        args.lane_id,
        "--epoch",
        str(epoch),
        "--manifest",
        str(args.deployment),
        "--out-dir",
        str(args.artifacts_dir),
        "--report-out",
        str(publisher_report_out),
        "--timeout",
        str(args.timeout),
        "--command-timeout",
        str(args.command_timeout),
        "--broadcast",
        "--import-artifact",
        "--sync-lane",
        "--sync-lane-broadcast",
        "--once",
        "--json",
    ]
    if args.rpc_url:
        publisher_cmd.extend(["--rpc-url", args.rpc_url])
    publisher_result = _run_command(publisher_cmd, timeout=args.operator_timeout, env=env)
    _assert_ok(publisher_result, label="Base Sepolia emission publisher")
    publisher_report = _load_json(publisher_report_out)

    eligibility = _eligibility(args, reward_address=reward_address, lane_id=args.lane_id)
    tx_response = _claim_transaction(args, reward_address=reward_address, lane_id=args.lane_id)
    tx = dict(tx_response.get("transaction") or {})
    claimable = _is_claimable(eligibility)
    calldata_ok = str(tx.get("data") or "").startswith("0x") and str(tx.get("to") or "").startswith("0x")
    published_epoch = _publisher_saw_epoch(publisher_report, epoch=epoch)
    ok = bool(seed_report.get("claim_count") == 1 and published_epoch and claimable and calldata_ok)
    report = {
        "schema": "sota-base-fresh-emission-tester/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "status": "green" if ok else "red",
        "message": (
            "Fresh Base Sepolia emission tester wallet is funded, self-validated, indexed, and claimable."
            if ok
            else "Fresh Base Sepolia emission tester preparation did not reach a claimable state."
        ),
        "reward_address": reward_address,
        "reward_key_file": str(reward_key_file),
        "private_key_printed": False,
        "lane_id": args.lane_id,
        "epoch": epoch,
        "seed_report": str(seed_report_out),
        "evidence_path": str(evidence_out),
        "reused_seed_report": bool(args.reuse_seed_report),
        "publisher_report": str(publisher_report_out),
        "funding": funding,
        "eligibility": eligibility,
        "claim_transaction": {
            "to": str(tx.get("to") or ""),
            "chain_id": str(tx.get("chainId") or ""),
            "data_prefix": str(tx.get("data") or "")[:18],
            "ok": calldata_ok,
        },
        "next_action": "Give wallet access out of band, open the Base Sepolia claims UI, connect this wallet, load mined emission, and submit the MetaMask claim.",
        "does_not": ["print_private_keys", "touch_production_bittensor", "touch_base_mainnet", "test_real_holder_claims"],
    }
    _write_json(args.report_out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a fresh emission-only Base Sepolia tester wallet/root cycle.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--reward-key-file", type=Path)
    parser.add_argument("--evidence-out", type=Path)
    parser.add_argument("--seed-report-out", type=Path)
    parser.add_argument("--publisher-report-out", type=Path)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-fresh-emission-tester.json")
    parser.add_argument("--deployment", type=Path, default=DEFAULT_DEPLOYMENT_MANIFEST)
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL)
    parser.add_argument("--claims-api-url", default=DEFAULT_CLAIMS_API_URL)
    parser.add_argument("--coordinator-url", default=DEFAULT_COORDINATOR_URL)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", DEFAULT_AWS_PROFILE))
    parser.add_argument("--aws-region", default=os.environ.get("AWS_REGION", DEFAULT_AWS_REGION))
    parser.add_argument("--indexer-admin-secret-id", default=DEFAULT_INDEXER_ADMIN_SECRET_ID)
    parser.add_argument("--root-publisher-secret-id", default=DEFAULT_ROOT_PUBLISHER_SECRET_ID)
    parser.add_argument("--deployer-secret-id", default=DEFAULT_DEPLOYER_SECRET_ID)
    parser.add_argument("--sponsor-key-file", type=Path, default=DEFAULT_SPONSOR_KEY_FILE)
    parser.add_argument("--min-wallet-balance-eth", default="0.005")
    parser.add_argument("--top-up-eth", default="0.01")
    parser.add_argument("--max-priority-fee-gwei", default="0.001")
    parser.add_argument("--metric-value", type=float, default=0.80)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--reuse-seed-report", action="store_true", help="Resume from --seed-report-out instead of creating another coordinator submission.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--command-timeout", type=float, default=600.0)
    parser.add_argument("--operator-timeout", type=float, default=900.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = prepare_emission_tester(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(f"Fresh Base Sepolia emission tester: {report['status'].upper()}")
        print(report["message"])
        print(f"Reward address: {report['reward_address']}")
        print(f"Epoch: {report['epoch']}")
        print(f"Next: {report['next_action']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

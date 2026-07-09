#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from web3 import Web3


REPOS = Path("/home/mekaneeky/repos")
DOCS_REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_WEBSITE_REPO = REPOS / "bitsota_website"
DEFAULT_RPC_URL = "https://sepolia.base.org"
DEFAULT_AWS_PROFILE = "moonrocklab-frankfurt"
DEFAULT_AWS_REGION = "eu-central-1"
DEFAULT_DEPLOYMENT = DEFAULT_ARTIFACTS_DIR / "base-sepolia-compact-deployment.json"
DEFAULT_CLAIMS_UI_URL = "https://d5dqb78dur.eu-central-1.awsapprunner.com"
DEFAULT_CLAIMS_API_URL = "https://gs4g5jntcn.eu-central-1.awsapprunner.com"
DEFAULT_COORDINATOR_URL = "https://zuyyfpgpnw.eu-central-1.awsapprunner.com"
DEFAULT_READINESS_URL = "https://d5dqb78dur.eu-central-1.awsapprunner.com/base-sota-testnet-readiness.json"
DEFAULT_INDEXER_ADMIN_SECRET_ID = "base-sota/test/base-sepolia/indexer-admin-token"
DEFAULT_SPONSOR_KEY_FILE = DEFAULT_ARTIFACTS_DIR / "faucet-wallet.json"
DEFAULT_OLD_COLDKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
DEFAULT_SNAPSHOT_DIR = Path("/mnt/4tb/tao_fork_snapshot")
BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
ONE_ETH_WEI = 10**18
PUBLIC_ARTIFACT_FILES = [
    "base-sota-testnet-readiness.json",
    "base-sota-testnet-seed-artifacts-finalized.json",
    "base-sota-testnet-genesis-claim-artifact.json",
    "base-sota-testnet-emission-claim-artifact.json",
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _eth_to_wei(value: str) -> int:
    amount = Decimal(str(value).strip())
    if amount < 0:
        raise ValueError("ETH amount must not be negative")
    wei = amount * Decimal(ONE_ETH_WEI)
    if wei != wei.to_integral_value():
        raise ValueError("ETH amount has more than 18 decimal places")
    return int(wei)


def _format_eth(value: int) -> str:
    return f"{Decimal(int(value)) / Decimal(ONE_ETH_WEI):.8f}"


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


def _secret_value(secret_string: str, *, env_name: str) -> str:
    text = secret_string.strip()
    if not text.startswith("{"):
        return text
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("JSON secret must be an object")
    for key in (env_name, "admin_token", "token", "SOTA_INDEXER_ADMIN_TOKEN"):
        if payload.get(key):
            return str(payload[key]).strip()
    raise ValueError(f"JSON secret does not contain {env_name} or a supported token field")


def _load_indexer_admin_token(args: argparse.Namespace) -> str:
    env_name = "SOTA_INDEXER_ADMIN_TOKEN"
    if os.environ.get(env_name):
        return os.environ[env_name]
    return _secret_value(
        _aws_secret_string(
            args.indexer_admin_secret_id,
            profile=args.aws_profile,
            region=args.aws_region,
            timeout=args.timeout,
        ),
        env_name=env_name,
    )


def _run_command(
    cmd: list[str],
    *,
    cwd: Path = DOCS_REPO,
    timeout: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=cwd, check=False, text=True, capture_output=True, timeout=timeout, env=env)
    return {
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": cmd,
    }


def _assert_ok(result: dict[str, Any], *, label: str) -> None:
    if int(result["returncode"]) == 0:
        return
    detail = str(result.get("stderr") or "").strip() or str(result.get("stdout") or "").strip()
    raise RuntimeError(f"{label} failed: {detail[:2000]}")


def _wallet_address(path: Path) -> str:
    payload = _load_json(path)
    return str(payload.get("address") or "").strip()


def _snapshot_bindings(args: argparse.Namespace) -> list[Path]:
    return [Path(item) for item in (getattr(args, "snapshot_claim_binding", None) or [])]


def _binding_reward_addresses(bindings: list[Path]) -> set[str]:
    addresses: set[str] = set()
    for path in bindings:
        payload = _load_json(path)
        message = payload.get("message") if isinstance(payload.get("message"), dict) else payload
        reward_address = str(dict(message).get("reward_address") or "").strip().lower()
        if not reward_address:
            raise ValueError(f"{path} is missing message.reward_address")
        addresses.add(reward_address)
    return addresses


def _assert_snapshot_binding_inputs(args: argparse.Namespace) -> list[Path]:
    bindings = _snapshot_bindings(args)
    if not bindings:
        raise RuntimeError(
            "Fresh Base Sepolia tester prep now requires a real signed snapshot binding. "
            "Use the claims UI Genesis binding panel or scripts/sota_sign_snapshot_binding.py to submit/sign it with "
            "the Bittensor coldkey, then rerun with --reward-key-file <known-wallet-key.json> and "
            "--snapshot-claim-binding <signed-binding.json>."
        )
    if args.reward_key_file is None:
        raise RuntimeError(
            "--reward-key-file is required with --snapshot-claim-binding so the coldkey binding can target a known MetaMask wallet."
        )
    return bindings


def _sponsor_key(path: Path) -> tuple[str, str]:
    payload = _load_json(path)
    address = str(payload.get("address") or "").strip()
    private_key = str(payload.get("private_key") or "").strip()
    if not address or not private_key:
        raise ValueError(f"{path} must contain address and private_key")
    return address, private_key


def _top_up_if_needed(args: argparse.Namespace, *, reward_key_file: Path) -> dict[str, Any]:
    w3 = Web3(Web3.HTTPProvider(args.rpc_url))
    chain_id = int(w3.eth.chain_id)
    if chain_id == BASE_MAINNET_CHAIN_ID:
        raise RuntimeError("refusing to fund on Base mainnet chain id 8453")
    if chain_id != BASE_SEPOLIA_CHAIN_ID:
        raise RuntimeError(f"expected Base Sepolia chain id 84532, got {chain_id}")
    recipient = Web3.to_checksum_address(_wallet_address(reward_key_file))
    before = int(w3.eth.get_balance(recipient))
    min_balance = _eth_to_wei(args.min_wallet_balance_eth)
    if before >= min_balance:
        return {
            "status": "skipped",
            "recipient": recipient,
            "balance_before_wei": str(before),
            "balance_after_wei": str(before),
            "balance_after_eth": _format_eth(before),
            "tx_hash": "",
        }
    if not args.sponsor_key_file.exists():
        raise RuntimeError(f"{args.sponsor_key_file} is missing; fund {recipient} with Base Sepolia ETH and rerun")
    sender_address, private_key = _sponsor_key(args.sponsor_key_file)
    sender = Web3.to_checksum_address(sender_address)
    sponsor_balance = int(w3.eth.get_balance(sender))
    amount = max(_eth_to_wei(args.top_up_eth), min_balance - before)
    gas_price = int(w3.eth.gas_price)
    max_priority = Web3.to_wei(Decimal(args.max_priority_fee_gwei), "gwei")
    max_fee = max(gas_price * 2, int(max_priority) * 2)
    estimated_cost = amount + 21_000 * int(max_fee)
    if sponsor_balance < estimated_cost:
        raise RuntimeError(
            f"sponsor wallet {sender} has {_format_eth(sponsor_balance)} ETH; needs at least {_format_eth(estimated_cost)} ETH"
        )
    tx = {
        "chainId": chain_id,
        "type": 2,
        "to": recipient,
        "value": amount,
        "nonce": int(w3.eth.get_transaction_count(sender)),
        "gas": 21_000,
        "maxFeePerGas": int(max_fee),
        "maxPriorityFeePerGas": int(max_priority),
    }
    signed = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if int(receipt.status) != 1:
        raise RuntimeError(f"funding transaction reverted: {tx_hash.hex()}")
    deadline = time.time() + 30
    after = int(w3.eth.get_balance(recipient))
    while after < before + amount and time.time() < deadline:
        time.sleep(1)
        after = int(w3.eth.get_balance(recipient))
    return {
        "status": "funded",
        "recipient": recipient,
        "sponsor": sender,
        "amount_wei": str(amount),
        "amount_eth": _format_eth(amount),
        "balance_before_wei": str(before),
        "balance_after_wei": str(after),
        "balance_after_eth": _format_eth(after),
        "tx_hash": tx_hash.hex() if tx_hash.hex().startswith("0x") else "0x" + tx_hash.hex(),
    }


def _http_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{url} returned non-object JSON")
    return payload


def _eligibility(args: argparse.Namespace, *, wallet: str, old_coldkey: str, lane_id: str) -> dict[str, Any]:
    base = args.claims_api_url.rstrip("/")
    genesis_query = urlencode({"old_coldkey": old_coldkey, "reward_address": wallet, "subnet_id": "genesis"})
    emission_query = urlencode({"evm_address": wallet, "subnet_id": lane_id})
    encoded_wallet = quote(wallet)
    return {
        "genesis": _http_json(f"{base}/api/v1/base/eligibility/{encoded_wallet}?{genesis_query}", timeout=args.timeout),
        "emission": _http_json(f"{base}/api/v1/base/eligibility/{encoded_wallet}?{emission_query}", timeout=args.timeout),
    }


def _is_claimable(payload: dict[str, Any]) -> bool:
    state = dict(payload.get("claim_state") or {})
    credits = dict(payload.get("credits") or {})
    unclaimed = str(dict(credits.get("unclaimed_sota") or {}).get("raw") or "0")
    try:
        unclaimed_units = int(unclaimed)
    except ValueError:
        unclaimed_units = 0
    return bool(payload.get("eligible")) and bool(state.get("claimable")) and unclaimed_units > 0


def _refresh_website_public_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_website_public_refresh:
        return {"status": "skipped", "copied": [], "reason": "disabled by --skip-website-public-refresh"}
    public_dir = args.website_repo / "public"
    nested_dir = public_dir / "base-sota"
    if not public_dir.exists():
        return {"status": "skipped", "copied": [], "reason": f"{public_dir} does not exist"}
    copied: list[str] = []
    for filename in PUBLIC_ARTIFACT_FILES:
        source = args.artifacts_dir / filename
        if not source.exists():
            raise RuntimeError(f"cannot refresh website public artifact; missing {source}")
        for target_dir in (public_dir, nested_dir):
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / filename
            shutil.copyfile(source, target)
            copied.append(str(target))
    return {
        "status": "green",
        "website_repo": str(args.website_repo),
        "copied": copied,
        "next_action": "Commit and push the refreshed public JSON artifacts so App Runner serves the current wallet/root cycle.",
    }


def prepare_fresh_tester(args: argparse.Namespace) -> dict[str, Any]:
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    snapshot_bindings = _assert_snapshot_binding_inputs(args)
    stamp = _timestamp()
    reward_key_file = args.reward_key_file or args.artifacts_dir / f"fresh-claim-wallet-{stamp}.json"
    evidence_out = args.evidence_out or args.artifacts_dir / f"base-sota-testnet-emission-evidence-fresh-{stamp}.json"
    seed_report_out = args.seed_report_out or args.artifacts_dir / f"base-sota-testnet-autoresearch-seed-{stamp}.json"
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
    seed_result = _run_command(seed_cmd, timeout=args.command_timeout)
    _assert_ok(seed_result, label="public autoresearch seed")
    seed_report = _load_json(seed_report_out)
    wallet = str(seed_report.get("reward_address") or _wallet_address(reward_key_file))
    binding_reward_addresses = _binding_reward_addresses(snapshot_bindings)
    if binding_reward_addresses != {wallet.lower()}:
        raise RuntimeError(
            "snapshot binding reward_address does not match the prepared tester wallet: "
            f"binding={sorted(binding_reward_addresses)} wallet={wallet}"
        )
    epoch = str(seed_report.get("epoch") or "")
    funding = _top_up_if_needed(args, reward_key_file=reward_key_file)
    env = os.environ.copy()
    env["SOTA_INDEXER_ADMIN_TOKEN"] = _load_indexer_admin_token(args)
    operator_cmd = [
        sys.executable,
        "scripts/sota_base_testnet_operator.py",
        "--aws-profile",
        args.aws_profile,
        "--aws-region",
        args.aws_region,
        "--deployment",
        str(args.deployment),
        "--claims-ui-url",
        args.claims_ui_url,
        "--claims-api-url",
        args.claims_api_url,
        "--coordinator-url",
        args.coordinator_url,
        "--readiness-url",
        args.readiness_url,
        "--emission-evidence",
        str(evidence_out),
        "--snapshot-dir",
        str(args.snapshot_dir),
        "--test-wallet-address",
        wallet,
        "--test-old-coldkey",
        args.test_old_coldkey,
        "--test-epoch",
        epoch,
        "--timeout",
        str(args.timeout),
        "--command-timeout",
        str(args.command_timeout),
        "--broadcast-roots",
        "--import-artifacts",
        "--allow-blocked",
    ]
    for binding in snapshot_bindings:
        operator_cmd.extend(["--snapshot-claim-binding", str(binding)])
    operator_result = _run_command(operator_cmd, timeout=args.operator_timeout, env=env)
    _assert_ok(operator_result, label="Base Sepolia operator")
    release_result = _run_command(
        [
            sys.executable,
            "scripts/sota_base_release_status.py",
            "--testnet-artifacts-dir",
            str(args.artifacts_dir),
            "--snapshot-dir",
            str(args.snapshot_dir),
            "--report-out",
            str(args.artifacts_dir / "base-sota-release-status.json"),
            "--allow-blocked",
        ],
        timeout=args.command_timeout,
    )
    _assert_ok(release_result, label="release status refresh")
    handoff_result = _run_command(
        [
            sys.executable,
            "scripts/sota_base_tester_handoff.py",
            "--environment",
            "both",
            "--release-status",
            str(args.artifacts_dir / "base-sota-release-status.json"),
            "--mirror-local",
        ],
        timeout=args.command_timeout,
    )
    _assert_ok(handoff_result, label="tester handoff refresh")
    website_public_artifacts = _refresh_website_public_artifacts(args)
    operator_report = _load_json(args.artifacts_dir / "base-sota-testnet-operator-run.json")
    eligibility = _eligibility(args, wallet=wallet, old_coldkey=args.test_old_coldkey, lane_id=args.lane_id)
    claimable = _is_claimable(dict(eligibility["genesis"])) and _is_claimable(dict(eligibility["emission"]))
    report = {
        "schema": "sota-base-fresh-testnet-tester/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": bool(operator_report.get("status") == "green" and claimable),
        "status": "green" if operator_report.get("status") == "green" and claimable else "red",
        "message": (
            "Fresh Base Sepolia tester wallet is funded and claimable."
            if operator_report.get("status") == "green" and claimable
            else "Fresh Base Sepolia tester preparation did not reach a claimable state."
        ),
        "reward_address": wallet,
        "reward_key_file": str(reward_key_file),
        "private_key_printed": False,
        "epoch": epoch,
        "old_coldkey": args.test_old_coldkey,
        "claims_ui_url": args.claims_ui_url.rstrip("/") + "/claims",
        "claims_api_url": args.claims_api_url,
        "seed_report": str(seed_report_out),
        "evidence_path": str(evidence_out),
        "funding": funding,
        "eligibility": eligibility,
        "operator_report": str(args.artifacts_dir / "base-sota-testnet-operator-run.json"),
        "release_status": str(args.artifacts_dir / "base-sota-release-status.json"),
        "handoff": str(args.artifacts_dir / "base-sota-tester-handoff.md"),
        "website_public_artifacts": website_public_artifacts,
        "next_action": "Give wallet access out of band, open the claims UI, submit genesis and emission in MetaMask, then run the claim transaction evidence verifier.",
        "does_not": ["print_private_keys", "touch_production_bittensor", "touch_base_mainnet"],
    }
    _write_json(args.report_out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a fresh claimable Base Sepolia test wallet/root cycle.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--reward-key-file", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--snapshot-claim-binding",
        type=Path,
        action="append",
        default=[],
        help="Signed Bittensor coldkey binding JSON for the tester reward wallet; repeat for multiple claimants.",
    )
    parser.add_argument("--evidence-out", type=Path)
    parser.add_argument("--seed-report-out", type=Path)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-fresh-testnet-tester.json")
    parser.add_argument("--deployment", type=Path, default=DEFAULT_DEPLOYMENT)
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL)
    parser.add_argument("--claims-ui-url", default=DEFAULT_CLAIMS_UI_URL)
    parser.add_argument("--claims-api-url", default=DEFAULT_CLAIMS_API_URL)
    parser.add_argument("--coordinator-url", default=DEFAULT_COORDINATOR_URL)
    parser.add_argument("--readiness-url", default=DEFAULT_READINESS_URL)
    parser.add_argument("--lane-id", default="base:sota-local")
    parser.add_argument("--test-old-coldkey", default=DEFAULT_OLD_COLDKEY)
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", DEFAULT_AWS_PROFILE))
    parser.add_argument("--aws-region", default=os.environ.get("AWS_REGION", DEFAULT_AWS_REGION))
    parser.add_argument("--indexer-admin-secret-id", default=DEFAULT_INDEXER_ADMIN_SECRET_ID)
    parser.add_argument("--sponsor-key-file", type=Path, default=DEFAULT_SPONSOR_KEY_FILE)
    parser.add_argument("--website-repo", type=Path, default=DEFAULT_WEBSITE_REPO)
    parser.add_argument("--skip-website-public-refresh", action="store_true")
    parser.add_argument("--min-wallet-balance-eth", default="0.005")
    parser.add_argument("--top-up-eth", default="0.01")
    parser.add_argument("--max-priority-fee-gwei", default="0.001")
    parser.add_argument("--metric-value", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--command-timeout", type=float, default=600.0)
    parser.add_argument("--operator-timeout", type=float, default=900.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = prepare_fresh_tester(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(f"Fresh Base Sepolia tester: {report['status'].upper()}")
        print(report["message"])
        print(f"Reward address: {report['reward_address']}")
        print(f"Claims UI: {report['claims_ui_url']}")
        print(f"Handoff: {report['handoff']}")
        print(f"Next: {report['next_action']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

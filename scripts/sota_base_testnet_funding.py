#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.request import Request, urlopen


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_LOCAL_STATE = REPOS / ".sota-base-local" / "state.json"
DEFAULT_REPORT_OUT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-funding.json"
DEFAULT_BLOCKER_REPORT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-blockers.json"
DEFAULT_RPC_URL = "https://sepolia.base.org"
DEFAULT_AWS_PROFILE = "moonrocklab-frankfurt"
DEFAULT_REGION = "eu-central-1"
DEFAULT_DEPLOYER_SECRET_ID = "base-sota/test/base-sepolia/deployer"
DEFAULT_ROOT_PUBLISHER_SECRET_ID = "base-sota/test/base-sepolia/root-publisher"
BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
EVM_ADDRESS_ANY_RE = re.compile(r"0x[0-9a-fA-F]{40}")
ONE_ETH_WEI = 10**18
BASE_SEPOLIA_EXPLORER_ADDRESS_URL = "https://sepolia.basescan.org/address/"
BASE_NETWORK_FAUCETS_URL = "https://docs.base.org/base-chain/network-information/network-faucets"
DEFAULT_MIN_BALANCE_ETH = {
    "deployer": "0.020",
    "root_publisher": "0.005",
    "test_wallet": "0.005",
    "default": "0.001",
}


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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


def _json_rpc(rpc_url: str, method: str, params: list[Any] | None = None, *, timeout: float) -> Any:
    request = Request(
        rpc_url,
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "sota-base-testnet-funding/1.0",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload.get("result")


def _chain_id(rpc_url: str, *, timeout: float) -> int:
    return int(str(_json_rpc(rpc_url, "eth_chainId", timeout=timeout)), 16)


def _balance_wei(rpc_url: str, address: str, *, timeout: float) -> int:
    return int(str(_json_rpc(rpc_url, "eth_getBalance", [address, "latest"], timeout=timeout)), 16)


def _eth_to_wei(value: str, *, field: str) -> int:
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a decimal ETH amount") from exc
    if amount < 0:
        raise ValueError(f"{field} must not be negative")
    wei = amount * Decimal(ONE_ETH_WEI)
    if wei != wei.to_integral_value():
        raise ValueError(f"{field} has more than 18 decimal places")
    return int(wei)


def _format_eth_from_wei(value: int) -> str:
    return f"{Decimal(int(value)) / Decimal(ONE_ETH_WEI):.8f}"


def _min_balances(args: argparse.Namespace) -> dict[str, int]:
    raw = dict(DEFAULT_MIN_BALANCE_ETH)
    for item in getattr(args, "min_balance", []) or []:
        if "=" not in item:
            raise SystemExit(f"--min-balance must be label=ETH, got {item!r}")
        label, amount = item.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"--min-balance label is empty in {item!r}")
        raw[label] = amount.strip()
    return {
        label: _eth_to_wei(amount, field=f"min_balance.{label}")
        for label, amount in raw.items()
    }


def _secret_tag(secret_id: str, tag_key: str, *, profile: str, region: str, timeout: float) -> str:
    payload = _run_aws(
        ["secretsmanager", "describe-secret", "--secret-id", secret_id],
        profile=profile,
        region=region,
        timeout=timeout,
    )
    for tag in payload.get("Tags") or []:
        if isinstance(tag, dict) and str(tag.get("Key") or "") == tag_key and tag.get("Value"):
            return str(tag["Value"]).strip()
    return ""


def _aws_identity_check(args: argparse.Namespace) -> Check:
    try:
        payload = _run_aws(["sts", "get-caller-identity"], profile=args.aws_profile, region=args.region, timeout=args.timeout)
    except Exception as exc:
        return Check(
            "aws_identity",
            "red",
            f"AWS identity unavailable: {exc}",
            "Authenticate with the approved testnet AWS profile before checking funding readiness.",
        )
    return Check(
        "aws_identity",
        "green",
        f"Authenticated to AWS account {payload.get('Account')} as {payload.get('Arn')}.",
    )


def _rpc_check(args: argparse.Namespace) -> Check:
    try:
        chain_id = _chain_id(args.rpc_url, timeout=args.timeout)
    except Exception as exc:
        return Check(
            "base_sepolia_rpc",
            "red",
            f"Could not read chain id from {args.rpc_url}: {exc}",
            "Use a reachable Base Sepolia JSON-RPC endpoint.",
        )
    if chain_id == BASE_MAINNET_CHAIN_ID:
        return Check("base_sepolia_rpc", "red", "RPC returned Base mainnet chain id 8453.", "Never use Base mainnet for this testnet gate.")
    if chain_id != BASE_SEPOLIA_CHAIN_ID:
        return Check("base_sepolia_rpc", "red", f"RPC returned chain id {chain_id}, expected 84532.", "Point the config at Base Sepolia.")
    return Check("base_sepolia_rpc", "green", f"RPC returned Base Sepolia chain id {chain_id}.")


def _target_from_secret(label: str, secret_id: str, args: argparse.Namespace) -> dict[str, str]:
    try:
        address = _secret_tag(secret_id, "sota-address", profile=args.aws_profile, region=args.region, timeout=args.timeout)
    except Exception as exc:
        return {
            "label": label,
            "source": f"secret-tag:{secret_id}",
            "address": "",
            "error": str(exc),
        }
    return {"label": label, "source": f"secret-tag:{secret_id}", "address": address}


def _fallback_reports(args: argparse.Namespace) -> list[Path]:
    configured = getattr(args, "fallback_report", None)
    if configured is None:
        return [DEFAULT_REPORT_OUT, DEFAULT_BLOCKER_REPORT]
    return [Path(item) for item in configured]


def _fallback_targets(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in _fallback_reports(args):
        if not path.exists():
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        for target in payload.get("funding_targets") or []:
            if not isinstance(target, dict):
                continue
            label = str(target.get("label") or "").strip()
            address = str(target.get("address") or "").strip()
            if label and EVM_ADDRESS_RE.fullmatch(address):
                out.setdefault(label, {"label": label, "source": f"fallback-report:{path}", "address": address})
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
                out.setdefault(label, {"label": label, "source": f"fallback-report:{path}", "address": match.group(0)})
    return out


def _apply_fallback_target(target: dict[str, str], fallbacks: dict[str, dict[str, str]]) -> dict[str, str]:
    label = str(target.get("label") or "").strip()
    fallback = fallbacks.get(label)
    if not fallback:
        return target
    if target.get("address") and not target.get("error"):
        return target
    merged = dict(fallback)
    if target.get("error"):
        merged["fallback_reason"] = str(target["error"])
    return merged


def _target_from_local_state(args: argparse.Namespace) -> dict[str, str] | None:
    if args.test_wallet_address:
        return {"label": "test_wallet", "source": "argument", "address": args.test_wallet_address}
    if not args.local_state.exists():
        return None
    try:
        state = _load_json(args.local_state)
    except Exception:
        return None
    address = str(dict(state.get("accounts") or {}).get("alice_reward") or "").strip()
    if not address:
        return None
    return {"label": "test_wallet", "source": str(args.local_state), "address": address}


def _target_check(target: dict[str, str], args: argparse.Namespace, min_balance_wei: int) -> tuple[Check, dict[str, Any]]:
    label = str(target.get("label") or "target")
    address = str(target.get("address") or "").strip()
    row: dict[str, Any] = {
        "label": label,
        "source": str(target.get("source") or ""),
        "address": address,
        "balance_wei": None,
        "balance_eth": None,
        "minimum_balance_wei": str(min_balance_wei),
        "minimum_balance_eth": _format_eth_from_wei(min_balance_wei),
        "needed_wei": None,
        "needed_eth": None,
        "explorer_url": f"{BASE_SEPOLIA_EXPLORER_ADDRESS_URL}{address}" if EVM_ADDRESS_RE.fullmatch(address) else "",
    }
    if target.get("error"):
        check = Check(
            f"funding_{label}",
            "red",
            f"Could not read public funding address for {label}: {target['error']}",
            "Fix AWS access or add the public sota-address tag to the approved testnet secret handle.",
        )
        row["status"] = check.status
        row["remediation"] = check.remediation
        return check, row
    if not address:
        check = Check(
            f"funding_{label}",
            "red",
            f"No public funding address is configured for {label}.",
            "Record the public Base Sepolia address before deployment/browser smoke.",
        )
        row["status"] = check.status
        row["remediation"] = check.remediation
        return check, row
    if not EVM_ADDRESS_RE.fullmatch(address):
        check = Check(
            f"funding_{label}",
            "red",
            f"{label} address is not a valid EVM address: {address!r}.",
            "Use a 20-byte Base Sepolia EVM address.",
        )
        row["status"] = check.status
        row["remediation"] = check.remediation
        return check, row
    try:
        balance = _balance_wei(args.rpc_url, address, timeout=args.timeout)
    except Exception as exc:
        check = Check(
            f"funding_{label}",
            "red",
            f"Could not read Base Sepolia ETH balance for {label} {address}: {exc}",
            "Fix the Base Sepolia RPC endpoint before deployment/browser smoke.",
        )
        row["status"] = check.status
        row["remediation"] = check.remediation
        return check, row
    row["balance_wei"] = str(balance)
    row["balance_eth"] = _format_eth_from_wei(balance)
    needed = max(0, int(min_balance_wei) - int(balance))
    row["needed_wei"] = str(needed)
    row["needed_eth"] = _format_eth_from_wei(needed)
    if balance < min_balance_wei:
        check = Check(
            f"funding_{label}",
            "red",
            (
                f"{label} {address} has {row['balance_eth']} ETH on Base Sepolia; "
                f"minimum is {row['minimum_balance_eth']} ETH."
            ),
            (
                f"Fund {address} with at least {row['needed_eth']} more Base Sepolia ETH "
                "before deployment/browser smoke."
            ),
        )
        row["status"] = check.status
        row["remediation"] = check.remediation
        return check, row
    check = Check(
        f"funding_{label}",
        "green",
        (
            f"{label} {address} has {row['balance_eth']} ETH on Base Sepolia "
            f"(minimum {row['minimum_balance_eth']} ETH)."
        ),
    )
    row["status"] = check.status
    return check, row


def _worst(checks: list[Check]) -> str:
    rank = {"green": 0, "yellow": 1, "red": 2}
    return max((check.status for check in checks), key=lambda status: rank.get(status, 2), default="green")


def _next_actions(checks: list[Check]) -> list[str]:
    actions: list[str] = []
    for check in checks:
        if check.status == "green" or not check.remediation:
            continue
        if check.remediation not in actions:
            actions.append(check.remediation)
    return actions


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    checks = [_aws_identity_check(args), _rpc_check(args)]
    min_balances = _min_balances(args)
    raw_targets = [
        _target_from_secret("deployer", args.deployer_secret_id, args),
        _target_from_secret("root_publisher", args.root_publisher_secret_id, args),
    ]
    test_wallet = _target_from_local_state(args)
    if test_wallet:
        raw_targets.append(test_wallet)
    for item in args.target:
        if "=" not in item:
            raise SystemExit(f"--target must be label=0xAddress, got {item!r}")
        label, address = item.split("=", 1)
        raw_targets.append({"label": label.strip(), "source": "argument", "address": address.strip()})

    funding_targets: list[dict[str, Any]] = []
    fallbacks = _fallback_targets(args)
    for target in raw_targets:
        target = _apply_fallback_target(target, fallbacks)
        label = str(target.get("label") or "target").strip() or "target"
        check, row = _target_check(target, args, min_balances.get(label, min_balances["default"]))
        if target.get("fallback_reason"):
            row["fallback_reason"] = str(target["fallback_reason"])
        checks.append(check)
        funding_targets.append(row)

    status = _worst(checks)
    return {
        "schema": "sota-base-testnet-funding/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": status == "green",
        "status": status,
        "message": (
            "Base Sepolia funding targets have gas."
            if status == "green"
            else "Base Sepolia funding targets are not ready for deployment/browser smoke."
        ),
        "read_only": True,
        "read_secret_values": False,
        "does_not": ["read_secret_values", "sign", "broadcast_transactions", "touch_base_mainnet", "touch_production_bittensor"],
        "rpc_url": args.rpc_url,
        "base_sepolia": {
            "chain_id": BASE_SEPOLIA_CHAIN_ID,
            "chain_name": "Base Sepolia",
            "native_currency_symbol": "ETH",
            "explorer_url": "https://sepolia.basescan.org",
        },
        "faucet_sources": [
            {
                "name": "Base network faucets",
                "url": BASE_NETWORK_FAUCETS_URL,
                "note": "Use a Base Sepolia faucet for native test ETH, not Base mainnet ETH.",
            }
        ],
        "aws": {"profile": args.aws_profile, "region": args.region},
        "funding_targets": funding_targets,
        "checks": [check.as_dict() for check in checks],
        "summary": {
            "green": sum(1 for check in checks if check.status == "green"),
            "yellow": sum(1 for check in checks if check.status == "yellow"),
            "red": sum(1 for check in checks if check.status == "red"),
        },
        "next_actions": _next_actions(checks),
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"Base SOTA testnet funding: {report['status'].upper()}")
    print(report["message"])
    print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
    for target in report.get("funding_targets") or []:
        balance = target.get("balance_eth")
        balance_text = f"{balance} ETH" if balance is not None else "unknown balance"
        minimum = target.get("minimum_balance_eth")
        minimum_text = f", minimum {minimum} ETH" if minimum is not None else ""
        print(f"- [{target.get('status')}] {target.get('label')}: {target.get('address') or 'missing'} ({balance_text}{minimum_text})")
        if target.get("remediation"):
            print(f"  next: {target['remediation']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Base SOTA Base Sepolia funding readiness report.")
    parser.add_argument("--rpc-url", default=DEFAULT_RPC_URL)
    parser.add_argument("--aws-profile", default=DEFAULT_AWS_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--deployer-secret-id", default=DEFAULT_DEPLOYER_SECRET_ID)
    parser.add_argument("--root-publisher-secret-id", default=DEFAULT_ROOT_PUBLISHER_SECRET_ID)
    parser.add_argument("--test-wallet-address", default="")
    parser.add_argument("--local-state", type=Path, default=DEFAULT_LOCAL_STATE)
    parser.add_argument(
        "--fallback-report",
        action="append",
        type=Path,
        default=None,
        help="Existing public funding/blocker report to reuse only for cached public addresses when AWS tags are unavailable. Repeatable.",
    )
    parser.add_argument("--target", action="append", default=[], help="Additional funding target as label=0xAddress. Repeatable.")
    parser.add_argument(
        "--min-balance",
        action="append",
        default=[],
        help="Required minimum balance as label=ETH. Defaults: deployer=0.020, root_publisher=0.005, test_wallet=0.005, default=0.001.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args)
    _write_json(args.report_out, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

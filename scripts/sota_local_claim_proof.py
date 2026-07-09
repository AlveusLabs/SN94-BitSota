#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen

from eth_account import Account
from web3 import Web3


REPOS = Path("/home/mekaneeky/repos")
DOCS_REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPOS / ".sota-base-local"
DEFAULT_STATE = RUN_DIR / "state.json"
DEFAULT_REPORT = RUN_DIR / "claim-proof" / "latest.json"
DEFAULT_EVIDENCE = RUN_DIR / "claim-proof" / "local-claim-tx-evidence.json"
LOCAL_PRIVATE_KEY = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
LOCAL_CHAIN_ID = 31337
LANE_ID = "base:sota-local"


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


def _join_url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _claims_base_url(state: dict[str, Any], claims_url: str) -> str:
    if claims_url:
        return claims_url.rsplit("/claims", 1)[0].rstrip("/")
    configured = str(dict(state.get("urls") or {}).get("claims_ui") or "http://127.0.0.1:3000/claims")
    return configured.rsplit("/claims", 1)[0].rstrip("/")


def _rpc_url(state: dict[str, Any], rpc_url: str) -> str:
    return rpc_url or str(dict(state.get("urls") or {}).get("anvil_rpc") or "http://127.0.0.1:8545")


def _http_json(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc
    parsed = json.loads(body) if body.strip() else {}
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {url} returned non-object JSON")
    return parsed


def _eligibility_urls(base_url: str, state: dict[str, Any]) -> dict[str, str]:
    alice = str(dict(state.get("accounts") or {}).get("alice_reward") or "")
    old_coldkey = str(dict(state.get("genesis") or {}).get("old_coldkey") or "")
    lane_id = str(dict(dict(state.get("autoresearch") or {}).get("subnet") or {}).get("id") or LANE_ID)
    genesis_query = urlencode({"old_coldkey": old_coldkey, "reward_address": alice, "subnet_id": "genesis"})
    emission_query = urlencode({"evm_address": alice, "subnet_id": lane_id})
    encoded_alice = quote(alice)
    return {
        "genesis": f"{base_url}/api/sota-claims/api/v1/base/eligibility/{encoded_alice}?{genesis_query}",
        "emission": f"{base_url}/api/sota-claims/api/v1/base/eligibility/{encoded_alice}?{emission_query}",
    }


def _unclaimed_raw(payload: dict[str, Any]) -> int:
    credits = dict(payload.get("credits") or {})
    unclaimed = dict(credits.get("unclaimed_sota") or {})
    try:
        return int(str(unclaimed.get("raw") or "0"))
    except ValueError:
        return 0


def _read_eligibility(base_url: str, state: dict[str, Any], *, timeout: float) -> dict[str, dict[str, Any]]:
    return {
        name: _http_json("GET", url, timeout=timeout)
        for name, url in _eligibility_urls(base_url, state).items()
    }


def _claim_transactions(base_url: str, state: dict[str, Any], *, timeout: float) -> dict[str, dict[str, Any]]:
    alice = str(dict(state.get("accounts") or {}).get("alice_reward") or "")
    lane_id = str(dict(dict(state.get("autoresearch") or {}).get("subnet") or {}).get("id") or LANE_ID)
    endpoint = _join_url(base_url, "/api/sota-claims/api/v1/base/claims/transaction")
    genesis = _http_json("POST", endpoint, payload={"program": "genesis", "rewardAddress": alice}, timeout=timeout)
    emission = _http_json(
        "POST",
        endpoint,
        payload={"program": "emission", "evmAddress": alice, "laneId": lane_id},
        timeout=timeout,
    )
    return {"genesis": dict(genesis.get("transaction") or {}), "emission": dict(emission.get("transaction") or {})}


def _chain_id(rpc_url: str) -> int:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    return int(w3.eth.chain_id)


def _send_claim_tx(rpc_url: str, private_key: str, tx: dict[str, Any]) -> str:
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    account = Account.from_key(private_key)
    to_address = Web3.to_checksum_address(str(tx["to"]))
    value_text = str(tx.get("value") or "0x0")
    value = int(value_text, 16 if value_text.startswith("0x") else 10)
    transaction: dict[str, Any] = {
        "to": to_address,
        "from": account.address,
        "data": str(tx["data"]),
        "value": value,
        "nonce": w3.eth.get_transaction_count(account.address),
        "chainId": int(tx.get("chainId") or w3.eth.chain_id),
        "gasPrice": int(w3.eth.gas_price),
    }
    transaction["gas"] = int(w3.eth.estimate_gas(transaction))
    signed = Account.sign_transaction(transaction, private_key)
    raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if int(receipt.status) != 1:
        raise RuntimeError(f"claim transaction reverted: {tx_hash.hex()}")
    text = tx_hash.hex()
    return text if text.startswith("0x") else f"0x{text}"


def _expect_duplicate_claim_rejected(rpc_url: str, private_key: str, tx: dict[str, Any]) -> dict[str, Any]:
    try:
        tx_hash = _send_claim_tx(rpc_url, private_key, tx)
    except Exception as exc:
        return {"rejected": True, "error": str(exc)[:500]}
    return {"rejected": False, "tx_hash": tx_hash}


def _run_evidence(
    *,
    state: Path,
    genesis_tx: str,
    emission_tx: str,
    evidence_out: Path,
    timeout: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(DOCS_REPO / "scripts" / "sota_base_claim_tx_evidence.py"),
        "--environment",
        "local",
        "--state",
        str(state),
        "--genesis-tx",
        genesis_tx,
        "--emission-tx",
        emission_tx,
        "--timeout",
        str(timeout),
        "--report-out",
        str(evidence_out),
        "--allow-blocked",
    ]
    result = subprocess.run(command, cwd=DOCS_REPO, check=False, text=True, capture_output=True, timeout=timeout + 20)
    if result.returncode != 0 and not evidence_out.exists():
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"evidence verifier exited {result.returncode}")
    return _load_json(evidence_out)


def _restart_local_stack(timeout: float) -> str:
    command = [
        sys.executable,
        str(DOCS_REPO / "scripts" / "sota_local_demo.py"),
        "launch",
        "--skip-claim-proof",
    ]
    result = subprocess.run(command, cwd=DOCS_REPO, check=False, text=True, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"local reset exited {result.returncode}")
    return result.stdout


def _summary(checks: list[Check]) -> dict[str, int]:
    return {
        "green": sum(1 for check in checks if check.status == "green"),
        "yellow": sum(1 for check in checks if check.status == "yellow"),
        "red": sum(1 for check in checks if check.status == "red"),
    }


def _status(checks: list[Check]) -> str:
    if any(check.status == "red" for check in checks):
        return "red"
    if any(check.status == "yellow" for check in checks):
        return "yellow"
    return "green"


def run_proof(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[Check] = []
    state = _load_json(args.state)
    base_url = _claims_base_url(state, args.claims_url)
    rpc_url = _rpc_url(state, args.rpc_url)
    expected_wallet = str(dict(state.get("accounts") or {}).get("alice_reward") or "")
    signer = Account.from_key(args.private_key)

    chain_id = _chain_id(rpc_url)
    checks.append(
        Check(
            "local_chain_id",
            "green" if chain_id == LOCAL_CHAIN_ID else "red",
            f"RPC chain id is local Anvil {LOCAL_CHAIN_ID}." if chain_id == LOCAL_CHAIN_ID else f"RPC chain id is {chain_id}, expected local Anvil {LOCAL_CHAIN_ID}.",
            "" if chain_id == LOCAL_CHAIN_ID else "Use the local Anvil RPC from scripts/sota_local_demo.py status.",
        )
    )
    checks.append(
        Check(
            "local_signer",
            "green" if signer.address.lower() == expected_wallet.lower() else "red",
            "Local-only private key matches the seeded tester wallet."
            if signer.address.lower() == expected_wallet.lower()
            else f"Local-only private key resolves to {signer.address}, expected {expected_wallet}.",
            "" if signer.address.lower() == expected_wallet.lower() else "Use the local-only private key printed by the handoff page.",
        )
    )

    eligibility: dict[str, dict[str, Any]] = {}
    transactions: dict[str, Any] = {}
    double_spend_checks: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    reset_stdout = ""
    if not any(check.status == "red" for check in checks):
        eligibility = _read_eligibility(base_url, state, timeout=args.timeout)
        for name, payload in eligibility.items():
            eligible = bool(payload.get("eligible"))
            unclaimed = _unclaimed_raw(payload)
            checks.append(
                Check(
                    f"{name}_unclaimed",
                    "green" if eligible and unclaimed > 0 else "red",
                    f"{name} claim is eligible and has {unclaimed} unclaimed raw SOTA units."
                    if eligible and unclaimed > 0
                    else f"{name} claim is not claimable now: eligible={eligible}, unclaimed_raw={unclaimed}.",
                    "" if eligible and unclaimed > 0 else "Reset the local stack before running a fresh claim proof.",
                )
            )

    if not any(check.status == "red" for check in checks):
        unsigned = _claim_transactions(base_url, state, timeout=args.timeout)
        genesis_tx_hash = _send_claim_tx(rpc_url, args.private_key, unsigned["genesis"])
        emission_tx_hash = _send_claim_tx(rpc_url, args.private_key, unsigned["emission"])
        transactions = {
            "genesis": {"tx_hash": genesis_tx_hash, "to": unsigned["genesis"].get("to")},
            "emission": {"tx_hash": emission_tx_hash, "to": unsigned["emission"].get("to")},
        }
        checks.append(Check("broadcast_genesis", "green", f"Broadcast local genesis claim {genesis_tx_hash}."))
        checks.append(Check("broadcast_emission", "green", f"Broadcast local emission claim {emission_tx_hash}."))
        for name in ("genesis", "emission"):
            result = _expect_duplicate_claim_rejected(rpc_url, args.private_key, unsigned[name])
            double_spend_checks[name] = result
            checks.append(
                Check(
                    f"{name}_double_spend_rejected",
                    "green" if result.get("rejected") else "red",
                    f"{name} duplicate claim was rejected by the local contract."
                    if result.get("rejected")
                    else f"{name} duplicate claim unexpectedly succeeded: {result.get('tx_hash')}.",
                    "" if result.get("rejected") else "Fix distributor claimed-leaf checks before releasing testnet testers.",
                )
            )
        evidence = _run_evidence(
            state=args.state,
            genesis_tx=genesis_tx_hash,
            emission_tx=emission_tx_hash,
            evidence_out=args.evidence_out,
            timeout=args.timeout,
        )
        checks.append(
            Check(
                "receipt_evidence",
                "green" if evidence.get("ok") else "red",
                f"Receipt evidence is green: {evidence.get('summary')}."
                if evidence.get("ok")
                else f"Receipt evidence is not green: {evidence.get('summary')}.",
                "" if evidence.get("ok") else "Inspect the local claim transaction evidence report.",
            )
        )

    if args.reset_after and transactions:
        try:
            reset_stdout = _restart_local_stack(args.reset_timeout)
            checks.append(Check("reset_after", "green", "Reset the local stack after proof so the next tester gets unclaimed claims."))
        except Exception as exc:
            checks.append(
                Check(
                    "reset_after",
                    "red",
                    f"Local proof ran, but reset failed: {exc}",
                    "Run python3 scripts/sota_local_demo.py launch before handing the demo to another tester.",
                )
            )

    status = _status(checks)
    report = {
        "schema": "sota-local-claim-proof/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": status == "green",
        "status": status,
        "message": (
            "Local claim proof submitted real genesis and emission claims and verified receipts."
            if status == "green"
            else "Local claim proof did not complete cleanly."
        ),
        "environment": "local",
        "state_changing": True,
        "does_not": ["touch_production_bittensor", "touch_base_mainnet", "touch_base_sepolia", "use_mock_claims"],
        "claims_base_url": base_url,
        "rpc_url": rpc_url,
        "wallet_address": expected_wallet,
        "eligibility": eligibility,
        "transactions": transactions,
        "double_spend_checks": double_spend_checks,
        "evidence_report": str(args.evidence_out),
        "evidence_summary": evidence.get("summary") if isinstance(evidence, dict) else {},
        "reset_after": bool(args.reset_after),
        "reset_stdout_tail": reset_stdout[-2000:] if reset_stdout else "",
        "checks": [check.as_dict() for check in checks],
        "summary": _summary(checks),
        "next_actions": [check.remediation for check in checks if check.status != "green" and check.remediation],
    }
    _write_json(args.report_out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit and verify real local SOTA genesis/emission claims.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--claims-url", default="")
    parser.add_argument("--rpc-url", default="")
    parser.add_argument("--private-key", default=LOCAL_PRIVATE_KEY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--evidence-out", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--reset-after", action="store_true", help="restart the local stack after recording proof evidence")
    parser.add_argument("--reset-timeout", type=float, default=180.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)

    report = run_proof(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"SOTA local claim proof: {report['status'].upper()}")
        print(report["message"])
        print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
        print(f"Report: {args.report_out}")
        if report["transactions"]:
            print(f"Genesis tx: {report['transactions']['genesis']['tx_hash']}")
            print(f"Emission tx: {report['transactions']['emission']['tx_hash']}")
        for action in report["next_actions"][:6]:
            print(f"- next: {action}")
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

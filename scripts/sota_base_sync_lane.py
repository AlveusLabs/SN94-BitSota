#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from eth_account import Account
from web3 import Web3


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_MANIFEST = DEFAULT_ARTIFACTS_DIR / "base-sepolia-deployment-manifest.json"
DEFAULT_ROOT_ARTIFACT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-emission-root-artifact.json"
DEFAULT_CLAIM_ARTIFACT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-emission-claim-template.json"
DEFAULT_ABI_DIR = REPOS / "Pool" / "contracts" / "sota-base" / "abi"
ZERO32 = "0x" + "00" * 32


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _abi(name: str, abi_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((abi_dir / f"{name}.json").read_text(encoding="utf-8"))
    abi = payload.get("abi") if isinstance(payload, dict) else payload
    if not isinstance(abi, list):
        raise ValueError(f"{name}.json does not contain an ABI list")
    return abi


def _manifest_contract(manifest: dict[str, Any], key: str) -> str:
    value = dict(manifest.get("contracts") or {}).get(key)
    if isinstance(value, dict):
        value = value.get("address")
    if not isinstance(value, str) or not Web3.is_address(value):
        raise ValueError(f"manifest contract {key!r} is missing or invalid")
    return Web3.to_checksum_address(value)


def _manifest_rpc(manifest: dict[str, Any]) -> str:
    chain = dict(manifest.get("chain") or {})
    rpc = str(chain.get("public_browser_rpc_url") or "").strip()
    if not rpc:
        raise ValueError("manifest chain.public_browser_rpc_url is missing")
    return rpc


def _hex32(value: Any, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if not text.startswith("0x"):
        text = "0x" + text
    if len(text) != 66:
        raise ValueError(f"{field_name} must be bytes32 hex")
    int(text[2:], 16)
    return text


def _root_value(root_artifact: dict[str, Any], *keys: str) -> Any:
    root = dict(root_artifact.get("root") or {})
    for key in keys:
        if root.get(key) is not None:
            return root[key]
    for key in keys:
        if root_artifact.get(key) is not None:
            return root_artifact[key]
    return None


def _uint(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} is missing")
    number = int(str(value), 10)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _claim_list(claim_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    claims = claim_artifact.get("claim_list")
    if not isinstance(claims, list) or not claims:
        raise ValueError("emission claim artifact must contain a non-empty claim_list")
    rows = []
    for row in claims:
        if not isinstance(row, dict):
            raise ValueError("claim_list rows must be objects")
        rows.append(row)
    return rows


def _claim_lane_id(claim_artifact: dict[str, Any]) -> str:
    first = _claim_list(claim_artifact)[0]
    lane = (
        first.get("offchain_lane_id")
        or first.get("offchainLaneId")
        or first.get("offchain_subnet_id")
        or dict(claim_artifact.get("subnet") or {}).get("offchain_lane_id")
        or dict(claim_artifact.get("subnet") or {}).get("offchain_subnet_id")
    )
    return _hex32(lane, "offchain_lane_id")


def _private_key_from_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required when --broadcast is used")
    if value.startswith("{"):
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError(f"{name} JSON value must be an object")
        for key in (name, "private_key", "deployer_private_key", "sota_deployer_private_key"):
            candidate = str(payload.get(key) or "").strip()
            if candidate:
                value = candidate
                break
    if not value.startswith("0x"):
        value = "0x" + value
    if len(value) != 66:
        raise ValueError(f"{name} must be a 32-byte private key")
    return value


def _tx_params(w3: Web3, account: str, *, gas: int) -> dict[str, Any]:
    gas_price = int(w3.eth.gas_price)
    return {
        "from": account,
        "nonce": int(w3.eth.get_transaction_count(account)),
        "chainId": int(w3.eth.chain_id),
        "gas": gas,
        "maxFeePerGas": gas_price * 3,
        "maxPriorityFeePerGas": gas_price,
    }


def sync_lane(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_json(args.manifest)
    root_artifact = _load_json(args.root_artifact)
    claim_artifact = _load_json(args.claim_artifact)

    rpc_url = args.rpc_url or _manifest_rpc(manifest)
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": args.timeout}))
    if not w3.is_connected():
        raise RuntimeError(f"could not connect to RPC: {rpc_url}")
    chain_id = int(w3.eth.chain_id)
    if chain_id != 84532 and not args.allow_non_testnet:
        raise RuntimeError(f"refusing to sync a lane on chain {chain_id}; expected Base Sepolia 84532")

    lane_registry = _manifest_contract(manifest, "lane_registry")
    lane = w3.eth.contract(address=lane_registry, abi=_abi("SOTALaneRegistry", args.abi_dir))
    lane_id = _claim_lane_id(claim_artifact)
    desired_budget = _uint(_root_value(root_artifact, "budget", "budget_cap", "total_amount_units"), "budget")
    desired_policy_hash = _hex32(_root_value(root_artifact, "policy_hash"), "policy_hash")

    current = lane.functions.getLane(bytes.fromhex(lane_id[2:])).call()
    current_record = {
        "budget_units": str(int(current[1])),
        "active": bool(current[2]),
        "policy_hash": "0x" + bytes(current[3]).hex(),
        "updated_at": int(current[5]),
    }
    desired_record = {
        "budget_units": str(desired_budget),
        "active": True,
        "policy_hash": desired_policy_hash,
    }
    already_synced = (
        int(current[1]) == desired_budget
        and bool(current[2])
        and ("0x" + bytes(current[3]).hex()).lower() == desired_policy_hash.lower()
    )

    tx = lane.functions.setLane(
        bytes.fromhex(lane_id[2:]),
        desired_budget,
        True,
        bytes.fromhex(desired_policy_hash[2:]),
    )
    tx_data = tx._encode_transaction_data()
    receipt_payload: dict[str, Any] | None = None
    tx_hash = ""
    status = "green" if already_synced else "yellow"

    if args.broadcast and not already_synced:
        account = Account.from_key(_private_key_from_env(args.private_key_env))
        owner = lane.functions.owner().call()
        if Web3.to_checksum_address(account.address) != Web3.to_checksum_address(owner):
            raise RuntimeError(f"{args.private_key_env} signer is not the lane registry owner")
        gas = int(tx.estimate_gas({"from": account.address}) * 1.3)
        built = tx.build_transaction(_tx_params(w3, account.address, gas=gas))
        signed = account.sign_transaction(built)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction).hex()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=args.receipt_timeout)
        receipt_payload = {
            "status": int(receipt.status),
            "block_number": int(receipt.blockNumber),
            "gas_used": int(receipt.gasUsed),
        }
        if int(receipt.status) != 1:
            status = "red"
        else:
            status = "green"
            refreshed = lane.functions.getLane(bytes.fromhex(lane_id[2:])).call(block_identifier="latest")
            current_record = {
                "budget_units": str(int(refreshed[1])),
                "active": bool(refreshed[2]),
                "policy_hash": "0x" + bytes(refreshed[3]).hex(),
                "updated_at": int(refreshed[5]),
            }

    ok = status == "green"
    report = {
        "schema": "sota-base-lane-sync/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "status": status,
        "environment": "base-sepolia" if chain_id == 84532 else f"chain-{chain_id}",
        "chain_id": chain_id,
        "read_only": not args.broadcast,
        "does_not": ["touch_production_bittensor", "touch_base_mainnet", "print_private_keys"],
        "lane_registry": lane_registry,
        "offchain_lane_id": lane_id,
        "current": current_record,
        "desired": desired_record,
        "transaction": {
            "to": lane_registry,
            "data": tx_data,
            "value": "0x0",
            "chainId": chain_id,
            "tx_hash": tx_hash,
            "receipt": receipt_payload,
        },
        "message": (
            "Emission lane cap matches the emission root budget."
            if ok
            else "Emission lane cap is not synced; broadcast this transaction before public emission claims."
        ),
    }
    if args.out:
        _write_json(args.out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync the Base SOTA emission lane cap to an emission root artifact.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root-artifact", type=Path, default=DEFAULT_ROOT_ARTIFACT)
    parser.add_argument("--claim-artifact", type=Path, default=DEFAULT_CLAIM_ARTIFACT)
    parser.add_argument("--abi-dir", type=Path, default=DEFAULT_ABI_DIR)
    parser.add_argument("--rpc-url", default="")
    parser.add_argument("--private-key-env", default="SOTA_DEPLOYER_PRIVATE_KEY")
    parser.add_argument("--broadcast", action="store_true")
    parser.add_argument("--allow-non-testnet", action="store_true")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--receipt-timeout", type=float, default=180.0)
    parser.add_argument("--out", type=Path, default=Path(""))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)

    report = sync_lane(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"SOTA lane sync: {str(report['status']).upper()}")
        print(report["message"])
        print(f"Lane: {report['offchain_lane_id']}")
        print(f"Current budget: {report['current']['budget_units']}")
        print(f"Desired budget: {report['desired']['budget_units']}")
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

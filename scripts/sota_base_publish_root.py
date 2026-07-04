#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.request import Request, urlopen

from eth_abi import encode
from eth_utils import keccak


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_MANIFEST = DEFAULT_ARTIFACTS_DIR / "base-sepolia-deployment-manifest.json"
BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
KIND_BY_NAME = {"genesis": 1, "emission": 2}
EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
HEX32_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")
ZERO32 = "0x" + ("00" * 32)
ROOT_PUBLISHED_TOPIC = "0x" + keccak(
    text="RootPublished(bytes32,uint8,bytes32,uint256,bytes32,bytes32,bytes32,address)"
).hex()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _address(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not EVM_ADDRESS_RE.fullmatch(text):
        raise ValueError(f"{field} must be a 0x-prefixed EVM address")
    if text.lower() == "0x0000000000000000000000000000000000000000":
        raise ValueError(f"{field} must not be the zero address")
    return text


def _bytes32(value: Any, field: str, *, allow_zero: bool = False) -> str:
    text = str(value or "").strip()
    if not HEX32_RE.fullmatch(text):
        raise ValueError(f"{field} must be a 32-byte hex string")
    if not text.startswith("0x"):
        text = "0x" + text
    text = text.lower()
    if not allow_zero and text == ZERO32:
        raise ValueError(f"{field} must not be zero")
    return text


def _uint(value: Any, field: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"{field} must be positive")
    return out


def _json_rpc(rpc_url: str, method: str, params: list[Any] | None = None, *, timeout: float = 20.0) -> Any:
    request = Request(
        rpc_url,
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload.get("result")


def _rpc_chain_id(rpc_url: str, *, timeout: float) -> int:
    raw = _json_rpc(rpc_url, "eth_chainId", timeout=timeout)
    return int(str(raw), 16)


def _manifest_chain_id(manifest: dict[str, Any]) -> int:
    return int(dict(manifest.get("chain") or {}).get("chain_id") or 0)


def _manifest_rpc_url(manifest: dict[str, Any], args: argparse.Namespace) -> str:
    if args.rpc_url:
        return str(args.rpc_url)
    chain = dict(manifest.get("chain") or {})
    return str(chain.get("public_browser_rpc_url") or "https://sepolia.base.org")


def _root_registry_address(manifest: dict[str, Any], args: argparse.Namespace) -> str:
    if args.root_registry:
        return _address(args.root_registry, "root_registry")
    browser_safe = dict(dict(manifest.get("browser_safe") or {}).get("contract_addresses") or {})
    contracts = dict(manifest.get("contracts") or {})
    value = browser_safe.get("root_registry") or dict(contracts.get("root_registry") or {}).get("address")
    return _address(value, "root_registry")


def _pick(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return None


def _artifact_root_value(artifact: dict[str, Any]) -> Any:
    root = artifact.get("root")
    if isinstance(root, dict):
        return _pick(root.get("root"), root.get("merkle_root"))
    merkle = artifact.get("merkle")
    if isinstance(merkle, dict):
        return _pick(merkle.get("root"), merkle.get("merkle_root"))
    return _pick(root, artifact.get("merkle_root"))


def _artifact_budget_value(artifact: dict[str, Any]) -> Any:
    root = artifact.get("root") if isinstance(artifact.get("root"), dict) else {}
    merkle = artifact.get("merkle") if isinstance(artifact.get("merkle"), dict) else {}
    return _pick(
        dict(root).get("total_amount_units"),
        dict(root).get("total_amount"),
        dict(root).get("budget"),
        dict(root).get("budget_cap"),
        artifact.get("total_amount_units"),
        artifact.get("total_amount"),
        artifact.get("budget"),
        artifact.get("budget_cap"),
        dict(merkle).get("total_units"),
    )


def _artifact_policy_hash_value(artifact: dict[str, Any]) -> Any:
    root = artifact.get("root") if isinstance(artifact.get("root"), dict) else {}
    return _pick(dict(root).get("policy_hash"), artifact.get("policy_hash"))


def _artifact_attestation_hash_value(artifact: dict[str, Any]) -> Any:
    root = artifact.get("root") if isinstance(artifact.get("root"), dict) else {}
    attestation = artifact.get("attestation") if isinstance(artifact.get("attestation"), dict) else {}
    return _pick(dict(root).get("attestation_hash"), artifact.get("attestation_hash"), dict(attestation).get("hash"))


def _artifact_nonce_value(artifact: dict[str, Any]) -> Any:
    root = artifact.get("root") if isinstance(artifact.get("root"), dict) else {}
    return _pick(dict(root).get("nonce"), artifact.get("nonce"))


def _calldata(*, kind: int, merkle_root: str, budget_cap: int, policy_hash: str, attestation_hash: str, nonce: str) -> str:
    selector = keccak(text="publishRoot(uint8,bytes32,uint256,bytes32,bytes32,bytes32)")[:4]
    encoded = encode(
        ["uint8", "bytes32", "uint256", "bytes32", "bytes32", "bytes32"],
        [
            int(kind),
            bytes.fromhex(merkle_root[2:]),
            int(budget_cap),
            bytes.fromhex(policy_hash[2:]),
            bytes.fromhex(attestation_hash[2:]),
            bytes.fromhex(nonce[2:]),
        ],
    )
    return "0x" + (selector + encoded).hex()


def _topic_hex(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        hex_method = getattr(value, "hex", None)
        text = hex_method() if callable(hex_method) else str(value)
    return text if text.startswith("0x") else f"0x{text}"


def _root_published_event(receipt: Any, *, root_registry: str) -> dict[str, Any] | None:
    registry = str(root_registry).lower()
    for log in getattr(receipt, "logs", []) or []:
        address = str(getattr(log, "address", "") or log.get("address", "")).lower()
        if address != registry:
            continue
        topics = list(getattr(log, "topics", None) or log.get("topics", []) or [])
        if len(topics) < 4 or _topic_hex(topics[0]).lower() != ROOT_PUBLISHED_TOPIC.lower():
            continue
        return {
            "root_id": _topic_hex(topics[1]).lower(),
            "kind_id": int(_topic_hex(topics[2]), 16),
            "merkle_root": _topic_hex(topics[3]).lower(),
            "log_index": int(getattr(log, "logIndex", None) if getattr(log, "logIndex", None) is not None else log.get("logIndex", 0)),
        }
    return None


def build_publish_request(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_json(args.manifest)
    artifact = _load_json(args.root_artifact)
    manifest_chain_id = _manifest_chain_id(manifest)
    if manifest_chain_id == BASE_MAINNET_CHAIN_ID:
        raise ValueError("manifest points at Base mainnet chain id 8453")
    if manifest_chain_id != BASE_SEPOLIA_CHAIN_ID and not args.allow_local:
        raise ValueError(f"manifest chain id must be Base Sepolia 84532, got {manifest_chain_id}")
    if manifest_chain_id not in {BASE_SEPOLIA_CHAIN_ID, 31337}:
        raise ValueError(f"unsupported chain id {manifest_chain_id}")

    kind = KIND_BY_NAME[args.kind]
    root_registry = _root_registry_address(manifest, args)
    merkle_root = _bytes32(args.merkle_root or _artifact_root_value(artifact), "merkle_root")
    budget_cap = _uint(args.budget_cap or _artifact_budget_value(artifact), "budget_cap")
    policy_hash = _bytes32(args.policy_hash or _artifact_policy_hash_value(artifact), "policy_hash")
    attestation_hash = _bytes32(args.attestation_hash or _artifact_attestation_hash_value(artifact), "attestation_hash")
    nonce = _bytes32(args.nonce or _artifact_nonce_value(artifact), "nonce")
    data = _calldata(
        kind=kind,
        merkle_root=merkle_root,
        budget_cap=budget_cap,
        policy_hash=policy_hash,
        attestation_hash=attestation_hash,
        nonce=nonce,
    )
    return {
        "schema": "sota-base-root-publish-request/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "base-sepolia" if manifest_chain_id == BASE_SEPOLIA_CHAIN_ID else "local",
        "chain_id": manifest_chain_id,
        "kind": args.kind,
        "kind_id": kind,
        "to": root_registry,
        "root_registry": root_registry,
        "merkle_root": merkle_root,
        "budget_cap": str(budget_cap),
        "policy_hash": policy_hash,
        "attestation_hash": attestation_hash,
        "nonce": nonce,
        "transaction": {"to": root_registry, "data": data, "value": "0x0", "chainId": manifest_chain_id},
        "source": {"manifest": str(args.manifest), "root_artifact": str(args.root_artifact)},
        "does_not": ["touch_production_bittensor", "touch_base_mainnet"],
    }


def broadcast_publish_request(request: dict[str, Any], *, rpc_url: str, timeout: float) -> dict[str, Any]:
    from web3 import Web3

    private_key = os.environ.get("SOTA_ROOT_PUBLISHER_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("SOTA_ROOT_PUBLISHER_PRIVATE_KEY is required for --broadcast")
    if int(request["chain_id"]) == BASE_MAINNET_CHAIN_ID:
        raise RuntimeError("refusing to broadcast to Base mainnet")
    rpc_chain_id = _rpc_chain_id(rpc_url, timeout=timeout)
    if int(rpc_chain_id) != int(request["chain_id"]):
        raise RuntimeError(f"RPC chain id {rpc_chain_id} does not match request chain id {request['chain_id']}")
    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout}))
    account = w3.eth.account.from_key(private_key)
    tx = {
        "chainId": int(request["chain_id"]),
        "from": account.address,
        "to": Web3.to_checksum_address(str(request["to"])),
        "data": request["transaction"]["data"],
        "value": 0,
        "nonce": w3.eth.get_transaction_count(account.address),
    }
    tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
    tx["maxFeePerGas"] = int(w3.eth.gas_price * 2)
    tx["maxPriorityFeePerGas"] = int(w3.to_wei(0.01, "gwei"))
    signed = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
    root_event = _root_published_event(receipt, root_registry=str(request["root_registry"]))
    if root_event is None:
        raise RuntimeError("broadcast succeeded but RootPublished event was not found in the receipt")
    return {
        **request,
        "status": "broadcasted" if int(receipt.status) == 1 else "reverted",
        "publisher": account.address,
        "tx_hash": tx_hash.hex(),
        "root_id": root_event["root_id"],
        "root_published_event": root_event,
        "receipt": {
            "status": int(receipt.status),
            "block_number": int(receipt.blockNumber),
            "gas_used": int(receipt.gasUsed),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or broadcast a guarded Base SOTA root publish transaction.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root-artifact", type=Path, required=True)
    parser.add_argument("--kind", choices=("genesis", "emission"), required=True)
    parser.add_argument("--root-registry", default="")
    parser.add_argument("--merkle-root", default="")
    parser.add_argument("--budget-cap", default="")
    parser.add_argument("--policy-hash", default="")
    parser.add_argument("--attestation-hash", default="")
    parser.add_argument("--nonce", default="")
    parser.add_argument("--rpc-url", default="")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--allow-local", action="store_true", help="Allow local chain id 31337 for development only.")
    parser.add_argument("--broadcast", action="store_true", help="Sign and broadcast with SOTA_ROOT_PUBLISHER_PRIVATE_KEY.")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-root-publish-request.json")
    args = parser.parse_args(argv)

    request = build_publish_request(args)
    if args.broadcast:
        manifest = _load_json(args.manifest)
        result = broadcast_publish_request(request, rpc_url=_manifest_rpc_url(manifest, args), timeout=args.timeout)
    else:
        result = {
            **request,
            "status": "dry_run",
            "does_not": [*request["does_not"], "sign", "broadcast_transactions"],
        }
    _write_json(args.out, result)
    print(json.dumps({"ok": result["status"] in {"dry_run", "broadcasted"}, "status": result["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if result["status"] in {"dry_run", "broadcasted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

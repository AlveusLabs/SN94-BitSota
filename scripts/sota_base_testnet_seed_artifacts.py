#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any

from eth_abi import encode
from eth_utils import keccak


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_MANIFEST = DEFAULT_ARTIFACTS_DIR / "base-sepolia-deployment-manifest.json"
BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
ONE_SOTA = 10**18
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
ZERO32 = "0x" + ("00" * 32)
EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
HEX32_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _keccak_json(value: Any) -> str:
    return "0x" + keccak(_canonical_json(value)).hex()


def _hex32(value: Any, field: str, *, allow_zero: bool = False) -> str:
    text = str(value or "").strip()
    if not HEX32_RE.fullmatch(text):
        raise ValueError(f"{field} must be a 32-byte hex string")
    if not text.startswith("0x"):
        text = "0x" + text
    text = text.lower()
    if not allow_zero and text == ZERO32:
        raise ValueError(f"{field} must not be zero")
    return text


def _address(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not EVM_ADDRESS_RE.fullmatch(text) or text.lower() == ZERO_ADDRESS:
        raise ValueError(f"{field} must be a nonzero EVM address")
    return text


def _normalize_address(value: Any, field: str) -> str:
    return _address(value, field).lower()


def _units_from_sota(value: Any, field: str) -> int:
    try:
        units = Decimal(str(value)) * Decimal(ONE_SOTA)
    except Exception as exc:
        raise ValueError(f"{field} must be a decimal SOTA amount") from exc
    if units <= 0 or units != units.to_integral_value():
        raise ValueError(f"{field} must resolve to a positive integer unit amount")
    return int(units)


def _positive_int(value: Any, field: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"{field} must be positive")
    return out


def _manifest_chain_id(manifest: dict[str, Any]) -> int:
    return int(dict(manifest.get("chain") or {}).get("chain_id") or 0)


def _assert_base_sepolia_manifest(manifest: dict[str, Any]) -> None:
    chain_id = _manifest_chain_id(manifest)
    if chain_id == BASE_MAINNET_CHAIN_ID:
        raise ValueError("manifest points at Base mainnet chain id 8453")
    if chain_id != BASE_SEPOLIA_CHAIN_ID or manifest.get("environment") != "base-sepolia":
        raise ValueError("manifest must be pinned to Base Sepolia chain id 84532")


def _manifest_contract(manifest: dict[str, Any], key: str) -> str:
    browser_safe = dict(dict(manifest.get("browser_safe") or {}).get("contract_addresses") or {})
    contracts = dict(manifest.get("contracts") or {})
    return _address(browser_safe.get(key) or dict(contracts.get(key) or {}).get("address"), f"contracts.{key}.address")


def _manifest_default_lane_id(manifest: dict[str, Any]) -> str:
    claims_ui = dict(dict(manifest.get("services") or {}).get("claims_ui") or {})
    browser_env = dict(claims_ui.get("browser_safe_env") or {})
    value = str(browser_env.get("NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID") or "").strip()
    return value


def _bytes32(value: str) -> bytes:
    return bytes.fromhex(_hex32(value, "bytes32")[2:])


def _hash_pair(left: str, right: str) -> str:
    left_bytes = _bytes32(left)
    right_bytes = _bytes32(right)
    first, second = sorted((left_bytes, right_bytes))
    return "0x" + keccak(first + second).hex()


def _merkle_layers(leaves: list[str]) -> list[list[str]]:
    if not leaves:
        raise ValueError("cannot build a Merkle root with no leaves")
    layers = [[_hex32(leaf, "leaf") for leaf in leaves]]
    while len(layers[-1]) > 1:
        layer = layers[-1]
        next_layer: list[str] = []
        for index in range(0, len(layer), 2):
            if index + 1 >= len(layer):
                next_layer.append(layer[index])
            else:
                next_layer.append(_hash_pair(layer[index], layer[index + 1]))
        layers.append(next_layer)
    return layers


def _merkle_root(leaves: list[str]) -> str:
    return _merkle_layers(leaves)[-1][0]


def _merkle_proof(index: int, layers: list[list[str]]) -> list[str]:
    if index < 0 or index >= len(layers[0]):
        raise ValueError("leaf index out of range")
    proof: list[str] = []
    position = index
    for layer in layers[:-1]:
        sibling = position ^ 1
        if sibling < len(layer):
            proof.append(layer[sibling])
        position //= 2
    return proof


def _genesis_leaf(account: str, amount: int, allocation_hash: str) -> str:
    encoded = encode(
        ["string", "address", "uint256", "bytes32"],
        ["SOTA_GENESIS_CLAIM", _normalize_address(account, "genesis account"), int(amount), _bytes32(allocation_hash)],
    )
    return "0x" + keccak(encoded).hex()


def _emission_leaf(*, epoch: int, offchain_lane_id: str, account: str, amount: int, reward_hash: str) -> str:
    encoded = encode(
        ["string", "uint64", "bytes32", "address", "uint256", "bytes32"],
        [
            "SOTA_EMISSION_CLAIM",
            int(epoch),
            _bytes32(offchain_lane_id),
            _normalize_address(account, "emission account"),
            int(amount),
            _bytes32(reward_hash),
        ],
    )
    return "0x" + keccak(encoded).hex()


def _allocation_hash(*, old_coldkey: str, reward_address: str, tao_credit: int, alpha_credit: int) -> str:
    encoded = encode(
        ["string", "string", "address", "uint256", "uint256"],
        [
            "SOTA_BASE_SEPOLIA_GENESIS_SEED",
            str(old_coldkey),
            _normalize_address(reward_address, "reward_address"),
            int(tao_credit),
            int(alpha_credit),
        ],
    )
    return "0x" + keccak(encoded).hex()


def _published_root_id(path: Path) -> str:
    payload = _load_json(path)
    root_id = payload.get("root_id")
    if not root_id and isinstance(payload.get("root_published_event"), dict):
        root_id = dict(payload["root_published_event"]).get("root_id")
    if str(payload.get("status") or "") != "broadcasted":
        raise ValueError(f"{path} must be a broadcasted root publish result")
    return _hex32(root_id, f"{path}.root_id")


def _consensus_by_index(bundle: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for raw in list(bundle.get("claim_evidence") or []):
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except Exception:
            continue
        evidence = dict(raw.get("evidence") or {})
        consensus = dict(evidence.get("self_validation_consensus") or {})
        if consensus:
            out[index] = consensus
    return out


def _assert_self_validated(bundle: dict[str, Any], *, min_accepted: int, min_committee: int) -> None:
    claims = list(bundle.get("claim_list") or [])
    consensus_by_index = _consensus_by_index(bundle)
    failures: list[str] = []
    for position, raw_claim in enumerate(claims):
        if not isinstance(raw_claim, dict):
            failures.append(f"{position}: claim is not an object")
            continue
        index = int(raw_claim.get("index", position))
        consensus = dict(consensus_by_index.get(index) or {})
        status = str(consensus.get("status") or "").lower()
        accepted_count = int(consensus.get("accepted_count") or 0)
        committee_count = int(consensus.get("committee_count") or consensus.get("committee_size") or 0)
        if status != "accepted" or accepted_count < min_accepted or committee_count < min_committee:
            failures.append(
                f"{index}: status={status or 'missing'}, accepted_count={accepted_count}, committee_count={committee_count}"
            )
    if failures:
        raise ValueError(
            "emission evidence is not backed by accepted self-validation consensus: " + "; ".join(failures)
        )


def _normalized_emission_artifact_inputs(
    evidence_payload: dict[str, Any],
    *,
    expected_wallet: str,
    min_accepted: int,
    min_committee: int,
) -> dict[str, Any]:
    response_root = dict(evidence_payload.get("root") or {})
    bundle = dict(evidence_payload.get("bundle") or evidence_payload)
    if not bundle:
        raise ValueError("emission evidence bundle is required")
    _assert_self_validated(bundle, min_accepted=min_accepted, min_committee=min_committee)

    subnet = dict(bundle.get("subnet") or {})
    merkle = dict(bundle.get("merkle") or {})
    claim_list = list(bundle.get("claim_list") or [])
    if not claim_list:
        raise ValueError("emission evidence claim_list is required")

    epoch = int(response_root.get("epoch") or bundle.get("epoch") or claim_list[0].get("epoch") or 0)
    if epoch <= 0:
        raise ValueError("emission epoch must be positive")
    root = _hex32(response_root.get("root") or merkle.get("root"), "emission root")
    policy_hash = _hex32(response_root.get("policy_hash") or subnet.get("policy_hash") or claim_list[0].get("policy_hash"), "emission policy_hash")
    evidence_hash = str(response_root.get("evidence_hash") or merkle.get("evidence_hash") or _keccak_json(bundle)).removeprefix("0x")
    attestation_hash = _hex32("0x" + evidence_hash[:64], "emission attestation_hash")
    subnet_id = str(response_root.get("subnet_id") or subnet.get("id") or claim_list[0].get("subnet_id") or "").strip()
    if not subnet_id:
        raise ValueError("emission subnet id is required")

    normalized_claims: list[dict[str, Any]] = []
    leaves: list[str] = []
    has_expected_wallet = False
    for position, raw_claim in enumerate(claim_list):
        if not isinstance(raw_claim, dict):
            raise ValueError(f"claim_list[{position}] must be an object")
        claim = dict(raw_claim)
        index = int(claim.get("index", position))
        reward_address = _normalize_address(claim.get("reward_address"), f"claim_list[{position}].reward_address")
        amount = _positive_int(claim.get("amount_units") or claim.get("amount"), f"claim_list[{position}].amount_units")
        reward_hash = _hex32(claim.get("reward_hash"), f"claim_list[{position}].reward_hash")
        offchain_lane_id = _hex32(
            claim.get("offchain_lane_id") or claim.get("offchain_subnet_id") or subnet.get("offchain_lane_id"),
            f"claim_list[{position}].offchain_lane_id",
        )
        leaf = _emission_leaf(
            epoch=int(claim.get("epoch") or epoch),
            offchain_lane_id=offchain_lane_id,
            account=reward_address,
            amount=amount,
            reward_hash=reward_hash,
        )
        claim.update(
            {
                "kind": "emission",
                "version": claim.get("version") or "sota-emission-claim-v1",
                "index": index,
                "epoch": int(claim.get("epoch") or epoch),
                "reward_address": reward_address,
                "account": reward_address,
                "amount_units": amount,
                "offchain_lane_id": offchain_lane_id,
                "offchain_subnet_id": offchain_lane_id,
                "reward_hash": reward_hash,
                "leaf": leaf,
            }
        )
        has_expected_wallet = has_expected_wallet or reward_address == expected_wallet.lower()
        normalized_claims.append(claim)
        leaves.append(leaf)

    layers = _merkle_layers(leaves)
    recomputed_root = layers[-1][0]
    if recomputed_root != root:
        raise ValueError(f"emission root mismatch: evidence={root}, recomputed={recomputed_root}")
    total_amount = sum(int(claim["amount_units"]) for claim in normalized_claims)
    declared_total = int(response_root.get("total_amount_units") or merkle.get("total_amount_units") or total_amount)
    if declared_total != total_amount:
        raise ValueError(f"emission total mismatch: declared={declared_total}, recomputed={total_amount}")
    if not has_expected_wallet:
        raise ValueError("seeded test wallet is not present in the emission claim list")

    leaf_records = [
        {"index": int(claim["index"]), "leaf": claim["leaf"], "proof": _merkle_proof(position, layers)}
        for position, claim in enumerate(normalized_claims)
    ]
    for claim, leaf_record in zip(normalized_claims, leaf_records, strict=True):
        claim["proof"] = list(leaf_record["proof"])

    return {
        "root": root,
        "policy_hash": policy_hash,
        "attestation_hash": attestation_hash,
        "nonce": _keccak_json(
            {
                "schema": "sota-base-root-nonce/v1",
                "kind": "emission",
                "chain_id": BASE_SEPOLIA_CHAIN_ID,
                "subnet_id": subnet_id,
                "epoch": epoch,
                "root": root,
                "evidence_hash": evidence_hash,
            }
        ),
        "subnet_id": subnet_id,
        "epoch": epoch,
        "total_amount_units": total_amount,
        "claim_list": normalized_claims,
        "leaves": leaf_records,
        "subnet": subnet,
        "bundle": bundle,
        "evidence_hash": "0x" + evidence_hash[:64],
    }


def _genesis_inputs(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    reward_address = _normalize_address(args.test_wallet_address, "test_wallet_address")
    tao_credit = _units_from_sota(args.tao_credit_sota, "tao_credit_sota")
    alpha_credit = _units_from_sota(args.alpha_synthetic_credit_sota, "alpha_synthetic_credit_sota")
    amount = tao_credit + alpha_credit
    allocation_hash = _allocation_hash(
        old_coldkey=args.test_old_coldkey,
        reward_address=reward_address,
        tao_credit=tao_credit,
        alpha_credit=alpha_credit,
    )
    leaf = _genesis_leaf(reward_address, amount, allocation_hash)
    return {
        "old_coldkey": str(args.test_old_coldkey),
        "reward_address": reward_address,
        "tao_credit": tao_credit,
        "alpha_synthetic_credit": alpha_credit,
        "amount": amount,
        "allocation_hash": allocation_hash,
        "leaf": leaf,
        "policy_hash": args.genesis_policy_hash
        or _keccak_json(
            {
                "schema": "sota-base-genesis-policy/v1",
                "conversion_rate": "1 TAO = 1 SOTA",
                "alpha_formula": "alpha_held_percent * tao_in_pool",
                "chain_id": BASE_SEPOLIA_CHAIN_ID,
                "root_registry": _manifest_contract(manifest, "root_registry"),
            }
        ),
        "attestation_hash": args.genesis_attestation_hash
        or _keccak_json(
            {
                "schema": "sota-base-genesis-attestation/v1",
                "old_coldkey": str(args.test_old_coldkey),
                "reward_address": reward_address,
                "tao_credit": tao_credit,
                "alpha_synthetic_credit": alpha_credit,
                "allocation_hash": allocation_hash,
            }
        ),
        "nonce": args.genesis_nonce
        or _keccak_json(
            {
                "schema": "sota-base-root-nonce/v1",
                "kind": "genesis",
                "chain_id": BASE_SEPOLIA_CHAIN_ID,
                "root": leaf,
                "allocation_hash": allocation_hash,
            }
        ),
    }


def _root_artifact(*, kind: str, root: str, total_amount: int, policy_hash: str, attestation_hash: str, nonce: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "sota-base-root-artifact/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "base-sepolia",
        "chain_id": BASE_SEPOLIA_CHAIN_ID,
        "kind": kind,
        "root": {
            "root": _hex32(root, f"{kind}.root"),
            "total_amount_units": str(_positive_int(total_amount, f"{kind}.total_amount_units")),
            "budget": str(_positive_int(total_amount, f"{kind}.budget")),
            "policy_hash": _hex32(policy_hash, f"{kind}.policy_hash"),
            "attestation_hash": _hex32(attestation_hash, f"{kind}.attestation_hash"),
            "nonce": _hex32(nonce, f"{kind}.nonce"),
            "status": "ready_to_publish",
        },
        "metadata": metadata,
        "does_not": ["sign", "broadcast_transactions", "touch_production_bittensor", "touch_base_mainnet"],
    }


def _pending_genesis_claim_artifact(genesis: dict[str, Any], *, owner: str) -> dict[str, Any]:
    return {
        "schema": "sota-base-claim-artifact/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "base-sepolia",
        "indexer_import_ready": False,
        "finalization_required": "publish genesis root and insert emitted root_id",
        "subnet": {
            "id": "genesis",
            "title": "SOTA Base Sepolia genesis fork claim",
            "owner": owner,
            "budget": str(genesis["amount"]),
            "metadata_uri": "sota://base-sepolia/genesis",
            "token": "SOTA",
        },
        "root": {
            "root_id": None,
            "subnet_id": "genesis",
            "epoch": 0,
            "root": genesis["leaf"],
            "total_amount_units": str(genesis["amount"]),
            "budget": str(genesis["amount"]),
            "policy_hash": genesis["policy_hash"],
            "attestation_hash": genesis["attestation_hash"],
            "nonce": genesis["nonce"],
            "status": "finalized",
            "validation_status": "accepted",
        },
        "allocations": [
            {
                "kind": "genesis",
                "version": "sota-genesis-claim-v1",
                "index": 0,
                "account": genesis["reward_address"],
                "reward_address": genesis["reward_address"],
                "amount": str(genesis["amount"]),
                "amount_units": str(genesis["amount"]),
                "allocation_hash": genesis["allocation_hash"],
                "old_coldkey": genesis["old_coldkey"],
                "tao_credit": str(genesis["tao_credit"]),
                "alpha_synthetic_credit": str(genesis["alpha_synthetic_credit"]),
                "leaf": genesis["leaf"],
                "proof": [],
            }
        ],
        "leaves": [{"index": 0, "leaf": genesis["leaf"], "proof": []}],
    }


def _pending_emission_claim_artifact(emission: dict[str, Any], *, owner: str) -> dict[str, Any]:
    subnet = {
        **dict(emission["subnet"] or {}),
        "id": emission["subnet_id"],
        "owner": owner,
        "token": "SOTA",
    }
    if not subnet.get("title"):
        subnet["title"] = f"SOTA Base Sepolia lane {emission['subnet_id']}"
    return {
        "schema": "sota-base-claim-artifact/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "base-sepolia",
        "indexer_import_ready": False,
        "finalization_required": "publish emission root and insert emitted root_id",
        "subnet": subnet,
        "root": {
            "root_id": None,
            "subnet_id": emission["subnet_id"],
            "epoch": int(emission["epoch"]),
            "root": emission["root"],
            "total_amount_units": str(emission["total_amount_units"]),
            "budget": str(emission["total_amount_units"]),
            "policy_hash": emission["policy_hash"],
            "attestation_hash": emission["attestation_hash"],
            "nonce": emission["nonce"],
            "status": "finalized",
            "validation_status": "accepted",
        },
        "claim_list": emission["claim_list"],
        "leaves": emission["leaves"],
        "claim_evidence": list(dict(emission["bundle"]).get("claim_evidence") or []),
    }


def build_seed_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_json(args.manifest)
    _assert_base_sepolia_manifest(manifest)
    root_registry = _manifest_contract(manifest, "root_registry")
    owner = str(dict(dict(manifest.get("roles") or {}).get("owner") or {}).get("address") or root_registry)
    _manifest_contract(manifest, "genesis_distributor")
    _manifest_contract(manifest, "emission_distributor")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not str(args.test_old_coldkey or "").strip():
        raise ValueError("test_old_coldkey is required")
    if not args.lane_id:
        args.lane_id = _manifest_default_lane_id(manifest)
    if not args.lane_id:
        raise ValueError("lane_id is required")

    genesis = _genesis_inputs(args, manifest)
    evidence = _load_json(args.emission_evidence)
    emission = _normalized_emission_artifact_inputs(
        evidence,
        expected_wallet=genesis["reward_address"],
        min_accepted=args.min_accepted_count,
        min_committee=args.min_committee_count,
    )
    if str(emission["subnet_id"]) != str(args.lane_id):
        raise ValueError(f"emission evidence subnet_id {emission['subnet_id']!r} does not match lane_id {args.lane_id!r}")

    paths = {
        "genesis_root_artifact": args.out_dir / "base-sota-testnet-genesis-root-artifact.json",
        "emission_root_artifact": args.out_dir / "base-sota-testnet-emission-root-artifact.json",
        "genesis_claim_template": args.out_dir / "base-sota-testnet-genesis-claim-template.json",
        "emission_claim_template": args.out_dir / "base-sota-testnet-emission-claim-template.json",
        "emission_evidence": args.out_dir / "base-sota-testnet-emission-evidence.json",
        "report": args.out_dir / "base-sota-testnet-seed-artifacts.json",
    }
    _write_json(
        paths["genesis_root_artifact"],
        _root_artifact(
            kind="genesis",
            root=genesis["leaf"],
            total_amount=genesis["amount"],
            policy_hash=genesis["policy_hash"],
            attestation_hash=genesis["attestation_hash"],
            nonce=genesis["nonce"],
            metadata={"old_coldkey": genesis["old_coldkey"], "reward_address": genesis["reward_address"]},
        ),
    )
    _write_json(
        paths["emission_root_artifact"],
        _root_artifact(
            kind="emission",
            root=emission["root"],
            total_amount=emission["total_amount_units"],
            policy_hash=emission["policy_hash"],
            attestation_hash=emission["attestation_hash"],
            nonce=emission["nonce"],
            metadata={"subnet_id": emission["subnet_id"], "epoch": emission["epoch"], "evidence_hash": emission["evidence_hash"]},
        ),
    )
    _write_json(paths["genesis_claim_template"], _pending_genesis_claim_artifact(genesis, owner=owner))
    _write_json(paths["emission_claim_template"], _pending_emission_claim_artifact(emission, owner=owner))
    _write_json(paths["emission_evidence"], evidence)

    genesis_publish_result = args.out_dir / "base-sota-testnet-genesis-root-publish-result.json"
    emission_publish_result = args.out_dir / "base-sota-testnet-emission-root-publish-result.json"
    report = {
        "schema": "sota-base-testnet-seed-artifacts/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "base-sepolia",
        "ok": True,
        "status": "ready_to_publish_roots",
        "indexer_import_ready": False,
        "message": "Root artifacts are ready. Broadcast them, then finalize claim artifacts with the emitted root IDs.",
        "manifest": str(args.manifest),
        "contracts": {
            "root_registry": root_registry,
            "genesis_distributor": _manifest_contract(manifest, "genesis_distributor"),
            "emission_distributor": _manifest_contract(manifest, "emission_distributor"),
        },
        "seeded_claims": {
            "test_wallet_address": genesis["reward_address"],
            "test_old_coldkey": genesis["old_coldkey"],
            "lane_id": emission["subnet_id"],
            "epoch": emission["epoch"],
            "genesis_total_units": str(genesis["amount"]),
            "emission_total_units": str(emission["total_amount_units"]),
            "self_validation_min_accepted_count": args.min_accepted_count,
            "self_validation_min_committee_count": args.min_committee_count,
        },
        "artifacts": {key: str(path) for key, path in paths.items()},
        "publish_results": {
            "genesis": str(genesis_publish_result),
            "emission": str(emission_publish_result),
        },
        "commands": {
            "dry_run_publish_genesis": (
                f"python3 scripts/sota_base_publish_root.py --manifest {args.manifest} "
                f"--root-artifact {paths['genesis_root_artifact']} --kind genesis --out {genesis_publish_result}"
            ),
            "dry_run_publish_emission": (
                f"python3 scripts/sota_base_publish_root.py --manifest {args.manifest} "
                f"--root-artifact {paths['emission_root_artifact']} --kind emission --out {emission_publish_result}"
            ),
            "broadcast_publish_genesis": (
                f"python3 scripts/sota_base_publish_root.py --manifest {args.manifest} "
                f"--root-artifact {paths['genesis_root_artifact']} --kind genesis --broadcast --out {genesis_publish_result}"
            ),
            "broadcast_publish_emission": (
                f"python3 scripts/sota_base_publish_root.py --manifest {args.manifest} "
                f"--root-artifact {paths['emission_root_artifact']} --kind emission --broadcast --out {emission_publish_result}"
            ),
            "finalize_claim_artifacts": (
                f"python3 scripts/sota_base_testnet_seed_artifacts.py finalize --build-report {paths['report']} "
                f"--genesis-publish-result {genesis_publish_result} --emission-publish-result {emission_publish_result}"
            ),
        },
        "does_not": ["sign", "broadcast_transactions", "touch_production_bittensor", "touch_base_mainnet"],
    }
    _write_json(paths["report"], report)
    return report


def _finalized_claim_artifact(template: dict[str, Any], *, root_id: str) -> dict[str, Any]:
    artifact = deepcopy(template)
    artifact["generated_at"] = datetime.now(timezone.utc).isoformat()
    artifact["indexer_import_ready"] = True
    artifact.pop("finalization_required", None)
    artifact["root"]["root_id"] = root_id
    return artifact


def finalize_seed_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    report = _load_json(args.build_report)
    out_dir = args.out_dir or args.build_report.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = dict(report.get("artifacts") or {})
    genesis_template = _load_json(Path(artifacts["genesis_claim_template"]))
    emission_template = _load_json(Path(artifacts["emission_claim_template"]))
    genesis_root_id = _published_root_id(args.genesis_publish_result)
    emission_root_id = _published_root_id(args.emission_publish_result)

    paths = {
        "genesis_claim_artifact": out_dir / "base-sota-testnet-genesis-claim-artifact.json",
        "emission_claim_artifact": out_dir / "base-sota-testnet-emission-claim-artifact.json",
        "report": out_dir / "base-sota-testnet-seed-artifacts-finalized.json",
    }
    _write_json(paths["genesis_claim_artifact"], _finalized_claim_artifact(genesis_template, root_id=genesis_root_id))
    _write_json(paths["emission_claim_artifact"], _finalized_claim_artifact(emission_template, root_id=emission_root_id))

    finalized = {
        **report,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_to_import_claim_artifacts",
        "indexer_import_ready": True,
        "message": "Claim artifacts contain emitted on-chain root IDs and are ready for indexer import.",
        "root_ids": {"genesis": genesis_root_id, "emission": emission_root_id},
        "artifacts": {
            **artifacts,
            "genesis_claim_artifact": str(paths["genesis_claim_artifact"]),
            "emission_claim_artifact": str(paths["emission_claim_artifact"]),
        },
        "commands": {
            **dict(report.get("commands") or {}),
            "import_genesis_claim_artifact": (
                "curl -fsS -X POST \"$SOTA_CLAIMS_API_URL/api/v1/base/index/artifact\" "
                f"-H 'content-type: application/json' --data-binary @{paths['genesis_claim_artifact']}"
            ),
            "import_emission_claim_artifact": (
                "curl -fsS -X POST \"$SOTA_CLAIMS_API_URL/api/v1/base/index/artifact\" "
                f"-H 'content-type: application/json' --data-binary @{paths['emission_claim_artifact']}"
            ),
        },
    }
    _write_json(paths["report"], finalized)
    return finalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and finalize Base Sepolia seed claim/root artifacts.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build publish-ready root artifacts from a manifest and real autoresearch evidence")
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--emission-evidence", type=Path, required=True)
    build.add_argument("--test-wallet-address", required=True)
    build.add_argument("--test-old-coldkey", required=True)
    build.add_argument("--lane-id", default="")
    build.add_argument("--tao-credit-sota", default="1")
    build.add_argument("--alpha-synthetic-credit-sota", default="0.5")
    build.add_argument("--min-accepted-count", type=int, default=3)
    build.add_argument("--min-committee-count", type=int, default=3)
    build.add_argument("--genesis-policy-hash", default="")
    build.add_argument("--genesis-attestation-hash", default="")
    build.add_argument("--genesis-nonce", default="")
    build.add_argument("--out-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)

    finalize = sub.add_parser("finalize", help="insert emitted root IDs and write indexer-importable claim artifacts")
    finalize.add_argument("--build-report", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-seed-artifacts.json")
    finalize.add_argument("--genesis-publish-result", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-genesis-root-publish-result.json")
    finalize.add_argument("--emission-publish-result", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-emission-root-publish-result.json")
    finalize.add_argument("--out-dir", type=Path)

    args = parser.parse_args(argv)
    report = build_seed_artifacts(args) if args.command == "build" else finalize_seed_artifacts(args)
    print(json.dumps({"ok": True, "status": report["status"], "report": str((args.out_dir or DEFAULT_ARTIFACTS_DIR) / ("base-sota-testnet-seed-artifacts-finalized.json" if args.command == "finalize" else "base-sota-testnet-seed-artifacts.json"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

REPOS = Path("/home/mekaneeky/repos")
SCRIPT_DIR = Path(__file__).resolve().parent
POOL_SCRIPTS = REPOS / "Pool" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(POOL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(POOL_SCRIPTS))

from base_claim_attestation import (  # type: ignore  # noqa: E402
    BindingMessage,
    BindingRegistry,
    SnapshotAllocation,
    SnapshotContext,
)
from sota_base_testnet_seed_artifacts import (  # noqa: E402
    BASE_MAINNET_CHAIN_ID,
    BASE_SEPOLIA_CHAIN_ID,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_MANIFEST,
    _assert_base_sepolia_manifest,
    _genesis_leaf,
    _hex32,
    _keccak_json,
    _load_json,
    _manifest_chain_id,
    _manifest_contract,
    _normalize_address,
    _positive_int,
    _root_artifact,
    _write_json,
)


DEFAULT_SNAPSHOT_DIR = Path("/mnt/4tb/tao_fork_snapshot")
DEFAULT_OUT_DIR = DEFAULT_ARTIFACTS_DIR / "snapshot-claims"
SOTA_UNITS_PER_RAO = 10**9


@dataclass(frozen=True)
class SnapshotBlock:
    number: int
    hash: str


@dataclass(frozen=True)
class SnapshotClaim:
    coldkey: str
    snapshot_index: int
    claim_id: str
    direct_tao_rao: int
    alpha_credit_rao: int
    alpha_credit_rao_by_netuid: dict[str, int]
    alpha_exposure_units_by_netuid: dict[str, int]
    total_alpha_exposure_units_by_netuid: dict[str, int]
    tao_in_pool_rao_by_netuid: dict[str, int]
    amount_rao: int
    amount_units: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_snapshot_block(snapshot_dir: Path) -> SnapshotBlock:
    block_path = snapshot_dir / "genesis_snapshot_block.json"
    if block_path.exists():
        payload = _load_json(block_path)
        return SnapshotBlock(
            number=int(payload["bittensor_block_number"]),
            hash=_hex32(payload["bittensor_block_hash"], "bittensor_block_hash"),
        )
    manifest = _load_json(snapshot_dir / "artifact_manifest.json")
    block = dict(manifest.get("block") or {})
    return SnapshotBlock(number=int(block["height"]), hash=_hex32(block["hash"], "block.hash"))


def _claim_id(*, block: SnapshotBlock, coldkey: str) -> str:
    digest = _sha256_hex(
        {
            "schema": "sota-genesis-snapshot-claim-id/v1",
            "bittensor_block_number": block.number,
            "bittensor_block_hash": block.hash,
            "coldkey": coldkey,
        }
    )
    return f"sota-genesis:{block.number}:{digest}"


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _included(row: dict[str, str]) -> bool:
    return str(row.get("included", "True")).strip().lower() in {"1", "true", "yes"}


def _read_direct_tao(snapshot_dir: Path, coldkey: str) -> tuple[int, int]:
    path = snapshot_dir / "coldkeys.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing snapshot coldkeys.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if str(row.get("coldkey") or "").strip() != coldkey:
                continue
            if not _included(row):
                reason = str(row.get("exclusion_reason") or "excluded").strip()
                raise ValueError(f"coldkey is excluded from the snapshot: {reason}")
            if row.get("tao_total_rao") not in {None, ""}:
                return index, _int(row, "tao_total_rao")
            return index, _int(row, "tao_free_rao") + _int(row, "tao_reserved_rao")
    raise ValueError(f"coldkey is not present in snapshot coldkeys.csv: {coldkey}")


def _scan_alpha(snapshot_dir: Path, coldkey: str) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    path = snapshot_dir / "alpha_exposures.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing snapshot alpha_exposures.csv: {path}")
    target_by_netuid: dict[str, int] = {}
    total_by_netuid: dict[str, int] = {}
    pool_by_netuid: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not _included(row):
                continue
            netuid = str(int(row["netuid"]))
            amount = _int(row, "total_alpha_units")
            if amount <= 0:
                continue
            total_by_netuid[netuid] = total_by_netuid.get(netuid, 0) + amount
            pool_by_netuid.setdefault(netuid, _int(row, "tao_in_pool_rao"))
            if str(row.get("coldkey") or "").strip() == coldkey:
                target_by_netuid[netuid] = target_by_netuid.get(netuid, 0) + amount
    return target_by_netuid, total_by_netuid, pool_by_netuid


def _snapshot_claim(snapshot_dir: Path, coldkey: str, *, sota_units_per_rao: int) -> SnapshotClaim:
    block = _load_snapshot_block(snapshot_dir)
    snapshot_index, direct_tao_rao = _read_direct_tao(snapshot_dir, coldkey)
    target_alpha, total_alpha, pool_by_netuid = _scan_alpha(snapshot_dir, coldkey)
    alpha_credit_by_netuid: dict[str, int] = {}
    for netuid, target_units in sorted(target_alpha.items(), key=lambda item: int(item[0])):
        denominator = total_alpha.get(netuid, 0)
        pool_rao = pool_by_netuid.get(netuid, 0)
        if target_units <= 0 or denominator <= 0 or pool_rao <= 0:
            continue
        credit = target_units * pool_rao // denominator
        if credit > 0:
            alpha_credit_by_netuid[netuid] = credit
    alpha_credit_rao = sum(alpha_credit_by_netuid.values())
    amount_rao = direct_tao_rao + alpha_credit_rao
    if amount_rao <= 0:
        raise ValueError(f"coldkey has zero SOTA genesis credit: {coldkey}")
    return SnapshotClaim(
        coldkey=coldkey,
        snapshot_index=snapshot_index,
        claim_id=_claim_id(block=block, coldkey=coldkey),
        direct_tao_rao=direct_tao_rao,
        alpha_credit_rao=alpha_credit_rao,
        alpha_credit_rao_by_netuid=alpha_credit_by_netuid,
        alpha_exposure_units_by_netuid=target_alpha,
        total_alpha_exposure_units_by_netuid={key: total_alpha[key] for key in target_alpha if key in total_alpha},
        tao_in_pool_rao_by_netuid={key: pool_by_netuid[key] for key in target_alpha if key in pool_by_netuid},
        amount_rao=amount_rao,
        amount_units=amount_rao * int(sota_units_per_rao),
    )


def _snapshot_context(
    *,
    snapshot_dir: Path,
    manifest: dict[str, Any],
    claims: list[SnapshotClaim],
    allow_local: bool = False,
) -> SnapshotContext:
    block = _load_snapshot_block(snapshot_dir)
    chain_id = _manifest_chain_id(manifest)
    if chain_id == BASE_MAINNET_CHAIN_ID:
        raise ValueError("refusing to build snapshot claims against Base mainnet")
    if chain_id == BASE_SEPOLIA_CHAIN_ID:
        _assert_base_sepolia_manifest(manifest)
    elif chain_id != 31337 or not allow_local:
        raise ValueError(f"manifest chain id must be Base Sepolia 84532 or local 31337, got {chain_id}")
    return SnapshotContext(
        snapshot_id=f"sota-genesis-{block.number}-{block.hash[:12]}",
        bittensor_block_number=block.number,
        bittensor_block_hash=block.hash,
        base_chain_id=chain_id,
        distributor_contract=_manifest_contract(manifest, "genesis_distributor"),
        allocations=tuple(
            SnapshotAllocation(
                index=claim.snapshot_index,
                coldkey=claim.coldkey,
                amount_units=claim.amount_units,
                claim_id=claim.claim_id,
            )
            for claim in claims
        ),
    )


def _load_binding(path: Path, signature: str = "") -> dict[str, Any]:
    payload = _load_json(path)
    message = payload.get("message") if isinstance(payload.get("message"), dict) else payload
    sig = signature or str(payload.get("signature") or "").strip()
    if not sig:
        raise ValueError(f"binding signature is required: {path}")
    return {"message": message, "signature": sig, "source": str(path)}


def _snapshot_claim_by_message(snapshot_dir: Path, message: dict[str, Any], *, sota_units_per_rao: int) -> SnapshotClaim:
    coldkey = str(message.get("coldkey") or "").strip()
    if not coldkey:
        raise ValueError("binding message is missing coldkey")
    claim = _snapshot_claim(snapshot_dir, coldkey, sota_units_per_rao=sota_units_per_rao)
    if str(message.get("claim_id") or "") != claim.claim_id:
        raise ValueError(f"binding claim_id does not match snapshot claim for {coldkey}")
    if int(message.get("allocation_amount") or 0) != claim.amount_units:
        raise ValueError(f"binding allocation_amount does not match snapshot claim for {coldkey}")
    return claim


def _policy_hash(*, manifest: dict[str, Any], snapshot_dir: Path, claims: list[SnapshotClaim], sota_units_per_rao: int) -> str:
    block = _load_snapshot_block(snapshot_dir)
    return _keccak_json(
        {
            "schema": "sota-base-genesis-policy/v1",
            "conversion_rate": "1 TAO = 1 SOTA",
            "bittensor_units_per_tao": 10**9,
            "sota_token_decimals": 18,
            "sota_units_per_rao": int(sota_units_per_rao),
            "alpha_formula": "floor(user_alpha_exposure * tao_in_pool_rao / total_eligible_alpha_exposure)",
            "snapshot_block_number": block.number,
            "snapshot_block_hash": block.hash,
            "root_registry": _manifest_contract(manifest, "root_registry"),
            "claim_count": len(claims),
        }
    )


def _claim_artifact(
    *,
    manifest: dict[str, Any],
    snapshot_dir: Path,
    snapshot_context: SnapshotContext,
    binding_manifest: dict[str, Any],
    claims_by_coldkey: dict[str, SnapshotClaim],
    root_id: str | None,
    policy_hash: str,
    attestation_hash: str,
    nonce: str,
    sota_units_per_rao: int,
) -> dict[str, Any]:
    del snapshot_dir
    leaves = list(dict(binding_manifest["merkle"]).get("leaves") or [])
    allocations: list[dict[str, Any]] = []
    leaf_records: list[dict[str, Any]] = []
    for leaf in leaves:
        coldkey = str(leaf["coldkey"])
        claim = claims_by_coldkey[coldkey]
        amount = int(leaf["amount_units"])
        allocation_hash = _hex32(leaf["allocation_hash"], "allocation_hash")
        reward_address = _normalize_address(leaf["reward_address"], "reward_address")
        recomputed_leaf = _genesis_leaf(reward_address, amount, allocation_hash)
        if recomputed_leaf != _hex32(leaf["leaf"], "leaf"):
            raise ValueError(f"binding manifest leaf mismatch for {coldkey}")
        proof = [_hex32(item, "proof") for item in list(leaf.get("proof") or [])]
        allocations.append(
            {
                "kind": "genesis",
                "version": "sota-genesis-claim-v1",
                "index": int(leaf["index"]),
                "snapshot_index": int(leaf["snapshot_index"]),
                "claim_id": str(leaf["claim_id"]),
                "account": reward_address,
                "reward_address": reward_address,
                "amount": str(amount),
                "amount_units": str(amount),
                "allocation_hash": allocation_hash,
                "binding_hash": allocation_hash,
                "old_coldkey": coldkey,
                "tao_credit": str(claim.direct_tao_rao * int(sota_units_per_rao)),
                "alpha_synthetic_credit": str(claim.alpha_credit_rao * int(sota_units_per_rao)),
                "tao_credit_rao": str(claim.direct_tao_rao),
                "alpha_synthetic_credit_rao": str(claim.alpha_credit_rao),
                "alpha_credit_rao_by_netuid": {
                    netuid: str(value) for netuid, value in sorted(claim.alpha_credit_rao_by_netuid.items())
                },
                "leaf": recomputed_leaf,
                "proof": proof,
            }
        )
        leaf_records.append({"index": int(leaf["index"]), "leaf": recomputed_leaf, "proof": proof})

    root = _hex32(dict(binding_manifest["merkle"])["root"], "root")
    total_amount = sum(int(row["amount_units"]) for row in allocations)
    owner = str(dict(dict(manifest.get("roles") or {}).get("owner") or {}).get("address") or _manifest_contract(manifest, "root_registry"))
    return {
        "schema": "sota-base-claim-artifact/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": "base-sepolia" if int(snapshot_context.base_chain_id) == BASE_SEPOLIA_CHAIN_ID else "local",
        "indexer_import_ready": bool(root_id),
        **({} if root_id else {"finalization_required": "publish genesis root and insert emitted root_id"}),
        "subnet": {
            "id": "genesis",
            "title": "SOTA genesis fork claim",
            "owner": owner,
            "budget": str(total_amount),
            "metadata_uri": f"sota://genesis/{snapshot_context.snapshot_id}",
            "token": "SOTA",
        },
        "root": {
            "root_id": root_id,
            "subnet_id": "genesis",
            "epoch": 0,
            "root": root,
            "total_amount_units": str(total_amount),
            "budget": str(total_amount),
            "policy_hash": policy_hash,
            "attestation_hash": attestation_hash,
            "nonce": nonce,
            "status": "finalized",
            "validation_status": "accepted",
        },
        "allocations": allocations,
        "leaves": leaf_records,
        "snapshot": snapshot_context.to_dict(include_allocations=False),
    }


def _report_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "context": out_dir / "sota-snapshot-genesis-context.json",
        "message": out_dir / "sota-snapshot-binding-message.json",
        "binding_manifest": out_dir / "sota-snapshot-binding-manifest.json",
        "root_artifact": out_dir / "sota-snapshot-genesis-root-artifact.json",
        "claim_template": out_dir / "sota-snapshot-genesis-claim-template.json",
        "claim_artifact": out_dir / "sota-snapshot-genesis-claim-artifact.json",
        "report": out_dir / "sota-snapshot-genesis-report.json",
    }


def build_message(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = args.snapshot_dir
    manifest = _load_json(args.manifest)
    claim = _snapshot_claim(snapshot_dir, args.coldkey, sota_units_per_rao=args.sota_units_per_rao)
    context = _snapshot_context(snapshot_dir=snapshot_dir, manifest=manifest, claims=[claim], allow_local=args.allow_local)
    message = BindingMessage.create(snapshot=context, allocation=context.allocations[0], reward_address=args.reward_address)
    paths = _report_paths(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "sota-snapshot-binding-message/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "message": message.to_payload_dict(),
        "signing_payload": message.signing_payload(),
        "signing_payload_sha256": message.signing_payload_sha256(),
        "snapshot_claim": _claim_summary(claim),
        "instruction": "Sign signing_payload bytes with the Bittensor coldkey, then pass the 64-byte hex signature to the build command.",
    }
    _write_json(paths["context"], context.to_dict(include_allocations=True))
    _write_json(paths["message"], payload)
    return {"ok": True, "status": "message_ready", "paths": {k: str(v) for k, v in paths.items() if k in {"context", "message"}}, **payload}


def _claim_summary(claim: SnapshotClaim) -> dict[str, Any]:
    return {
        "coldkey": claim.coldkey,
        "snapshot_index": claim.snapshot_index,
        "claim_id": claim.claim_id,
        "direct_tao_rao": str(claim.direct_tao_rao),
        "alpha_credit_rao": str(claim.alpha_credit_rao),
        "alpha_credit_rao_by_netuid": {
            netuid: str(value) for netuid, value in sorted(claim.alpha_credit_rao_by_netuid.items())
        },
        "amount_rao": str(claim.amount_rao),
        "amount_units": str(claim.amount_units),
    }


def build_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_dir = args.snapshot_dir
    manifest = _load_json(args.manifest)
    bindings = [_load_binding(path, args.signature if len(args.binding) == 1 else "") for path in args.binding]
    claims = [
        _snapshot_claim_by_message(snapshot_dir, dict(binding["message"]), sota_units_per_rao=args.sota_units_per_rao)
        for binding in bindings
    ]
    context = _snapshot_context(snapshot_dir=snapshot_dir, manifest=manifest, claims=claims, allow_local=args.allow_local)
    registry = BindingRegistry(snapshot=context)
    binding_artifacts = [
        registry.submit(message=binding["message"], signature=binding["signature"])
        for binding in bindings
    ]
    binding_manifest = registry.build_manifest()
    claims_by_coldkey = {claim.coldkey: claim for claim in claims}
    root = _hex32(dict(binding_manifest["merkle"])["root"], "binding_manifest.root")
    total_amount = _positive_int(dict(binding_manifest["merkle"])["total_units"], "binding_manifest.total_units")
    policy_hash = args.policy_hash or _policy_hash(
        manifest=manifest,
        snapshot_dir=snapshot_dir,
        claims=claims,
        sota_units_per_rao=args.sota_units_per_rao,
    )
    attestation_hash = args.attestation_hash or _keccak_json(
        {
            "schema": "sota-base-genesis-attestation/v1",
            "snapshot": context.to_dict(include_allocations=False),
            "binding_manifest_hash": _sha256_hex(binding_manifest),
            "binding_count": len(binding_artifacts),
        }
    )
    nonce = args.nonce or _keccak_json(
        {
            "schema": "sota-base-root-nonce/v1",
            "kind": "genesis",
            "chain_id": context.base_chain_id,
            "snapshot_id": context.snapshot_id,
            "root": root,
            "claim_list_hash": binding_manifest["claim_list_hash"],
        }
    )
    claim_artifact = _claim_artifact(
        manifest=manifest,
        snapshot_dir=snapshot_dir,
        snapshot_context=context,
        binding_manifest=binding_manifest,
        claims_by_coldkey=claims_by_coldkey,
        root_id=args.root_id or None,
        policy_hash=_hex32(policy_hash, "policy_hash"),
        attestation_hash=_hex32(attestation_hash, "attestation_hash"),
        nonce=_hex32(nonce, "nonce"),
        sota_units_per_rao=args.sota_units_per_rao,
    )
    root_artifact = _root_artifact(
        kind="genesis",
        root=root,
        total_amount=total_amount,
        policy_hash=_hex32(policy_hash, "policy_hash"),
        attestation_hash=_hex32(attestation_hash, "attestation_hash"),
        nonce=_hex32(nonce, "nonce"),
        metadata={
            "snapshot": context.to_dict(include_allocations=False),
            "claim_count": len(claims),
            "claim_list_hash": binding_manifest["claim_list_hash"],
        },
    )
    paths = _report_paths(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(paths["context"], context.to_dict(include_allocations=True))
    _write_json(paths["binding_manifest"], binding_manifest)
    _write_json(paths["root_artifact"], root_artifact)
    _write_json(paths["claim_artifact"] if args.root_id else paths["claim_template"], claim_artifact)
    publish_result = args.out_dir / "sota-snapshot-genesis-root-publish-result.json"
    report = {
        "schema": "sota-snapshot-genesis-claim-bridge/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "status": "ready_to_import_claim_artifact" if args.root_id else "ready_to_publish_root",
        "indexer_import_ready": bool(args.root_id),
        "snapshot": context.to_dict(include_allocations=False),
        "claim_count": len(claims),
        "claims": [_claim_summary(claim) for claim in claims],
        "root": {
            "root": root,
            "root_id": args.root_id,
            "total_amount_units": str(total_amount),
            "policy_hash": _hex32(policy_hash, "policy_hash"),
            "attestation_hash": _hex32(attestation_hash, "attestation_hash"),
            "nonce": _hex32(nonce, "nonce"),
        },
        "artifacts": {
            "context": str(paths["context"]),
            "binding_manifest": str(paths["binding_manifest"]),
            "root_artifact": str(paths["root_artifact"]),
            "claim_artifact": str(paths["claim_artifact"] if args.root_id else paths["claim_template"]),
            "report": str(paths["report"]),
        },
        "commands": {
            "publish_genesis_root": (
                f"python3 scripts/sota_base_publish_root.py --manifest {args.manifest} "
                f"--root-artifact {paths['root_artifact']} --kind genesis --broadcast --out {publish_result}"
            ),
            "finalize_claim_artifact": (
                f"python3 scripts/sota_snapshot_claim_bridge.py finalize --claim-template {paths['claim_template']} "
                f"--publish-result {publish_result} --out {paths['claim_artifact']}"
            ),
            "import_claim_artifact": (
                "curl -fsS -X POST \"$SOTA_CLAIMS_API_URL/api/v1/base/index/artifact\" "
                f"-H 'content-type: application/json' --data-binary @{paths['claim_artifact']}"
            ),
        },
        "does_not": ["touch_production_bittensor", "touch_base_mainnet", "custody_user_keys", "bridge_alpha_tokens"],
    }
    _write_json(paths["report"], report)
    return report


def finalize_artifact(args: argparse.Namespace) -> dict[str, Any]:
    template = _load_json(args.claim_template)
    publish = _load_json(args.publish_result)
    root_id = publish.get("root_id")
    if not root_id and isinstance(publish.get("root_published_event"), dict):
        root_id = dict(publish["root_published_event"]).get("root_id")
    if str(publish.get("status") or "") != "broadcasted":
        raise ValueError("publish result must have status=broadcasted")
    artifact = dict(template)
    artifact["generated_at"] = datetime.now(timezone.utc).isoformat()
    artifact["indexer_import_ready"] = True
    artifact.pop("finalization_required", None)
    artifact["root"] = dict(artifact["root"])
    artifact["root"]["root_id"] = _hex32(root_id, "root_id")
    _write_json(args.out, artifact)
    return {"ok": True, "status": "ready_to_import_claim_artifact", "claim_artifact": str(args.out), "root_id": artifact["root"]["root_id"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bridge real Bittensor snapshot coldkey allocations into Base SOTA genesis claim artifacts.")
    sub = parser.add_subparsers(dest="command", required=True)

    message = sub.add_parser("message", help="build the exact coldkey binding message a claimant must sign")
    message.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    message.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    message.add_argument("--coldkey", required=True)
    message.add_argument("--reward-address", required=True)
    message.add_argument("--sota-units-per-rao", type=int, default=SOTA_UNITS_PER_RAO)
    message.add_argument("--allow-local", action="store_true")
    message.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    build = sub.add_parser("build", help="verify signed coldkey bindings and build publish/import artifacts")
    build.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--binding", type=Path, action="append", required=True, help="binding JSON containing message and signature")
    build.add_argument("--signature", default="", help="signature override for a single --binding message file")
    build.add_argument("--root-id", default="")
    build.add_argument("--policy-hash", default="")
    build.add_argument("--attestation-hash", default="")
    build.add_argument("--nonce", default="")
    build.add_argument("--sota-units-per-rao", type=int, default=SOTA_UNITS_PER_RAO)
    build.add_argument("--allow-local", action="store_true")
    build.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    finalize = sub.add_parser("finalize", help="insert the on-chain root id into a claim artifact template")
    finalize.add_argument("--claim-template", type=Path, required=True)
    finalize.add_argument("--publish-result", type=Path, required=True)
    finalize.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "message":
        result = build_message(args)
    elif args.command == "build":
        result = build_artifacts(args)
    else:
        result = finalize_artifact(args)
    print(json.dumps({"ok": True, "status": result["status"], "result": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

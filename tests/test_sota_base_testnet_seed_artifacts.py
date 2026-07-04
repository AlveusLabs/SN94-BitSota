from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_seed_artifacts.py"
COMMUNITY_REPO = Path("/home/mekaneeky/repos/94-agent-community")


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_seed_artifacts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(tmp_path: Path) -> Path:
    payload = {
        "manifest_schema_version": "sota-base-sepolia-deployment-manifest/v1",
        "environment": "base-sepolia",
        "chain": {"chain_id": 84532, "chain_name": "base-sepolia"},
        "roles": {"owner": {"address": "0x00000000000000000000000000000000000000aa"}},
        "browser_safe": {
            "contract_addresses": {
                "root_registry": "0x0000000000000000000000000000000000000003",
                "genesis_distributor": "0x0000000000000000000000000000000000000005",
                "emission_distributor": "0x0000000000000000000000000000000000000006",
            }
        },
        "contracts": {
            "root_registry": {"address": "0x0000000000000000000000000000000000000003"},
            "genesis_distributor": {"address": "0x0000000000000000000000000000000000000005"},
            "emission_distributor": {"address": "0x0000000000000000000000000000000000000006"},
        },
        "services": {
            "claims_ui": {
                "browser_safe_env": {
                    "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID": "base:sota-test"
                }
            }
        },
    }
    return _write_json(tmp_path / "base-sepolia-deployment-manifest.json", payload)


def _emission_evidence(module, tmp_path: Path, *, wallet: str, accepted_count: int = 3) -> Path:
    lane_id = "base:sota-test"
    epoch = 1
    policy_hash = "0x" + "22" * 32
    offchain_lane_id = module._keccak_json("sota-foundation/base-sota-test")
    claims = []
    leaf_hashes = []
    for index, (account, amount) in enumerate(
        [
            (wallet.lower(), 2 * 10**18),
            ("0x00000000000000000000000000000000000000bb", 10**18),
        ]
    ):
        reward_hash = module._keccak_json({"submission_id": f"submission-{index}", "account": account, "amount": amount})
        claim = {
            "version": "sota-emission-claim-v1",
            "subnet_id": lane_id,
            "offchain_lane_id": offchain_lane_id,
            "offchain_subnet_id": offchain_lane_id,
            "epoch": epoch,
            "index": index,
            "evm_miner_address": account,
            "reward_address": account,
            "amount_units": amount,
            "submission_id": f"submission-{index}",
            "task_id": f"task-{index}",
            "task_slug": "seeded-linear-frontier",
            "miner_hotkey": f"hotkey-{index}",
            "metric_name": "heldout_ppl",
            "metric_value": 0.82 - index / 100,
            "policy_hash": policy_hash,
            "reward_hash": reward_hash,
        }
        claims.append(claim)
        leaf_hashes.append(
            module._emission_leaf(
                epoch=epoch,
                offchain_lane_id=offchain_lane_id,
                account=account,
                amount=amount,
                reward_hash=reward_hash,
            )
        )
    layers = module._merkle_layers(leaf_hashes)
    root = layers[-1][0]
    evidence = {
        "root": {
            "id": "root-row-1",
            "subnet_id": lane_id,
            "epoch": epoch,
            "root": root,
            "leaf_count": len(claims),
            "total_amount_units": sum(int(claim["amount_units"]) for claim in claims),
            "policy_hash": policy_hash,
            "claim_list_hash": "0x" + "33" * 32,
            "evidence_hash": "44" * 32,
            "ready_for_attestation": True,
            "created_at": "2026-07-04T00:00:00+00:00",
        },
        "bundle": {
            "version": "sota-emission-root-v1",
            "epoch": epoch,
            "subnet": {
                "id": lane_id,
                "title": "Seeded SOTA test lane",
                "policy_hash": policy_hash,
                "offchain_lane_id": offchain_lane_id,
                "offchain_subnet_id": offchain_lane_id,
            },
            "merkle": {
                "root": root,
                "leaf_count": len(claims),
                "total_amount_units": sum(int(claim["amount_units"]) for claim in claims),
                "leaf_format": "SOTA_EMISSION_CLAIM",
                "sort_pairs": True,
                "odd_strategy": "promote",
                "hash": "keccak256",
            },
            "claim_list": claims,
            "claim_evidence": [
                {
                    "index": claim["index"],
                    "leaf": leaf,
                    "evidence": {
                        "self_validation_consensus": {
                            "status": "accepted",
                            "accepted_count": accepted_count,
                            "committee_count": 3,
                            "committee_size": 3,
                        }
                    },
                }
                for claim, leaf in zip(claims, leaf_hashes, strict=True)
            ],
        },
    }
    return _write_json(tmp_path / "emission-evidence.json", evidence)


def _build_args(tmp_path: Path, module, *, accepted_count: int = 3) -> argparse.Namespace:
    wallet = "0x00000000000000000000000000000000000000aa"
    return argparse.Namespace(
        manifest=_manifest(tmp_path),
        emission_evidence=_emission_evidence(module, tmp_path, wallet=wallet, accepted_count=accepted_count),
        test_wallet_address=wallet,
        test_old_coldkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        lane_id="base:sota-test",
        tao_credit_sota="1",
        alpha_synthetic_credit_sota="0.5",
        min_accepted_count=3,
        min_committee_count=3,
        genesis_policy_hash="",
        genesis_attestation_hash="",
        genesis_nonce="",
        out_dir=tmp_path / "out",
    )


def test_seed_artifacts_build_publish_artifacts_and_pending_templates(tmp_path: Path) -> None:
    module = _load_module()
    args = _build_args(tmp_path, module)

    report = module.build_seed_artifacts(args)

    assert report["schema"] == "sota-base-testnet-seed-artifacts/v1"
    assert report["status"] == "ready_to_publish_roots"
    assert report["indexer_import_ready"] is False
    assert "broadcast_publish_genesis" in report["commands"]
    genesis_root = json.loads(Path(report["artifacts"]["genesis_root_artifact"]).read_text(encoding="utf-8"))
    emission_root = json.loads(Path(report["artifacts"]["emission_root_artifact"]).read_text(encoding="utf-8"))
    emission_template = json.loads(Path(report["artifacts"]["emission_claim_template"]).read_text(encoding="utf-8"))
    assert genesis_root["root"]["total_amount_units"] == str(15 * 10**17)
    assert emission_root["root"]["total_amount_units"] == str(3 * 10**18)
    assert emission_template["indexer_import_ready"] is False
    assert emission_template["leaves"][0]["proof"]


def test_seed_artifacts_reject_non_self_validated_emission(tmp_path: Path) -> None:
    module = _load_module()
    args = _build_args(tmp_path, module, accepted_count=1)

    with pytest.raises(ValueError, match="self-validation"):
        module.build_seed_artifacts(args)


def test_seed_artifacts_finalize_writes_indexer_importable_claims(tmp_path: Path) -> None:
    module = _load_module()
    args = _build_args(tmp_path, module)
    report = module.build_seed_artifacts(args)
    genesis_result = _write_json(
        tmp_path / "out" / "base-sota-testnet-genesis-root-publish-result.json",
        {"status": "broadcasted", "root_id": "0x" + "11" * 32},
    )
    emission_result = _write_json(
        tmp_path / "out" / "base-sota-testnet-emission-root-publish-result.json",
        {"status": "broadcasted", "root_published_event": {"root_id": "0x" + "12" * 32}},
    )

    finalized = module.finalize_seed_artifacts(
        argparse.Namespace(
            build_report=Path(report["artifacts"]["report"]) if "report" in report["artifacts"] else tmp_path / "out" / "base-sota-testnet-seed-artifacts.json",
            genesis_publish_result=genesis_result,
            emission_publish_result=emission_result,
            out_dir=tmp_path / "out",
        )
    )

    genesis_claim = json.loads(Path(finalized["artifacts"]["genesis_claim_artifact"]).read_text(encoding="utf-8"))
    emission_claim = json.loads(Path(finalized["artifacts"]["emission_claim_artifact"]).read_text(encoding="utf-8"))
    assert genesis_claim["indexer_import_ready"] is True
    assert genesis_claim["root"]["root_id"] == "0x" + "11" * 32
    assert emission_claim["root"]["root_id"] == "0x" + "12" * 32

    sys.path.insert(0, str(COMMUNITY_REPO))
    from experiments.base_protocol_design.sota_base_indexer.indexer import SotaBaseIndexer
    from experiments.base_protocol_design.sota_base_indexer.store import SotaBaseStore

    store = SotaBaseStore()
    try:
        indexer = SotaBaseIndexer(store)
        assert indexer.ingest_claim_artifact(genesis_claim) == 3
        assert indexer.ingest_claim_artifact(emission_claim) == 4
        wallet = args.test_wallet_address.lower()
        genesis_eligibility = store.eligibility(wallet, subnet_id="genesis")
        emission_proof = store.proof(wallet, subnet_id="base:sota-test")
        assert genesis_eligibility["eligible"] is True
        assert emission_proof["root_id"] == "0x" + "12" * 32
        assert emission_proof["proof"]
        assert emission_proof["claim_args"]["kind"] == "emission"
    finally:
        store.close()

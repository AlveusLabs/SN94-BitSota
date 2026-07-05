from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from substrateinterface import Keypair


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_snapshot_claim_bridge.py"
COMMUNITY_REPO = Path("/home/mekaneeky/repos/94-agent-community")


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_snapshot_claim_bridge", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest(path: Path) -> Path:
    return _write_json(
        path,
        {
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
        },
    )


def _snapshot_dir(path: Path, alice: str, bob: str) -> Path:
    path.mkdir(parents=True)
    _write_json(
        path / "genesis_snapshot_block.json",
        {
            "schema": "sota-genesis-snapshot-block-lock/v1",
            "bittensor_block_number": 123,
            "bittensor_block_hash": "0x" + "ab" * 32,
        },
    )
    (path / "coldkeys.csv").write_text(
        "\n".join(
            [
                "coldkey,included,exclusion_reason,tao_total_rao,direct_alpha_units,lp_alpha_units,total_alpha_units,lp_tao_rao",
                f"{alice},True,,100,25,0,25,0",
                f"{bob},True,,0,75,0,75,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "alpha_exposures.csv").write_text(
        "\n".join(
            [
                "coldkey,netuid,included,exclusion_reason,direct_alpha_units,lp_alpha_units,total_alpha_units,lp_tao_rao,tao_in_pool_rao,alpha_in_pool_units,alpha_out_pool_units",
                f"{alice},7,True,,25,0,25,0,1000,100,100",
                f"{bob},7,True,,75,0,75,0,1000,100,100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_snapshot_claim_bridge_includes_alpha_credit_and_indexes_artifact(tmp_path: Path) -> None:
    module = _load_module()
    alice = Keypair.create_from_uri("//Alice")
    bob = Keypair.create_from_uri("//Bob")
    snapshot_dir = _snapshot_dir(tmp_path / "snapshot", alice.ss58_address, bob.ss58_address)
    manifest = _manifest(tmp_path / "manifest.json")
    reward_address = "0x00000000000000000000000000000000000000a1"
    out_dir = tmp_path / "out"

    message_result = module.build_message(
        argparse.Namespace(
            snapshot_dir=snapshot_dir,
            manifest=manifest,
            coldkey=alice.ss58_address,
            reward_address=reward_address,
            sota_units_per_rao=10**9,
            allow_local=False,
            out_dir=out_dir,
        )
    )
    message = message_result["message"]
    signature = "0x" + alice.sign(message_result["signing_payload"].encode("utf-8")).hex()
    binding_path = _write_json(tmp_path / "binding.json", {"message": message, "signature": signature})

    report = module.build_artifacts(
        argparse.Namespace(
            snapshot_dir=snapshot_dir,
            manifest=manifest,
            binding=[binding_path],
            signature="",
            root_id="0x" + "11" * 32,
            policy_hash="",
            attestation_hash="",
            nonce="",
            sota_units_per_rao=10**9,
            allow_local=False,
            out_dir=out_dir,
        )
    )

    assert report["status"] == "ready_to_import_claim_artifact"
    claim = report["claims"][0]
    assert claim["direct_tao_rao"] == "100"
    assert claim["alpha_credit_rao"] == "250"
    assert claim["amount_rao"] == "350"
    assert claim["amount_units"] == str(350 * 10**9)

    artifact = json.loads(Path(report["artifacts"]["claim_artifact"]).read_text(encoding="utf-8"))
    allocation = artifact["allocations"][0]
    assert allocation["tao_credit_rao"] == "100"
    assert allocation["alpha_synthetic_credit_rao"] == "250"
    assert allocation["alpha_credit_rao_by_netuid"] == {"7": "250"}
    assert allocation["amount_units"] == str(350 * 10**9)

    sys.path.insert(0, str(COMMUNITY_REPO))
    from experiments.base_protocol_design.sota_base_indexer.indexer import SotaBaseIndexer
    from experiments.base_protocol_design.sota_base_indexer.store import SotaBaseStore

    store = SotaBaseStore()
    try:
        indexer = SotaBaseIndexer(store)
        assert indexer.ingest_claim_artifact(artifact) == 3
        eligibility = store.eligibility(reward_address, subnet_id="genesis")
        proof = store.proof(reward_address, subnet_id="genesis")
        assert eligibility["credits"]["tao"]["raw"] == str(100 * 10**9)
        assert eligibility["credits"]["alpha_synthetic"]["raw"] == str(250 * 10**9)
        assert eligibility["credits"]["total_sota"]["raw"] == str(350 * 10**9)
        assert proof["claim_args"]["allocation_hash"] == allocation["allocation_hash"]
    finally:
        store.close()

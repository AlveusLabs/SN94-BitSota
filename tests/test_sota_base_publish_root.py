from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
from eth_utils import keccak


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_publish_root.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_publish_root", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path, *, chain_id: int = 84532) -> Path:
    payload = {
        "manifest_schema_version": "sota-base-sepolia-deployment-manifest/v1",
        "environment": "base-sepolia" if chain_id == 84532 else "base",
        "chain": {
            "chain_id": chain_id,
            "chain_name": "base-sepolia" if chain_id == 84532 else "base",
            "public_browser_rpc_url": "https://sepolia.base.org",
        },
        "browser_safe": {
            "contract_addresses": {
                "root_registry": "0x0000000000000000000000000000000000000003",
            }
        },
        "contracts": {
            "root_registry": {
                "address": "0x0000000000000000000000000000000000000003",
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _root_artifact(tmp_path: Path) -> Path:
    payload = {
        "root": {
            "root": "0x" + "11" * 32,
            "total_amount_units": "2000000000000000000",
            "policy_hash": "0x" + "22" * 32,
            "attestation_hash": "0x" + "33" * 32,
            "nonce": "0x" + "44" * 32,
        }
    }
    path = tmp_path / "root-artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_publish_root_builds_dry_run_calldata(tmp_path: Path) -> None:
    module = _load_module()
    request = module.build_publish_request(
        type(
            "Args",
            (),
            {
                "manifest": _manifest(tmp_path),
                "root_artifact": _root_artifact(tmp_path),
                "kind": "emission",
                "root_registry": "",
                "merkle_root": "",
                "budget_cap": "",
                "policy_hash": "",
                "attestation_hash": "",
                "nonce": "",
                "allow_local": False,
            },
        )()
    )

    assert request["schema"] == "sota-base-root-publish-request/v1"
    assert request["chain_id"] == 84532
    assert request["kind_id"] == 2
    assert request["to"] == "0x0000000000000000000000000000000000000003"
    assert request["merkle_root"] == "0x" + "11" * 32
    assert request["budget_cap"] == "2000000000000000000"
    assert request["transaction"]["data"].startswith("0x")
    assert len(request["transaction"]["data"]) == 2 + 8 + 64 * 6
    assert "touch_base_mainnet" in request["does_not"]


def test_publish_root_main_writes_dry_run_output(tmp_path: Path) -> None:
    module = _load_module()
    out = tmp_path / "publish-request.json"

    exit_code = module.main(
        [
            "--manifest",
            str(_manifest(tmp_path)),
            "--root-artifact",
            str(_root_artifact(tmp_path)),
            "--kind",
            "genesis",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run"
    assert "sign" in payload["does_not"]
    assert "broadcast_transactions" in payload["does_not"]


def test_publish_root_rejects_base_mainnet_manifest(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="mainnet"):
        module.build_publish_request(
            type(
                "Args",
                (),
                {
                    "manifest": _manifest(tmp_path, chain_id=8453),
                    "root_artifact": _root_artifact(tmp_path),
                    "kind": "genesis",
                    "root_registry": "",
                    "merkle_root": "",
                    "budget_cap": "",
                    "policy_hash": "",
                    "attestation_hash": "",
                    "nonce": "",
                    "allow_local": False,
                },
            )()
        )


def test_publish_root_requires_attestation_hash(tmp_path: Path) -> None:
    module = _load_module()
    artifact = json.loads(_root_artifact(tmp_path).read_text(encoding="utf-8"))
    del artifact["root"]["attestation_hash"]
    artifact_path = tmp_path / "root-artifact-without-attestation.json"
    artifact_path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="attestation_hash"):
        module.build_publish_request(
            type(
                "Args",
                (),
                {
                    "manifest": _manifest(tmp_path),
                    "root_artifact": artifact_path,
                    "kind": "genesis",
                    "root_registry": "",
                    "merkle_root": "",
                    "budget_cap": "",
                    "policy_hash": "",
                    "attestation_hash": "",
                    "nonce": "",
                    "allow_local": False,
                },
            )()
        )


def test_publish_root_decodes_root_published_event() -> None:
    module = _load_module()
    root_registry = "0x0000000000000000000000000000000000000003"
    root_id = "0x" + "55" * 32
    merkle_root = "0x" + "66" * 32
    receipt = type(
        "Receipt",
        (),
        {
            "logs": [
                {
                    "address": root_registry,
                    "topics": [
                        "0x"
                        + keccak(
                            text="RootPublished(bytes32,uint8,bytes32,uint256,bytes32,bytes32,bytes32,address)"
                        ).hex(),
                        root_id,
                        "0x" + "00" * 31 + "02",
                        merkle_root,
                    ],
                    "logIndex": 4,
                }
            ]
        },
    )()

    event = module._root_published_event(receipt, root_registry=root_registry)

    assert event == {
        "root_id": root_id,
        "kind_id": 2,
        "merkle_root": merkle_root,
        "log_index": 4,
    }

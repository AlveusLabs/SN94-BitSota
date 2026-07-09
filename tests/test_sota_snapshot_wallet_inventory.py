from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_snapshot_wallet_inventory.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_snapshot_wallet_inventory", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_snapshot(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "genesis_snapshot_block.json").write_text(
        '{"bittensor_block_number": 99, "bittensor_block_hash": "0xabc"}\n',
        encoding="utf-8",
    )
    (path / "coldkeys.csv").write_text(
        "coldkey,included,tao_free_rao,tao_reserved_rao,tao_total_rao\n"
        "5Alice,True,0,0,100\n"
        "5Hotkey,True,0,0,0\n",
        encoding="utf-8",
    )
    (path / "alpha_exposures.csv").write_text(
        "coldkey,netuid,included,total_alpha_units,tao_in_pool_rao\n"
        "5Alice,7,True,50,200\n"
        "5Bob,7,True,50,200\n",
        encoding="utf-8",
    )
    (path / "stake_edges.csv").write_text(
        "coldkey,hotkey,netuid,alpha_units,included,source\n"
        "5Alice,5Hotkey,7,50,True,AlphaV2\n",
        encoding="utf-8",
    )


def _write_coldkeypub(wallet_root: Path, wallet_name: str, address: str) -> None:
    wallet_dir = wallet_root / wallet_name
    wallet_dir.mkdir(parents=True, exist_ok=True)
    (wallet_dir / "coldkeypub.txt").write_text(
        json.dumps({"ss58Address": address}) + "\n",
        encoding="utf-8",
    )


def test_inventory_finds_claimable_wallets(tmp_path: Path) -> None:
    module = _load_module()
    snapshot = tmp_path / "snapshot"
    wallet_root = tmp_path / "wallets"
    _write_snapshot(snapshot)
    _write_coldkeypub(wallet_root, "alice", "5Alice")
    _write_coldkeypub(wallet_root, "hotkey_wallet", "5Hotkey")

    report = module.scan_wallets(snapshot, [wallet_root])

    assert report["wallet_count"] == 2
    assert report["claimable_wallet_count"] == 1
    assert report["claimable"] is True
    alice = next(wallet for wallet in report["wallets"] if wallet["wallet_name"] == "alice")
    hotkey = next(wallet for wallet in report["wallets"] if wallet["wallet_name"] == "hotkey_wallet")
    assert alice["classification"] == "claimable_coldkey"
    assert alice["claimable"] is True
    assert alice["coldkey"]["amount_rao"] == 200
    assert hotkey["classification"] == "hotkey_with_staked_alpha"
    assert hotkey["claimable"] is False
    assert hotkey["hotkey_stake_edge_count"] == 1


def test_inventory_reports_no_claimable_wallets(tmp_path: Path) -> None:
    module = _load_module()
    snapshot = tmp_path / "snapshot"
    wallet_root = tmp_path / "wallets"
    _write_snapshot(snapshot)
    _write_coldkeypub(wallet_root, "hotkey_wallet", "5Hotkey")

    report = module.scan_wallets(snapshot, [wallet_root])

    assert report["claimable"] is False
    assert report["claimable_wallet_count"] == 0
    assert report["next_action"] == "No local public coldkey matched a claimable snapshot allocation."

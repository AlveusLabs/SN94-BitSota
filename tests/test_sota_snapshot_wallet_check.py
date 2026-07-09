from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_snapshot_wallet_check.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_snapshot_wallet_check", SCRIPT)
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
        "5Alice,True,40,60,100\n"
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


def test_snapshot_wallet_check_reports_claimable_coldkey(tmp_path: Path) -> None:
    module = _load_module()
    _write_snapshot(tmp_path)

    report = module.check_snapshot_wallet(tmp_path, "5Alice")

    assert report["classification"] == "claimable_coldkey"
    assert report["claimable"] is True
    assert report["coldkey"]["direct_tao_rao"] == 100
    assert report["coldkey"]["alpha_credit_rao"] == 100
    assert report["coldkey"]["amount_rao"] == 200


def test_snapshot_wallet_check_reports_hotkey_not_claimable(tmp_path: Path) -> None:
    module = _load_module()
    _write_snapshot(tmp_path)

    report = module.check_snapshot_wallet(tmp_path, "5Hotkey")

    assert report["classification"] == "hotkey_with_staked_alpha"
    assert report["claimable"] is False
    assert report["coldkey"]["amount_rao"] == 0
    assert report["hotkey_stake_edges"] == [
        {
            "coldkey": "5Alice",
            "netuid": 7,
            "alpha_units": 50,
            "included": True,
            "source": "AlphaV2",
        }
    ]
    assert report["next_action"] == "Ask one of the listed staking coldkeys to sign the genesis binding."

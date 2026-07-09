#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_SNAPSHOT_DIR = "/mnt/4tb/tao_fork_snapshot"
SOTA_UNITS_PER_RAO = 10**9
RAO_PER_TAO = 10**9


def _included(row: dict[str, str]) -> bool:
    return str(row.get("included", "True")).strip().lower() in {"1", "true", "yes"}


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _derive_address(mnemonic: str) -> str:
    try:
        from substrateinterface import Keypair
    except ImportError as exc:
        raise RuntimeError("substrateinterface is required when --mnemonic is used") from exc
    return str(Keypair.create_from_mnemonic(mnemonic).ss58_address)


def _load_snapshot_block(snapshot_dir: Path) -> dict[str, Any]:
    path = snapshot_dir / "genesis_snapshot_block.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "bittensor_block_number": payload.get("bittensor_block_number"),
        "bittensor_block_hash": payload.get("bittensor_block_hash"),
    }


def _read_coldkey_row(snapshot_dir: Path, address: str) -> tuple[int | None, dict[str, str] | None]:
    path = snapshot_dir / "coldkeys.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if str(row.get("coldkey") or "").strip() == address:
                return index, row
    return None, None


def _alpha_credit(snapshot_dir: Path, address: str) -> tuple[dict[str, int], int]:
    path = snapshot_dir / "alpha_exposures.csv"
    target_by_netuid: dict[str, int] = {}
    total_by_netuid: dict[str, int] = {}
    pool_by_netuid: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not _included(row):
                continue
            amount = _int(row, "total_alpha_units")
            if amount <= 0:
                continue
            netuid = str(int(row["netuid"]))
            total_by_netuid[netuid] = total_by_netuid.get(netuid, 0) + amount
            pool_by_netuid.setdefault(netuid, _int(row, "tao_in_pool_rao"))
            if str(row.get("coldkey") or "").strip() == address:
                target_by_netuid[netuid] = target_by_netuid.get(netuid, 0) + amount

    credit_by_netuid: dict[str, int] = {}
    for netuid, target_units in sorted(target_by_netuid.items(), key=lambda item: int(item[0])):
        denominator = total_by_netuid.get(netuid, 0)
        pool_rao = pool_by_netuid.get(netuid, 0)
        if target_units <= 0 or denominator <= 0 or pool_rao <= 0:
            continue
        credit = target_units * pool_rao // denominator
        if credit > 0:
            credit_by_netuid[netuid] = credit
    return credit_by_netuid, sum(credit_by_netuid.values())


def _hotkey_stake_edges(snapshot_dir: Path, address: str) -> list[dict[str, Any]]:
    path = snapshot_dir / "stake_edges.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("hotkey") or "").strip() != address:
                continue
            rows.append(
                {
                    "coldkey": str(row.get("coldkey") or "").strip(),
                    "netuid": int(row["netuid"]),
                    "alpha_units": int(row["alpha_units"]),
                    "included": _included(row),
                    "source": str(row.get("source") or "").strip(),
                }
            )
    return rows


def check_snapshot_wallet(snapshot_dir: Path, address: str) -> dict[str, Any]:
    snapshot_index, coldkey_row = _read_coldkey_row(snapshot_dir, address)
    block = _load_snapshot_block(snapshot_dir)
    hotkey_edges = _hotkey_stake_edges(snapshot_dir, address)

    result: dict[str, Any] = {
        "schema": "sota-snapshot-wallet-check/v1",
        "snapshot_dir": str(snapshot_dir),
        "snapshot": block,
        "address": address,
        "classification": "absent",
        "claimable": False,
        "coldkey": None,
        "hotkey_stake_edges": hotkey_edges,
        "next_action": "Use a snapshot coldkey with nonzero TAO or alpha credit.",
    }

    if coldkey_row is not None:
        direct_tao_rao = _int(coldkey_row, "tao_total_rao", _int(coldkey_row, "tao_free_rao") + _int(coldkey_row, "tao_reserved_rao"))
        alpha_credit_by_netuid, alpha_credit_rao = _alpha_credit(snapshot_dir, address)
        amount_rao = direct_tao_rao + alpha_credit_rao
        result["classification"] = "claimable_coldkey" if amount_rao > 0 else "zero_allocation_coldkey"
        result["claimable"] = amount_rao > 0
        result["coldkey"] = {
            "snapshot_index": snapshot_index,
            "included": _included(coldkey_row),
            "direct_tao_rao": direct_tao_rao,
            "alpha_credit_rao": alpha_credit_rao,
            "alpha_credit_rao_by_netuid": alpha_credit_by_netuid,
            "amount_rao": amount_rao,
            "amount_tao": amount_rao / RAO_PER_TAO,
            "amount_units": amount_rao * SOTA_UNITS_PER_RAO,
        }
        result["next_action"] = (
            "Sign the genesis binding with this coldkey."
            if amount_rao > 0
            else "This coldkey is in the snapshot but has zero claimable SOTA."
        )

    if hotkey_edges and not result["claimable"]:
        result["classification"] = "hotkey_with_staked_alpha"
        result["next_action"] = "Ask one of the listed staking coldkeys to sign the genesis binding."

    return result


def _print_human(report: dict[str, Any]) -> None:
    print(f"Address: {report['address']}")
    print(f"Classification: {report['classification']}")
    print(f"Claimable: {str(report['claimable']).lower()}")
    coldkey = report.get("coldkey") or {}
    if coldkey:
        print(f"Direct TAO rao: {coldkey['direct_tao_rao']}")
        print(f"Alpha credit rao: {coldkey['alpha_credit_rao']}")
        print(f"Total SOTA: {coldkey['amount_tao']:.9f}")
    edges = list(report.get("hotkey_stake_edges") or [])
    if edges:
        print(f"Hotkey stake edges: {len(edges)}")
        for edge in edges:
            print(f"- coldkey={edge['coldkey']} netuid={edge['netuid']} alpha_units={edge['alpha_units']} source={edge['source']}")
    print(f"Next action: {report['next_action']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a Bittensor key/address can claim SOTA genesis.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--address", help="SS58 address to check.")
    source.add_argument("--mnemonic", help="Mnemonic to derive and check. The mnemonic is not printed or stored.")
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser()
    address = str(args.address or "").strip() or _derive_address(str(args.mnemonic or "").strip())
    try:
        report = check_snapshot_wallet(snapshot_dir, address)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

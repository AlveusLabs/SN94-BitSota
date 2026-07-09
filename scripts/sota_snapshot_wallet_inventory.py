#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT_DIR = "/mnt/4tb/tao_fork_snapshot"
DEFAULT_WALLET_ROOTS = ("~/.bittensor/wallets", "~/.bitsota/wallets")


def _load_checker():
    path = SCRIPT_DIR / "sota_snapshot_wallet_check.py"
    spec = importlib.util.spec_from_file_location("sota_snapshot_wallet_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load snapshot checker: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _find_first(payload: Any, keys: set[str]) -> str:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            found = _find_first(value, keys)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_first(value, keys)
            if found:
                return found
    return ""


def _coldkeypub_address(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _find_first(payload, {"ss58Address", "ss58_address", "address"})


def scan_wallets(snapshot_dir: Path, wallet_roots: list[Path]) -> dict[str, Any]:
    checker = _load_checker()
    wallets: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for root in wallet_roots:
        expanded = root.expanduser()
        if not expanded.exists():
            continue
        for coldkeypub in sorted(expanded.glob("*/coldkeypub.txt")):
            resolved = coldkeypub.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            wallet_dir = coldkeypub.parent
            try:
                address = _coldkeypub_address(coldkeypub)
                if not address:
                    raise ValueError("missing ss58 address")
                check = checker.check_snapshot_wallet(snapshot_dir, address)
                wallets.append(
                    {
                        "wallet_name": wallet_dir.name,
                        "wallet_dir": str(wallet_dir),
                        "coldkeypub": str(coldkeypub),
                        "address": address,
                        "classification": check["classification"],
                        "claimable": bool(check["claimable"]),
                        "coldkey": check.get("coldkey"),
                        "hotkey_stake_edge_count": len(check.get("hotkey_stake_edges") or []),
                        "next_action": check["next_action"],
                    }
                )
            except Exception as exc:
                wallets.append(
                    {
                        "wallet_name": wallet_dir.name,
                        "wallet_dir": str(wallet_dir),
                        "coldkeypub": str(coldkeypub),
                        "address": "",
                        "classification": "error",
                        "claimable": False,
                        "error": str(exc),
                        "next_action": "Inspect this wallet's coldkeypub file.",
                    }
                )

    claimable = [wallet for wallet in wallets if wallet.get("claimable")]
    return {
        "schema": "sota-snapshot-wallet-inventory/v1",
        "snapshot_dir": str(snapshot_dir),
        "wallet_roots": [str(root.expanduser()) for root in wallet_roots],
        "wallet_count": len(wallets),
        "claimable_wallet_count": len(claimable),
        "claimable": bool(claimable),
        "wallets": wallets,
        "next_action": (
            "Use a claimable wallet name with scripts/sota_sign_snapshot_binding.py."
            if claimable
            else "No local public coldkey matched a claimable snapshot allocation."
        ),
    }


def _print_human(report: dict[str, Any]) -> None:
    print(f"Wallets scanned: {report['wallet_count']}")
    print(f"Claimable wallets: {report['claimable_wallet_count']}")
    for wallet in report["wallets"]:
        amount = ""
        coldkey = wallet.get("coldkey") or {}
        if coldkey:
            amount = f" amount_sota={coldkey.get('amount_tao', 0):.9f}"
        print(
            f"- {wallet['wallet_name']} {wallet.get('address', '')} "
            f"classification={wallet['classification']} claimable={str(wallet['claimable']).lower()}{amount}"
        )
        if wallet.get("classification") == "hotkey_with_staked_alpha":
            print(f"  hotkey_stake_edges={wallet.get('hotkey_stake_edge_count', 0)}")
        if wallet.get("error"):
            print(f"  error={wallet['error']}")
    print(f"Next action: {report['next_action']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan local public coldkey files for claimable SOTA genesis allocations.")
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--wallet-root",
        action="append",
        default=[],
        help="Wallet root to scan. Can be repeated. Defaults to ~/.bittensor/wallets and ~/.bitsota/wallets.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_dir = Path(str(args.snapshot_dir)).expanduser()
    wallet_roots = [Path(root) for root in args.wallet_root] if args.wallet_root else [Path(root) for root in DEFAULT_WALLET_ROOTS]
    report = scan_wallets(snapshot_dir, wallet_roots)
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

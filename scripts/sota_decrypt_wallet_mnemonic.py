#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bittensor_network.keyfile import KeyFileError, decrypt_keyfile_data  # noqa: E402


DEFAULT_WALLET_PATH = "~/.bittensor/wallets/"
MNEMONIC_KEYS = {"secretPhrase", "secret_phrase", "mnemonic"}
ADDRESS_KEYS = {"ss58Address", "ss58_address", "address"}


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


def _keyfile_path(args: argparse.Namespace) -> Path:
    wallet_root = Path(str(args.wallet_path)).expanduser()
    wallet_dir = wallet_root / str(args.wallet_name)
    if args.key == "coldkey":
        return wallet_dir / "coldkey"
    return wallet_dir / "hotkeys" / str(args.hotkey_name)


def run(args: argparse.Namespace) -> int:
    path = _keyfile_path(args)
    if not path.is_file():
        raise KeyFileError(f"keyfile not found: {path}")

    password = getpass.getpass(f"Enter {args.key} password for wallet {args.wallet_name}: ")
    decrypted = decrypt_keyfile_data(path.read_bytes(), password=password)
    payload = json.loads(decrypted.decode("utf-8"))

    mnemonic = _find_first(payload, MNEMONIC_KEYS)
    if not mnemonic:
        raise KeyFileError(
            "decrypted keyfile does not contain a mnemonic/secretPhrase; "
            "it may only contain a raw secret seed"
        )

    if args.mnemonic_only:
        print(mnemonic)
        return 0

    address = _find_first(payload, ADDRESS_KEYS)
    if address:
        print(f"SS58 address: {address}")
    print(f"Mnemonic: {mnemonic}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decrypt a local SOTA/Bittensor wallet keyfile and print its mnemonic."
    )
    parser.add_argument("--wallet-name", required=True, help="Wallet directory name.")
    parser.add_argument("--wallet-path", default=DEFAULT_WALLET_PATH)
    parser.add_argument("--key", choices=("coldkey", "hotkey"), default="coldkey")
    parser.add_argument("--hotkey-name", default="default")
    parser.add_argument(
        "--mnemonic-only",
        action="store_true",
        help="Print only the mnemonic phrase, with no labels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except (KeyFileError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

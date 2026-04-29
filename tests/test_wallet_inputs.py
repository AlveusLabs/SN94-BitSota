from __future__ import annotations

import pytest
from substrateinterface import Keypair

from miner.wallet_inputs import extract_mnemonic_from_text, load_wallet


def test_extract_mnemonic_prefers_hotkey_label() -> None:
    coldkey_mnemonic = Keypair.generate_mnemonic()
    hotkey_mnemonic = Keypair.generate_mnemonic()

    text = f"""
    Coldkey mnemonic: {coldkey_mnemonic}
    Hotkey mnemonic: {hotkey_mnemonic}
    """

    assert extract_mnemonic_from_text(text) == hotkey_mnemonic


def test_extract_mnemonic_rejects_ambiguous_unlabeled_file() -> None:
    mnemonic_a = Keypair.generate_mnemonic()
    mnemonic_b = Keypair.generate_mnemonic()

    with pytest.raises(ValueError, match="multiple mnemonic candidates"):
        extract_mnemonic_from_text(f"{mnemonic_a}\n\n{mnemonic_b}\n")


def test_load_wallet_accepts_wallet_file(tmp_path) -> None:
    mnemonic = Keypair.generate_mnemonic()
    wallet_file = tmp_path / "Wallet mine.txt"
    wallet_file.write_text(f"Hotkey mnemonic: {mnemonic}\n", encoding="utf-8")

    wallet = load_wallet(wallet_file=str(wallet_file))

    assert wallet.hotkey.ss58_address == Keypair.create_from_mnemonic(mnemonic).ss58_address

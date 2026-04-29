from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from substrateinterface import Keypair

from bittensor_network.wallet import Wallet


_MNEMONIC_WORD_COUNTS = (24, 21, 18, 15, 12)
_HOTKEY_LABELS = (
    "hotkey mnemonic",
    "hotkey_mnemonic",
    "hotkey seed",
    "hotkey_seed",
    "hotkey phrase",
    "hotkey_phrase",
)
_GENERIC_LABELS = (
    "mnemonic",
    "secretphrase",
    "secret_phrase",
    "seed phrase",
    "seed_phrase",
)


@dataclass(slots=True)
class EphemeralWallet:
    hotkey: Keypair


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z]+", str(value or "").lower())).strip()


def _is_valid_mnemonic(value: str) -> bool:
    phrase = _normalized_words(value)
    if not phrase:
        return False
    try:
        Keypair.create_from_mnemonic(phrase)
    except Exception:
        return False
    return True


def _collect_labeled_candidates(payload: Any, *, wanted_labels: tuple[str, ...]) -> list[str]:
    wanted = {label.lower() for label in wanted_labels}
    found: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                normalized_key = _normalized_words(str(key))
                if normalized_key in wanted and isinstance(inner, str):
                    found.append(str(inner))
                _walk(inner)
        elif isinstance(value, list):
            for inner in value:
                _walk(inner)

    _walk(payload)
    return found


def _iter_line_candidates(text: str, *, labels: tuple[str, ...]) -> list[str]:
    candidates: list[str] = []
    patterns = [
        re.compile(rf"(?i)\b{re.escape(label)}\b\s*[:=]\s*(.+)$")
        for label in labels
    ]
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for pattern in patterns:
            match = pattern.search(line)
            if match is None:
                continue
            candidates.append(match.group(1).strip())
    return candidates


def _unique_valid_candidates(candidates: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalized_words(candidate)
        if not normalized or normalized in seen or not _is_valid_mnemonic(normalized):
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def extract_mnemonic_from_text(text: str) -> str:
    raw_text = str(text or "")
    direct = _normalized_words(raw_text)
    if direct and _is_valid_mnemonic(direct):
        return direct

    try:
        payload = json.loads(raw_text)
    except Exception:
        payload = None

    if payload is not None:
        hotkey_candidates = _unique_valid_candidates(
            _collect_labeled_candidates(payload, wanted_labels=_HOTKEY_LABELS)
        )
        if len(hotkey_candidates) == 1:
            return hotkey_candidates[0]
        if len(hotkey_candidates) > 1:
            raise ValueError("wallet file contains multiple hotkey mnemonics")

        generic_candidates = _unique_valid_candidates(
            _collect_labeled_candidates(payload, wanted_labels=_GENERIC_LABELS)
        )
        if len(generic_candidates) == 1:
            return generic_candidates[0]
        if len(generic_candidates) > 1:
            raise ValueError("wallet file contains multiple mnemonic candidates")

    hotkey_line_candidates = _unique_valid_candidates(
        _iter_line_candidates(raw_text, labels=_HOTKEY_LABELS)
    )
    if len(hotkey_line_candidates) == 1:
        return hotkey_line_candidates[0]
    if len(hotkey_line_candidates) > 1:
        raise ValueError("wallet file contains multiple hotkey mnemonics")

    generic_line_candidates = _unique_valid_candidates(
        _iter_line_candidates(raw_text, labels=_GENERIC_LABELS)
    )
    if len(generic_line_candidates) == 1:
        return generic_line_candidates[0]
    if len(generic_line_candidates) > 1:
        raise ValueError("wallet file contains multiple mnemonic candidates")

    words = re.findall(r"[a-z]+", raw_text.lower())
    fallback_candidates: list[str] = []
    seen: set[str] = set()
    for count in _MNEMONIC_WORD_COUNTS:
        if len(words) < count:
            continue
        for start in range(len(words) - count + 1):
            phrase = " ".join(words[start : start + count])
            if phrase in seen or not _is_valid_mnemonic(phrase):
                continue
            seen.add(phrase)
            fallback_candidates.append(phrase)
    if len(fallback_candidates) == 1:
        return fallback_candidates[0]
    if len(fallback_candidates) > 1:
        raise ValueError(
            "wallet file contains multiple mnemonic candidates; label the hotkey mnemonic or pass it explicitly"
        )
    raise ValueError("wallet file does not contain a valid mnemonic")


def load_mnemonic_from_file(path: str) -> str:
    wallet_path = Path(str(path or "")).expanduser()
    if not wallet_path.exists():
        raise FileNotFoundError(f"wallet file not found: {wallet_path}")
    return extract_mnemonic_from_text(wallet_path.read_text(encoding="utf-8"))


def load_wallet(
    *,
    hotkey_mnemonic: str = "",
    wallet_file: str = "",
    wallet_name: str = "default",
    wallet_hotkey: str = "default",
    wallet_path: str = "~/.bittensor/wallets/",
) -> EphemeralWallet | Any:
    mnemonic = str(hotkey_mnemonic or "").strip()
    if not mnemonic and str(wallet_file or "").strip():
        mnemonic = load_mnemonic_from_file(str(wallet_file))
    if mnemonic:
        return EphemeralWallet(hotkey=Keypair.create_from_mnemonic(mnemonic))
    return Wallet(
        name=str(wallet_name or "default"),
        hotkey=str(wallet_hotkey or "default"),
        path=str(wallet_path or "~/.bittensor/wallets/"),
    )


def load_signer_keypair(
    *,
    mnemonic: str = "",
    wallet_file: str = "",
    wallet_name: str = "default",
    wallet_hotkey: str = "default",
    wallet_path: str = "~/.bittensor/wallets/",
) -> Keypair:
    wallet = load_wallet(
        hotkey_mnemonic=mnemonic,
        wallet_file=wallet_file,
        wallet_name=wallet_name,
        wallet_hotkey=wallet_hotkey,
        wallet_path=wallet_path,
    )
    signer = getattr(wallet, "hotkey", None)
    if signer is None:
        raise RuntimeError("wallet signer is unavailable")
    return signer

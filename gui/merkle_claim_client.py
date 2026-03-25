from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import requests
from substrateinterface import Keypair, SubstrateInterface
from substrateinterface.contracts import ContractInstance


_CLAIM_GAS_LIMIT = {"ref_time": 50_000_000_000, "proof_size": 2_000_000}
_CLAIM_STORAGE_DEPOSIT_LIMIT = 1_000_000_000


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(str(value or "0").strip() or "0")


def _normalize_endpoint(value: str) -> str:
    return str(value or "").strip().rstrip("/")


def _claim_state_path() -> Path:
    return Path.home() / ".bitsota" / "merkle_claim_state.json"


def _load_claim_state() -> dict[str, list[str]]:
    path = _claim_state_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    str(k): [str(v) for v in list(values or [])]
                    for k, values in data.items()
                    if isinstance(k, str)
                }
    except Exception:
        pass
    return {}


def _save_claim_state(data: dict[str, list[str]]) -> None:
    path = _claim_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pass


def _candidate_pool_claim_endpoint(pool_endpoint: str) -> str:
    pool_endpoint = _normalize_endpoint(pool_endpoint)
    if not pool_endpoint:
        return ""
    parsed = urlparse(pool_endpoint)
    hostname = parsed.hostname or ""
    if hostname not in {"127.0.0.1", "localhost"}:
        return ""
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:8844"
    elif parsed.scheme == "https":
        netloc = f"{hostname}:8844"
    else:
        netloc = f"{hostname}:8844"
    return urlunparse((parsed.scheme or "http", netloc, "", "", "", "")).rstrip("/")


def resolve_claim_endpoint(*, explicit: str, pool_endpoint: str) -> str:
    endpoint = _normalize_endpoint(explicit)
    if endpoint:
        return endpoint
    return _candidate_pool_claim_endpoint(pool_endpoint)


def resolve_metadata_path(explicit: str = "") -> str:
    candidate = str(explicit or "").strip()
    if candidate:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path.resolve())
    repo_root = Path(__file__).resolve().parents[2]
    fallbacks = [
        repo_root / "Pool" / "new_merkle" / "app" / "assets" / "merklepool.json",
        repo_root / "Pool" / "new_merkle" / "target" / "ink" / "merklepool.json",
        repo_root / "Pool" / "new_merkle" / "app" / "assets" / "merklepool.contract",
    ]
    for path in fallbacks:
        if path.exists():
            return str(path.resolve())
    return candidate


@dataclass(frozen=True)
class MerkleClaimPackage:
    epoch: int
    index: int
    recipient_address: str
    amount_rao: int
    proof: list[str]
    root: str
    identity_address: str

    @property
    def claim_key(self) -> tuple[int, int]:
        return (self.epoch, self.index)


def _normalize_claim_package(payload: dict[str, Any]) -> MerkleClaimPackage:
    recipient_address = (
        str(payload.get("recipient_coldkey") or "").strip()
        or str(payload.get("recipient_address") or "").strip()
        or str(payload.get("hotkey") or "").strip()
    )
    if not recipient_address:
        raise ValueError("claim package missing recipient address")
    proof = [str(item).strip() for item in list(payload.get("proof") or []) if str(item).strip()]
    return MerkleClaimPackage(
        epoch=_to_int(payload.get("epoch")),
        index=_to_int(payload.get("index")),
        recipient_address=recipient_address,
        amount_rao=_to_int(payload.get("amount") if payload.get("amount") is not None else payload.get("amount_units")),
        proof=proof,
        root=str(payload.get("root") or "").strip(),
        identity_address=str(payload.get("hotkey") or recipient_address).strip(),
    )


class MerkleClaimClient:
    def __init__(
        self,
        *,
        claim_endpoint: str,
        onchain_ws_url: str,
        contract_address: str,
        metadata_path: str,
        timeout_s: float = 10.0,
    ) -> None:
        self.claim_endpoint = _normalize_endpoint(claim_endpoint)
        self.onchain_ws_url = str(onchain_ws_url or "").strip()
        self.contract_address = str(contract_address or "").strip()
        self.metadata_path = str(metadata_path or "").strip()
        self.timeout_s = float(timeout_s)

    def is_configured(self) -> bool:
        return bool(
            self.claim_endpoint and self.onchain_ws_url and self.contract_address and self.metadata_path
        )

    def assert_configured(self) -> None:
        if not self.claim_endpoint:
            raise RuntimeError("Merkle claim endpoint is not configured")
        if not self.onchain_ws_url:
            raise RuntimeError("On-chain websocket URL is not configured")
        if not self.contract_address:
            raise RuntimeError("On-chain contract address is not configured")
        if not self.metadata_path:
            raise RuntimeError("On-chain contract metadata path is not configured")
        if not Path(self.metadata_path).expanduser().exists():
            raise RuntimeError(f"Contract metadata file not found: {self.metadata_path}")

    def _get_json(self, path: str) -> Any:
        url = f"{self.claim_endpoint}{path}"
        response = requests.get(url, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def _claim_scope_key(self, identity_address: str) -> str:
        return f"{self.contract_address}::{str(identity_address or '').strip()}"

    @staticmethod
    def _claim_key_str(claim_key: tuple[int, int]) -> str:
        return f"{int(claim_key[0])}:{int(claim_key[1])}"

    def locally_claimed_keys(self, *, identity_address: str) -> set[tuple[int, int]]:
        key = self._claim_scope_key(identity_address)
        values = _load_claim_state().get(key) or []
        out: set[tuple[int, int]] = set()
        for value in values:
            try:
                epoch_str, index_str = str(value).split(":", 1)
                out.add((int(epoch_str), int(index_str)))
            except Exception:
                continue
        return out

    def is_locally_claimed(self, claim: MerkleClaimPackage) -> bool:
        return claim.claim_key in self.locally_claimed_keys(identity_address=claim.identity_address)

    def mark_locally_claimed(self, claim: MerkleClaimPackage) -> None:
        key = self._claim_scope_key(claim.identity_address)
        state = _load_claim_state()
        values = set(str(v) for v in list(state.get(key) or []))
        values.add(self._claim_key_str(claim.claim_key))
        state[key] = sorted(values)
        _save_claim_state(state)

    def list_claim_packages(self, *, hotkey: str, max_epochs: int = 20) -> list[MerkleClaimPackage]:
        self.assert_configured()
        hotkey = str(hotkey or "").strip()
        if not hotkey:
            return []
        epochs_payload = self._get_json("/epochs")
        epochs = list(epochs_payload.get("epochs") or [])
        claimed_keys = self.locally_claimed_keys(identity_address=hotkey)
        claims: list[MerkleClaimPackage] = []
        for epoch in sorted((_to_int(v) for v in epochs), reverse=True)[: max(1, int(max_epochs))]:
            try:
                payload = self._get_json(f"/epoch/{epoch}/claim/{hotkey}")
            except requests.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 404:
                    continue
                raise
            claim = _normalize_claim_package(payload)
            if claim.claim_key in claimed_keys:
                continue
            claims.append(claim)
        return claims

    def submit_claim(self, *, signer: Keypair, claim: MerkleClaimPackage, transfer: bool = True) -> dict[str, Any]:
        self.assert_configured()
        substrate = SubstrateInterface(url=self.onchain_ws_url)
        try:
            substrate.runtime_config.update_type_registry_types({"Balance": "u64"})
            contract = ContractInstance.create_from_address(
                contract_address=self.contract_address,
                metadata_file=str(Path(self.metadata_path).expanduser()),
                substrate=substrate,
            )
            proof_hex = [p if str(p).startswith("0x") else f"0x{p}" for p in claim.proof]
            receipt = contract.exec(
                signer,
                "claim_single",
                args={
                    "data": {
                        "epoch": int(claim.epoch),
                        "index": int(claim.index),
                        "recipient_coldkey": claim.recipient_address,
                        "amount": int(claim.amount_rao),
                        "proof": proof_hex,
                    },
                    "transfer": bool(transfer),
                },
                gas_limit=dict(_CLAIM_GAS_LIMIT),
                storage_deposit_limit=int(_CLAIM_STORAGE_DEPOSIT_LIMIT),
                wait_for_inclusion=True,
            )
            if not getattr(receipt, "is_success", False):
                message = str(getattr(receipt, "error_message", "") or "claim_single failed")
                raise RuntimeError(message)
            self.mark_locally_claimed(claim)
            extrinsic_hash = ""
            if getattr(receipt, "extrinsic_hash", None) is not None:
                try:
                    extrinsic_hash = str(receipt.extrinsic_hash)
                except Exception:
                    extrinsic_hash = ""
            return {
                "ok": True,
                "epoch": claim.epoch,
                "index": claim.index,
                "extrinsic_hash": extrinsic_hash,
            }
        finally:
            try:
                substrate.close()
            except Exception:
                pass

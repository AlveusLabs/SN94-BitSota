from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import time
from typing import Any

from substrateinterface import Keypair


AUTH_VERSION = "BT-AUTH-V1"


@dataclass(frozen=True)
class SignedHeaders:
    hotkey: str
    timestamp: str
    signature: str

    def as_headers(self) -> dict[str, str]:
        return {
            "X-Hotkey": self.hotkey,
            "X-Timestamp": self.timestamp,
            "X-Signature": self.signature,
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonicalize_body(body: Any) -> bytes:
    if body is None:
        return b""
    if isinstance(body, (dict, list, tuple, int, float, bool)):
        return _canonical_json_bytes(body)
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return body.encode("utf-8")
    return _canonical_json_bytes(body)


def body_sha256(body: Any) -> str:
    return sha256(canonicalize_body(body)).hexdigest()


def build_auth_message(
    *,
    method: str,
    path: str,
    query: str = "",
    timestamp: str,
    body: Any = None,
) -> str:
    return "\n".join(
        [
            AUTH_VERSION,
            str(method).upper(),
            str(path),
            str(query or ""),
            str(timestamp),
            body_sha256(body),
        ]
    )


def sign_hotkey_request(
    *,
    keypair: Keypair,
    method: str,
    path: str,
    query: str = "",
    body: Any = None,
    timestamp: int | None = None,
) -> SignedHeaders:
    ts = str(int(timestamp if timestamp is not None else time.time()))
    message = build_auth_message(
        method=method,
        path=path,
        query=query,
        timestamp=ts,
        body=body,
    )
    signature = keypair.sign(message.encode("utf-8")).hex()
    return SignedHeaders(hotkey=keypair.ss58_address, timestamp=ts, signature=signature)

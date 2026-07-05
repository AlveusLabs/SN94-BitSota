#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from substrateinterface import Keypair, KeypairType


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bittensor_network.wallet import Wallet  # noqa: E402


DEFAULT_WALLET_PATH = "~/.bittensor/wallets/"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _post_json(url: str, payload: dict[str, Any], *, timeout: float = 20) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body) if body.strip() else {}
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object from {url}")
    return parsed


def _keypair_from_args(args: argparse.Namespace) -> Keypair:
    dev_uri = str(getattr(args, "dev_coldkey_uri", "") or "").strip()
    if dev_uri:
        return Keypair.create_from_uri(dev_uri, crypto_type=KeypairType.SR25519)
    password = os.environ.get(args.password_env, None) if args.password_env else None
    wallet = Wallet(name=args.wallet_name, hotkey="default", path=args.wallet_path)
    return wallet.get_coldkey(password=password)


def _binding_message(args: argparse.Namespace, keypair: Keypair) -> dict[str, Any]:
    if args.message_file:
        return _load_json(args.message_file)
    if not args.claims_api_url:
        raise ValueError("--claims-api-url is required when --message-file is not supplied")
    if not args.reward_address:
        raise ValueError("--reward-address is required when --message-file is not supplied")
    return _post_json(
        _url(args.claims_api_url, "/api/v1/base/genesis/binding-message"),
        {"coldkey": keypair.ss58_address, "reward_address": args.reward_address},
        timeout=args.timeout,
    )


def build_signed_binding(args: argparse.Namespace) -> dict[str, Any]:
    keypair = _keypair_from_args(args)
    payload = _binding_message(args, keypair)
    result = dict(payload.get("result") if isinstance(payload.get("result"), dict) else payload)
    message = result.get("message")
    signing_payload = result.get("signing_payload") or result.get("signingPayload")
    if not isinstance(message, dict) or not isinstance(signing_payload, str):
        raise ValueError("binding message response must include message and signing_payload")
    coldkey = str(message.get("coldkey") or "").strip()
    if coldkey != keypair.ss58_address:
        raise ValueError(f"local coldkey {keypair.ss58_address} does not match binding message coldkey {coldkey}")
    signature = "0x" + keypair.sign(signing_payload.encode("utf-8")).hex()
    return {
        "schema": "sota-snapshot-signed-binding/v1",
        "message": message,
        "signature": signature,
        "signing_payload_sha256": result.get("signing_payload_sha256") or result.get("signingPayloadSha256"),
        "snapshot_claim": result.get("snapshot_claim") or result.get("snapshotClaim"),
        "does_not": ["request_seed_phrase", "custody_user_keys", "bridge_alpha_tokens"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    signed = build_signed_binding(args)
    out = args.out
    if out:
        _write_json(out, signed)
    submit_result: dict[str, Any] | None = None
    if args.submit:
        if not args.claims_api_url:
            raise ValueError("--claims-api-url is required with --submit")
        submit_result = _post_json(
            _url(args.claims_api_url, "/api/v1/base/genesis/bindings"),
            {"message": signed["message"], "signature": signed["signature"]},
            timeout=args.timeout,
        )
    return {
        "ok": True,
        "status": "submitted" if submit_result else "signed",
        "coldkey": signed["message"]["coldkey"],
        "reward_address": signed["message"]["reward_address"],
        "claim_id": signed["message"]["claim_id"],
        "amount_units": str(signed["message"]["allocation_amount"]),
        "signed_binding": str(out) if out else None,
        "submit_result": submit_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sign a SOTA genesis snapshot binding with a local Bittensor coldkey."
    )
    parser.add_argument("--claims-api-url", default=os.environ.get("SOTA_CLAIMS_API_URL", ""))
    parser.add_argument("--reward-address", default="")
    parser.add_argument("--message-file", type=Path)
    parser.add_argument("--wallet-name", default=os.environ.get("BT_WALLET_NAME", "default"))
    parser.add_argument("--wallet-path", default=os.environ.get("BT_WALLET_PATH", DEFAULT_WALLET_PATH))
    parser.add_argument("--password-env", default="BT_WALLET_PASSWORD")
    parser.add_argument("--out", type=Path, default=Path("sota-signed-snapshot-binding.json"))
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument(
        "--dev-coldkey-uri",
        default="",
        help="Local testing only: sign with a Substrate dev URI such as //Alice instead of a wallet file.",
    )
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

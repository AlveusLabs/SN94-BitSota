from __future__ import annotations

import argparse
import json
from typing import Sequence

from gui.merkle_claim_client import MerkleClaimClient, resolve_metadata_path
from miner.wallet_inputs import load_signer_keypair


_DEFAULT_CLAIM_ENDPOINT = "https://3fhi3ukpyw.eu-central-1.awsapprunner.com/claims"
_DEFAULT_ONCHAIN_WS_URL = "wss://test.finney.opentensor.ai:443"
_DEFAULT_CONTRACT_ADDRESS = "5G1fuA6RPVCUu7K5ep7SWJLzQaqzdJAwchQHppkfVKzEVv49"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List and submit live autoresearch Merkle claim packages from the public SN94 repo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List currently claimable Merkle packages for a miner hotkey.")
    _add_common_args(list_parser)
    list_parser.add_argument("--hotkey", required=True)
    list_parser.add_argument("--max-epochs", type=int, default=20)

    claim_parser = subparsers.add_parser("claim", help="Submit one or more claim_single extrinsics for a miner hotkey.")
    _add_common_args(claim_parser)
    claim_parser.add_argument("--hotkey", required=True)
    claim_parser.add_argument("--max-epochs", type=int, default=20)
    claim_parser.add_argument("--all", action="store_true", help="Claim every currently available package.")
    claim_parser.add_argument("--no-transfer", action="store_true", help="Pass transfer=false to claim_single.")
    claim_parser.add_argument("--signer-mnemonic", default="")
    claim_parser.add_argument("--wallet-file", default="")
    claim_parser.add_argument("--wallet-name", default="default")
    claim_parser.add_argument("--wallet-hotkey", default="default")
    claim_parser.add_argument("--wallet-path", default="~/.bittensor/wallets/")

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--claim-endpoint", default=_DEFAULT_CLAIM_ENDPOINT)
    parser.add_argument("--onchain-ws-url", default=_DEFAULT_ONCHAIN_WS_URL)
    parser.add_argument("--contract-address", default=_DEFAULT_CONTRACT_ADDRESS)
    parser.add_argument("--metadata-path", default="")


def _build_client(args: argparse.Namespace) -> MerkleClaimClient:
    return MerkleClaimClient(
        claim_endpoint=str(args.claim_endpoint),
        onchain_ws_url=str(args.onchain_ws_url),
        contract_address=str(args.contract_address),
        metadata_path=resolve_metadata_path(str(getattr(args, "metadata_path", "") or "")),
    )


def _list_claims(args: argparse.Namespace) -> int:
    client = _build_client(args)
    claims = client.list_claim_packages(
        hotkey=str(args.hotkey),
        max_epochs=int(args.max_epochs),
    )
    payload = [
        {
            "epoch": int(claim.epoch),
            "index": int(claim.index),
            "hotkey": str(claim.identity_address),
            "recipient_coldkey": str(claim.recipient_address),
            "amount_rao": int(claim.amount_rao),
            "proof_count": len(claim.proof),
            "root": str(claim.root),
        }
        for claim in claims
    ]
    print(json.dumps(payload, indent=2))  # noqa: T201
    return 0


def _claim(args: argparse.Namespace) -> int:
    client = _build_client(args)
    claims = client.list_claim_packages(
        hotkey=str(args.hotkey),
        max_epochs=int(args.max_epochs),
    )
    if not claims:
        print("[]")  # noqa: T201
        return 0
    selected = claims if bool(args.all) else claims[:1]
    signer = load_signer_keypair(
        mnemonic=str(getattr(args, "signer_mnemonic", "") or ""),
        wallet_file=str(getattr(args, "wallet_file", "") or ""),
        wallet_name=str(getattr(args, "wallet_name", "default")),
        wallet_hotkey=str(getattr(args, "wallet_hotkey", "default")),
        wallet_path=str(getattr(args, "wallet_path", "~/.bittensor/wallets/")),
    )
    payload = [
        client.submit_claim(
            signer=signer,
            claim=claim,
            transfer=not bool(getattr(args, "no_transfer", False)),
        )
        for claim in selected
    ]
    print(json.dumps(payload, indent=2))  # noqa: T201
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "list":
        return _list_claims(args)
    if args.command == "claim":
        return _claim(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

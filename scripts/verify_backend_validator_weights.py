#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validator.backend_weight_policy import (  # noqa: E402
    BackendWeightPolicyError,
    SN94_BURN_UID,
    SN94_CHAIN_ENDPOINT,
    SN94_CONTRACT_HOTKEY,
    SN94_MAINNET_CONTRACT,
    SN94_NETUID,
    SN94_REQUIRED_BURN_REST_WEIGHT,
    SN94_REQUIRED_CONTRACT_WEIGHT,
    parse_backend_weight_override,
)


def _snapshot_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if base.endswith("/api/v1/reward-snapshot"):
        return base
    return f"{base}/api/v1/reward-snapshot"


def _load_payload(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    sources = [bool(args.url), bool(args.json_file), bool(args.stdin)]
    if sum(sources) != 1:
        raise SystemExit("Provide exactly one of --url, --json-file, or --stdin")

    if args.url:
        url = _snapshot_url(args.url)
        response = requests.get(url, timeout=float(args.timeout_s))
        response.raise_for_status()
        return response.json(), url

    if args.stdin:
        return json.load(sys.stdin), "stdin"

    path = Path(args.json_file).expanduser()
    return json.loads(path.read_text()), str(path)


def _validator_weights(payload: dict[str, Any]) -> dict[str, Any]:
    reward_policy = payload.get("reward_policy") if isinstance(payload, dict) else None
    if not isinstance(reward_policy, dict):
        raise BackendWeightPolicyError("payload.reward_policy must be an object")
    validator_weights = reward_policy.get("validator_weights")
    if not isinstance(validator_weights, dict):
        raise BackendWeightPolicyError("payload.reward_policy.validator_weights must be an object")
    return validator_weights


def _transition_status(validator_weights: dict[str, Any]) -> str:
    transition = validator_weights.get("transition_policy")
    if not isinstance(transition, dict):
        return ""
    return str(transition.get("status") or "").strip().lower()


def _print_report(payload: dict[str, Any], source: str) -> int:
    try:
        validator_weights = _validator_weights(payload)
        mode = str(validator_weights.get("mode") or "local").strip().lower()
        transition_status = _transition_status(validator_weights) or "unspecified"
        override = parse_backend_weight_override(
            payload["reward_policy"],
            enforce_sn94_contract_targets=True,
            sn94_require_burn_rest=True,
        )
    except BackendWeightPolicyError as exc:
        print("Decision: block")
        print("Validator: backend reward-snapshot policy verifier")
        print(f"Netuid: {SN94_NETUID}")
        print("Transaction: not checked")
        print("Block/timestamp: not checked")
        print("Signer: not checked")
        print("Contract hotkey allocation: unverifiable")
        print("Remaining allocation: unverifiable")
        print(f"Verification evidence: {source}")
        print(f"Mismatch or uncertainty: {exc}")
        print("Required owner: Bittensor / Validator Engineer")
        print("Next action: fix backend validator_weights before any SN94 set_weights operation")
        return 2

    if override is None:
        print("Decision: needs-owner")
        print("Validator: backend reward-snapshot policy verifier")
        print(f"Netuid: {SN94_NETUID}")
        print("Transaction: not checked")
        print("Block/timestamp: not checked")
        print("Signer: not checked")
        print("Contract hotkey allocation: 0.000000 via backend override")
        print("Remaining allocation: local validator fallback behavior")
        print(
            "Verification evidence: "
            f"{source}; validator_weights.mode={mode}; "
            f"transition_policy.status={transition_status}"
        )
        print(
            "Mismatch or uncertainty: backend is not publishing explicit SN94 "
            "10% contract-hotkey / 90% burn-rest targets"
        )
        print("Required owner: Bittensor / Validator Engineer")
        print(
            "Next action: keep payout activation blocked or publish active targets, "
            "then verify the on-chain set_weights transaction"
        )
        return 1

    contract_weight = override.contract_hotkey_weight
    burn_weight = override.burn_uid_weight
    decision = "needs-owner"
    uncertainty = (
        "latest SN94 set_weights transaction hash, block, timestamp, signer, "
        "and on-chain confirmation are not checked by this backend policy command"
    )
    if transition_status not in {"ready", "active"}:
        uncertainty = (
            f"transition_policy.status={transition_status}; {uncertainty}"
        )

    print(f"Decision: {decision}")
    print("Validator: backend reward-snapshot policy verifier")
    print(f"Netuid: {SN94_NETUID}")
    print("Transaction: not checked")
    print("Block/timestamp: not checked")
    print("Signer: not checked")
    print(
        "Contract hotkey allocation: "
        f"{contract_weight:.6f} to {SN94_CONTRACT_HOTKEY}"
    )
    print(
        "Remaining allocation: "
        f"{burn_weight:.6f} to burn UID {SN94_BURN_UID}"
    )
    print(
        "Verification evidence: "
        f"{source}; chain_endpoint={SN94_CHAIN_ENDPOINT}; "
        f"mainnet_contract={SN94_MAINNET_CONTRACT}; "
        f"mode={override.mode}; transition_policy.status={transition_status}; "
        f"required_contract_weight={SN94_REQUIRED_CONTRACT_WEIGHT:.2f}; "
        f"required_burn_rest_weight={SN94_REQUIRED_BURN_REST_WEIGHT:.2f}"
    )
    print(f"Mismatch or uncertainty: {uncertainty}")
    print("Required owner: Bittensor / Validator Engineer")
    print(
        "Next action: run SN94 on-chain weight-setting verification and record "
        "transaction hash, block, timestamp, signer, and confirmed allocation"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify backend reward-snapshot validator_weights for SN94 "
            "10% contract-hotkey / 90% burn-rest routing."
        )
    )
    parser.add_argument("--url", help="Backend base URL or full /api/v1/reward-snapshot URL")
    parser.add_argument("--json-file", help="Path to a saved reward-snapshot JSON payload")
    parser.add_argument("--stdin", action="store_true", help="Read reward-snapshot JSON from stdin")
    parser.add_argument("--timeout-s", type=float, default=10.0, help="HTTP timeout in seconds")
    args = parser.parse_args(argv)

    payload, source = _load_payload(args)
    return _print_report(payload, source)


if __name__ == "__main__":
    raise SystemExit(main())

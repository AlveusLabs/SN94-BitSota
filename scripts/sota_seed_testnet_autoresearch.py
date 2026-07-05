#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from eth_account import Account
from substrateinterface import Keypair


REPOS = Path("/home/mekaneeky/repos")
DOCS_REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
DEFAULT_COORDINATOR_URL = "https://zuyyfpgpnw.eu-central-1.awsapprunner.com"
DEFAULT_ADMIN_TOKEN_SECRET_ID = "base-sota/test/base-sepolia/autoresearch-admin-token"
DEFAULT_AWS_PROFILE = "moonrocklab-frankfurt"
DEFAULT_AWS_REGION = "eu-central-1"
DEFAULT_REWARD_KEY_FILE = DEFAULT_ARTIFACTS_DIR / "fresh-claim-wallet.json"
DEFAULT_EVIDENCE_OUT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-emission-evidence-fresh-public.json"
DEFAULT_REPORT_OUT = DEFAULT_ARTIFACTS_DIR / "base-sota-testnet-autoresearch-seed.json"
DEFAULT_MANIFEST = DEFAULT_ARTIFACTS_DIR / "base-sepolia-deployment-manifest.json"
DEFAULT_REPOSITORY = "https://github.com/AlveusLabs/autoresearch-bittensor.git"
DEFAULT_BASE_REF = "6decee76dff85ea03cad88b13b0332a82081ef16"
LANE_ID = "base:sota-local"
ONE_SOTA = 10**18
COMMITTEE_SIZE = 3
ANVIL_MINER_PRIVATE_KEY = "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"

sys.path.insert(0, str(DOCS_REPO / "scripts"))
from sota_emission_policy import frontier_capacitor_reward_policy, sota_epoch_budget_units  # noqa: E402
import sota_local_demo as local_demo  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_secret_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _aws_secret_string(secret_id: str, *, profile: str, region: str, timeout: float) -> str:
    cmd = [
        "aws",
        "secretsmanager",
        "get-secret-value",
        "--secret-id",
        secret_id,
        "--query",
        "SecretString",
        "--output",
        "text",
    ]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"aws exited {result.returncode}")
    value = result.stdout.strip()
    if not value or value == "None":
        raise RuntimeError(f"secret {secret_id!r} has no SecretString")
    if value.startswith("{"):
        payload = json.loads(value)
        if isinstance(payload, dict):
            for key in ("admin_token", "ADMIN_TOKEN", "token"):
                if payload.get(key):
                    return str(payload[key]).strip()
    return value


def _admin_token(args: argparse.Namespace) -> str:
    env_value = os.environ.get(args.admin_token_env, "").strip()
    if env_value:
        return env_value
    return _aws_secret_string(
        args.admin_token_secret_id,
        profile=args.aws_profile,
        region=args.aws_region,
        timeout=args.timeout,
    )


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _demo_validator_keypairs() -> list[tuple[str, Keypair]]:
    return [(name, Keypair.create_from_uri(f"//{name}")) for name in ("Bob", "Charlie", "Dave")]


def _load_or_create_reward_key(path: Path) -> dict[str, Any]:
    if path.exists():
        payload = _load_json(path)
        private_key = str(
            payload.get("private_key")
            or payload.get("reward_private_key")
            or payload.get("SOTA_TEST_WALLET_PRIVATE_KEY")
            or ""
        ).strip()
        if not private_key:
            raise RuntimeError(f"{path} does not contain a private_key field")
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key
        account = Account.from_key(private_key)
        return {
            "schema": str(payload.get("schema") or "sota-base-test-wallet-key/v1"),
            "address": account.address,
            "private_key": private_key,
            "path": str(path),
            "created": False,
        }
    account = Account.create(os.urandom(32))
    payload = {
        "schema": "sota-base-test-wallet-key/v1",
        "address": account.address,
        "private_key": account.key.hex(),
        "network": "base-sepolia",
        "purpose": "fresh first-time Base SOTA testnet claim wallet",
        "warning": "testnet only; do not fund with mainnet assets",
    }
    _write_secret_json(path, payload)
    return {**payload, "path": str(path), "created": True}


def _manifest_lane_registry(path: Path) -> str:
    manifest = _load_json(path)
    browser_safe = dict(dict(manifest.get("browser_safe") or {}).get("contract_addresses") or {})
    contracts = dict(manifest.get("contracts") or {})
    value = browser_safe.get("lane_registry") or dict(contracts.get("lane_registry") or {}).get("address")
    return str(value or "").strip()


def _next_epoch(coordinator_url: str, *, timeout: float) -> int:
    roots = local_demo._request_json(
        "GET",
        _url(coordinator_url, f"/api/v1/sota/emission-roots?subnet_id={LANE_ID}"),
        timeout=timeout,
    )
    max_epoch = 0
    for row in roots if isinstance(roots, list) else []:
        try:
            max_epoch = max(max_epoch, int(dict(row).get("epoch") or 0))
        except Exception:
            continue
    return max_epoch + 1


def _task_body(*, slug: str, repository: str, base_ref: str, validator_hotkeys: list[str]) -> dict[str, Any]:
    return {
        "slug": slug,
        "title": "SOTA Base Sepolia Self-Validation Frontier",
        "brief": "Test-only Base SOTA fork task used to seed a real self-validated claim root for Base Sepolia.",
        "repository": repository,
        "base_ref": base_ref,
        "setup_command": None,
        "benchmark_command": "python3 - <<'PY'\nprint({'heldout_ppl': 0.80})\nPY",
        "allowed_patch_paths": ["README.md"],
        "metric_name": "heldout_ppl",
        "metric_direction": "minimize",
        "competition_mode": "self_validation",
        "min_peer_evaluations": COMMITTEE_SIZE,
        "self_validation_policy": {
            "committee_size": COMMITTEE_SIZE,
            "committee_hotkeys": validator_hotkeys,
            "approval_threshold": 0.5,
            "min_effective_committee_size": float(COMMITTEE_SIZE),
            "max_approval_concentration": 1.0,
            "new_identity_weight": 1.0,
            "reputation_gain": 0.0,
            "max_reputation_weight": 1.0,
            "slash_tolerance": 0.05,
            "min_improvement": 0.0,
            "sortition_seed": "sota-base-sepolia-testnet",
        },
        "time_budget_seconds": 900,
    }


def _fresh_slug() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"sota-base-sepolia-self-validation-{stamp}"


def seed_public_autoresearch(args: argparse.Namespace) -> dict[str, Any]:
    admin_token = _admin_token(args)
    coordinator_url = args.coordinator_url.rstrip("/")
    reward_key = _load_or_create_reward_key(args.reward_key_file)
    reward_private_key = str(reward_key["private_key"])
    reward_address = Account.from_key(reward_private_key).address
    validators = _demo_validator_keypairs()
    validator_hotkeys = [keypair.ss58_address for _, keypair in validators]
    slug = args.task_slug or _fresh_slug()
    task = local_demo._request_json(
        "POST",
        _url(coordinator_url, "/api/v1/tasks"),
        _task_body(
            slug=slug,
            repository=args.repository,
            base_ref=args.base_ref,
            validator_hotkeys=validator_hotkeys,
        ),
        headers={"X-Admin-Token": admin_token},
        timeout=args.timeout,
    )
    lane_registry = args.lane_registry_address or _manifest_lane_registry(args.manifest)
    subnet = local_demo._request_json(
        "POST",
        _url(coordinator_url, "/api/v1/sota/subnets"),
        {
            "id": LANE_ID,
            "title": "SOTA Base Sepolia self-validation lane",
            "task_slugs": [task["slug"]],
            "budget_units_per_epoch": sota_epoch_budget_units(),
            "reward_policy": frontier_capacitor_reward_policy(),
            "active": True,
            "base_registry_chain_id": 84532,
            "base_registry_address": lane_registry,
            "base_registry_subnet_key": "sota-foundation/local-binary-frontier",
            "metadata": {"environment": "base-sepolia", "test_only": True, "fork": "sota-base"},
        },
        headers={"X-Admin-Token": admin_token},
        timeout=args.timeout,
    )
    alice = Keypair.create_from_uri("//Alice")
    claim_body = {
        "claim_description": f"Base Sepolia SOTA self-validation seed claim for reward wallet {reward_address}"
    }
    claim_path = f"/api/v1/tasks/{task['id']}/claim"
    claim = local_demo._request_json(
        "POST",
        _url(coordinator_url, claim_path),
        claim_body,
        headers=local_demo._signed_headers(alice, "POST", claim_path, claim_body),
        timeout=args.timeout,
    )
    metric_value = float(args.metric_value)
    submission_body = {
        "claim_id": claim["id"],
        "base_ref": args.base_ref,
        "patch": (
            "diff --git a/README.md b/README.md\n"
            "--- a/README.md\n"
            "+++ b/README.md\n"
            "@@\n"
            f"+Base Sepolia self-validation seed score: {metric_value:.6f}\n"
        ),
        "summary": "Test-only fresh-wallet Base Sepolia self-validation seed submission.",
        "claimed_metrics": {"heldout_ppl": metric_value},
    }
    local_demo._attach_evm_authorization(
        submission_body,
        claim_id=claim["id"],
        task_id=task["id"],
        miner_private_key=ANVIL_MINER_PRIVATE_KEY,
        reward_private_key=reward_private_key,
    )
    submission = local_demo._request_json(
        "POST",
        _url(coordinator_url, "/api/v1/submissions"),
        submission_body,
        headers=local_demo._signed_headers(alice, "POST", "/api/v1/submissions", submission_body),
        timeout=args.timeout,
    )
    evaluations = []
    evaluation_path = f"/api/v1/submissions/{submission['id']}/peer-evaluate"
    for name, validator in validators:
        body = {
            "status": "accepted",
            "observed_metrics": {"heldout_ppl": metric_value},
            "notes": f"{name} accepts test-only Base Sepolia self-validation seed submission",
        }
        evaluations.append(
            local_demo._request_json(
                "POST",
                _url(coordinator_url, evaluation_path),
                body,
                headers=local_demo._signed_headers(validator, "POST", evaluation_path, body),
                timeout=args.timeout,
            )
        )
    consensus = local_demo._request_json(
        "GET",
        _url(coordinator_url, f"/api/v1/submissions/{submission['id']}/peer-consensus"),
        timeout=args.timeout,
    )
    if consensus.get("status") != "accepted":
        raise RuntimeError(f"public self-validation consensus was not accepted: {consensus}")
    epoch = int(args.epoch or _next_epoch(coordinator_url, timeout=args.timeout))
    root = local_demo._request_json(
        "POST",
        _url(coordinator_url, f"/api/v1/sota/subnets/{LANE_ID}/epochs/{epoch}/root"),
        {"include_proofs": True},
        headers={"X-Admin-Token": admin_token},
        timeout=args.timeout,
    )
    evidence = local_demo._request_json(
        "GET",
        _url(coordinator_url, f"/api/v1/sota/subnets/{LANE_ID}/epochs/{epoch}/evidence"),
        timeout=args.timeout,
    )
    claims = list(dict(evidence.get("bundle") or {}).get("claim_list") or [])
    matching_claims = [
        dict(item)
        for item in claims
        if str(dict(item).get("reward_address") or "").lower() == reward_address.lower()
    ]
    if not matching_claims:
        raise RuntimeError(f"reward wallet {reward_address} is missing from public epoch {epoch} evidence")
    if args.require_single_claim and len(claims) != 1:
        raise RuntimeError(f"public epoch {epoch} has {len(claims)} claims, expected 1")
    _write_json(args.evidence_out, evidence)
    report = {
        "schema": "sota-base-testnet-autoresearch-seed/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coordinator_url": coordinator_url,
        "reward_address": reward_address,
        "reward_key_file": str(args.reward_key_file),
        "reward_key_created": bool(reward_key.get("created")),
        "task": task,
        "subnet": subnet,
        "claim": claim,
        "submission": submission,
        "evaluations": evaluations,
        "consensus": consensus,
        "epoch": epoch,
        "root": root,
        "evidence_path": str(args.evidence_out),
        "matching_claim": matching_claims[0],
        "claim_count": len(claims),
        "does_not": ["print_private_keys", "touch_production_bittensor", "touch_base_mainnet"],
    }
    _write_json(args.report_out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed real self-validation evidence on the Base SOTA testnet coordinator.")
    parser.add_argument("--coordinator-url", default=DEFAULT_COORDINATOR_URL)
    parser.add_argument("--admin-token-env", default="SOTA_AUTORESEARCH_ADMIN_TOKEN")
    parser.add_argument("--admin-token-secret-id", default=DEFAULT_ADMIN_TOKEN_SECRET_ID)
    parser.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", DEFAULT_AWS_PROFILE))
    parser.add_argument("--aws-region", default=os.environ.get("AWS_REGION", DEFAULT_AWS_REGION))
    parser.add_argument("--reward-key-file", type=Path, default=DEFAULT_REWARD_KEY_FILE)
    parser.add_argument("--evidence-out", type=Path, default=DEFAULT_EVIDENCE_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lane-registry-address", default="")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--base-ref", default=DEFAULT_BASE_REF)
    parser.add_argument("--task-slug", default="")
    parser.add_argument("--metric-value", type=float, default=0.80)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--require-single-claim", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = seed_public_autoresearch(args)
    if args.json:
        safe_report = {key: value for key, value in report.items() if key != "reward_key_file"}
        print(json.dumps(safe_report, indent=2, sort_keys=True, default=str))
    else:
        print("public Base Sepolia autoresearch evidence seeded")
        print(f"Reward address: {report['reward_address']}")
        print(f"Evidence: {report['evidence_path']}")
        print(f"Epoch: {report['epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

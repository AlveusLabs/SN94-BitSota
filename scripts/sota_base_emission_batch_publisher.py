#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sota_base_testnet_seed_artifacts import (
    BASE_SEPOLIA_CHAIN_ID,
    _finalized_claim_artifact,
    _load_json,
    _manifest_contract,
    _normalized_emission_artifact_inputs,
    _pending_emission_claim_artifact,
    _root_artifact,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = Path("/home/mekaneeky/repos/.sota-base-testnet")
DEFAULT_MANIFEST = DEFAULT_ARTIFACTS_DIR / "base-sepolia-deployment-manifest.json"
DEFAULT_COORDINATOR_URL = "https://zuyyfpgpnw.eu-central-1.awsapprunner.com"
DEFAULT_CLAIMS_API_URL = "https://gs4g5jntcn.eu-central-1.awsapprunner.com"
DEFAULT_LANE_ID = "base:sota-local"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _url(base: str, path: str, params: dict[str, Any] | None = None) -> str:
    text = base.rstrip("/") + "/" + path.lstrip("/")
    clean_params = {key: value for key, value in (params or {}).items() if value is not None}
    if clean_params:
        text += "?" + urlencode(clean_params)
    return text


def _admin_token(env_name: str) -> str:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        for key in ("admin_token", "token", env_name):
            value = payload.get(key)
            if value:
                return str(value)
    return raw


def _request_json(
    method: str,
    url: str,
    *,
    admin_token: str = "",
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any] | list[Any]:
    headers = {"accept": "application/json"}
    body = None
    if payload is not None:
        headers["content-type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    if admin_token:
        headers["X-Admin-Token"] = admin_token
    request = Request(url, data=body, method=method, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        decoded = response.read().decode("utf-8")
    loaded = json.loads(decoded)
    if not isinstance(loaded, (dict, list)):
        raise RuntimeError(f"expected JSON object or array from {url}")
    return loaded


def _run(cmd: list[str], *, cwd: Path, timeout: float) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"command exited {result.returncode}: {' '.join(cmd)}")
    stdout = result.stdout.strip()
    if not stdout:
        return {}
    try:
        loaded = json.loads(stdout)
    except json.JSONDecodeError:
        return {"stdout": stdout}
    return loaded if isinstance(loaded, dict) else {"stdout": stdout}


def _root_id_from_publish_result(path: Path) -> str:
    payload = _load_json(path)
    root_id = payload.get("root_id")
    if not root_id and isinstance(payload.get("root_published_event"), dict):
        root_id = dict(payload["root_published_event"]).get("root_id")
    if not root_id:
        raise RuntimeError(f"publish result {path} is missing root_id")
    return str(root_id)


def _manifest_owner(manifest: dict[str, Any]) -> str:
    root_registry = _manifest_contract(manifest, "root_registry")
    return str(dict(dict(manifest.get("roles") or {}).get("owner") or {}).get("address") or root_registry)


def fetch_coordinator_roots(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = _request_json(
        "GET",
        _url(args.coordinator_url, "/api/v1/sota/emission-roots", {"subnet_id": args.lane_id}),
        timeout=args.timeout,
    )
    if not isinstance(payload, list):
        raise RuntimeError("coordinator did not return an emission root list")
    roots = [dict(item) for item in payload if isinstance(item, dict)]
    roots = [root for root in roots if bool(root.get("ready_for_attestation")) and str(root.get("root") or "").startswith("0x")]
    if args.epoch is not None:
        roots = [root for root in roots if int(root.get("epoch") or 0) == int(args.epoch)]
    roots.sort(key=lambda item: int(item.get("epoch") or 0), reverse=not args.oldest_first)
    if not args.include_backlog and args.epoch is None:
        roots = roots[:1]
    return roots[: max(1, int(args.max_roots))]


def fetch_indexed_roots(args: argparse.Namespace) -> set[str]:
    payload = _request_json(
        "GET",
        _url(args.claims_api_url, "/api/v1/base/roots", {"subnet_id": args.lane_id}),
        timeout=args.timeout,
    )
    if not isinstance(payload, list):
        raise RuntimeError("claims API did not return a root list")
    return {str(dict(item).get("root") or "").lower() for item in payload if isinstance(item, dict) and dict(item).get("root")}


def import_local_finalized_artifacts(args: argparse.Namespace, *, admin_token: str) -> list[dict[str, Any]]:
    if not args.import_artifact or not admin_token:
        return []
    candidates = [
        args.out_dir / "base-sota-testnet-emission-claim-artifact.json",
        *sorted((args.out_dir / "emission-batches").glob("*/sota-emission-claim-artifact.json")),
    ]
    imported: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        artifact = _load_json(path)
        try:
            response = _request_json(
                "POST",
                _url(args.claims_api_url, "/api/v1/base/index/artifact"),
                admin_token=admin_token,
                payload=artifact,
                timeout=args.timeout,
            )
        except Exception as exc:
            imported.append({"path": str(path), "status": "red", "error": str(exc)})
        else:
            imported.append({"path": str(path), "status": "green", "response": response})
    return imported


def _batch_id(root: dict[str, Any]) -> str:
    epoch = int(root.get("epoch") or 0)
    merkle_root = str(root.get("root") or "").removeprefix("0x")
    return f"emission-epoch-{epoch:06d}-{merkle_root[:12]}"


def build_emission_artifacts(args: argparse.Namespace, root: dict[str, Any], *, batch_dir: Path) -> dict[str, Path | dict[str, Any]]:
    manifest = _load_json(args.manifest)
    if int(dict(manifest.get("chain") or {}).get("chain_id") or 0) != BASE_SEPOLIA_CHAIN_ID and not args.allow_local:
        raise RuntimeError("emission publisher requires the Base Sepolia manifest unless --allow-local is set")
    owner = _manifest_owner(manifest)
    epoch = int(root.get("epoch") or 0)
    evidence = _request_json(
        "GET",
        _url(args.coordinator_url, f"/api/v1/sota/subnets/{quote(args.lane_id, safe='')}/epochs/{epoch}/evidence"),
        timeout=args.timeout,
    )
    if not isinstance(evidence, dict):
        raise RuntimeError("coordinator evidence response must be an object")
    emission = _normalized_emission_artifact_inputs(
        evidence,
        min_accepted=args.min_accepted_count,
        min_committee=args.min_committee_count,
    )
    if str(emission["subnet_id"]) != str(args.lane_id):
        raise RuntimeError(f"emission evidence subnet_id {emission['subnet_id']!r} does not match lane_id {args.lane_id!r}")

    paths = {
        "evidence": batch_dir / "sota-emission-evidence.json",
        "root_artifact": batch_dir / "sota-emission-root-artifact.json",
        "claim_template": batch_dir / "sota-emission-claim-template.json",
        "publish_result": batch_dir / "sota-emission-root-publish-result.json",
        "claim_artifact": batch_dir / "sota-emission-claim-artifact.json",
        "lane_sync": batch_dir / "sota-emission-lane-sync.json",
    }
    _write_json(paths["evidence"], evidence)
    _write_json(
        paths["root_artifact"],
        _root_artifact(
            kind="emission",
            root=emission["root"],
            total_amount=int(emission["total_amount_units"]),
            policy_hash=emission["policy_hash"],
            attestation_hash=emission["attestation_hash"],
            nonce=emission["nonce"],
            metadata={"subnet_id": emission["subnet_id"], "epoch": emission["epoch"], "evidence_hash": emission["evidence_hash"]},
        ),
    )
    _write_json(paths["claim_template"], _pending_emission_claim_artifact(emission, owner=owner))
    return {"emission": emission, "paths": paths}


def publish_emission_root(args: argparse.Namespace, *, paths: dict[str, Path]) -> dict[str, Any]:
    if args.sync_lane:
        sync_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sota_base_sync_lane.py"),
            "--manifest",
            str(args.manifest),
            "--root-artifact",
            str(paths["root_artifact"]),
            "--claim-artifact",
            str(paths["claim_template"]),
            "--out",
            str(paths["lane_sync"]),
            "--json",
        ]
        if args.rpc_url:
            sync_cmd.extend(["--rpc-url", args.rpc_url])
        if args.sync_lane_broadcast:
            sync_cmd.append("--broadcast")
        else:
            sync_cmd.append("--allow-blocked")
        lane_sync = _run(sync_cmd, cwd=REPO_ROOT, timeout=args.command_timeout)
    else:
        lane_sync = {"status": "skipped"}

    publish_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sota_base_publish_root.py"),
        "--manifest",
        str(args.manifest),
        "--root-artifact",
        str(paths["root_artifact"]),
        "--kind",
        "emission",
        "--out",
        str(paths["publish_result"]),
        "--timeout",
        str(args.timeout),
    ]
    if args.rpc_url:
        publish_cmd.extend(["--rpc-url", args.rpc_url])
    if args.broadcast:
        publish_cmd.append("--broadcast")
    if args.allow_local:
        publish_cmd.append("--allow-local")
    publish = _run(publish_cmd, cwd=REPO_ROOT, timeout=args.command_timeout)
    return {"lane_sync": lane_sync, "publish": publish}


def publish_root(args: argparse.Namespace, root: dict[str, Any], *, admin_token: str) -> dict[str, Any]:
    batch_id = _batch_id(root)
    batch_dir = args.out_dir / "emission-batches" / batch_id
    built = build_emission_artifacts(args, root, batch_dir=batch_dir)
    paths = dict(built["paths"])
    emission = dict(built["emission"])
    actions = publish_emission_root(args, paths=paths)

    result: dict[str, Any] = {
        "schema": "sota-base-emission-batch-publisher/v1",
        "generated_at": _utc_now(),
        "status": "dry_run" if not args.broadcast else "published",
        "ok": True,
        "batch_id": batch_id,
        "lane_id": args.lane_id,
        "epoch": int(emission["epoch"]),
        "root": emission["root"],
        "claim_count": len(list(emission["claim_list"])),
        "total_amount_units": str(emission["total_amount_units"]),
        "artifacts": {key: str(value) for key, value in paths.items()},
        **actions,
        "does_not": ["touch_production_bittensor", "touch_base_mainnet", "custody_user_keys", "test_real_holder_claims"],
    }
    if not args.broadcast:
        result["message"] = "dry run built the emission root publish request; no transaction broadcast or indexer import performed"
        return result

    root_id = _root_id_from_publish_result(paths["publish_result"])
    claim_artifact = _finalized_claim_artifact(_load_json(paths["claim_template"]), root_id=root_id)
    _write_json(paths["claim_artifact"], claim_artifact)
    result["root_id"] = root_id
    if args.import_artifact:
        imported = _request_json(
            "POST",
            _url(args.claims_api_url, "/api/v1/base/index/artifact"),
            admin_token=admin_token,
            payload=claim_artifact,
            timeout=args.timeout,
        )
        result["indexer_import"] = imported
    return result


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    admin_token = _admin_token(args.admin_token_env)
    recovery_imports = import_local_finalized_artifacts(args, admin_token=admin_token)
    indexed_roots = fetch_indexed_roots(args)
    roots = fetch_coordinator_roots(args)
    if not roots:
        return {
            "schema": "sota-base-emission-batch-publisher/v1",
            "generated_at": _utc_now(),
            "status": "idle",
            "ok": True,
            "lane_id": args.lane_id,
            "recovery_imports": recovery_imports,
            "message": "no ready coordinator emission roots found",
        }

    published: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for root in roots:
        merkle_root = str(root.get("root") or "").lower()
        if merkle_root in indexed_roots:
            skipped.append({"epoch": int(root.get("epoch") or 0), "root": merkle_root, "reason": "already_indexed"})
            continue
        published.append(publish_root(args, root, admin_token=admin_token))
        if not args.include_backlog:
            break

    status = "published" if any(item.get("status") == "published" for item in published) else "dry_run" if published else "idle"
    result = {
        "schema": "sota-base-emission-batch-publisher/v1",
        "generated_at": _utc_now(),
        "status": status,
        "ok": all(bool(item.get("ok")) for item in published) if published else True,
        "lane_id": args.lane_id,
        "selected_root_count": len(roots),
        "published": published,
        "skipped": skipped,
        "recovery_imports": recovery_imports,
        "message": "latest ready coordinator emission root is already indexed" if skipped and not published else "",
    }
    _write_json(args.report_out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish accepted autoresearch emission roots to Base Sepolia and import claim artifacts.")
    parser.add_argument("--coordinator-url", default=os.environ.get("SOTA_COORDINATOR_URL", DEFAULT_COORDINATOR_URL))
    parser.add_argument("--claims-api-url", default=os.environ.get("SOTA_CLAIMS_API_URL", DEFAULT_CLAIMS_API_URL))
    parser.add_argument("--admin-token-env", default="SOTA_BASE_INDEXER_ADMIN_TOKEN")
    parser.add_argument("--lane-id", default=os.environ.get("SOTA_LANE_ID", DEFAULT_LANE_ID))
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--include-backlog", action="store_true")
    parser.add_argument("--oldest-first", action="store_true")
    parser.add_argument("--max-roots", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-emission-batch-publisher.json")
    parser.add_argument("--min-accepted-count", type=int, default=3)
    parser.add_argument("--min-committee-count", type=int, default=3)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--command-timeout", type=float, default=300.0)
    parser.add_argument("--rpc-url", default="")
    parser.add_argument("--broadcast", action="store_true")
    parser.add_argument("--import-artifact", action="store_true")
    parser.add_argument("--sync-lane", action="store_true")
    parser.add_argument("--sync-lane-broadcast", action="store_true")
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    while True:
        result = run_once(args)
        _write_json(args.report_out, result)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"{result['status']}: {result.get('message') or result.get('lane_id')}")
        if args.once:
            return 0 if result.get("ok") else 1
        time.sleep(max(1, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())

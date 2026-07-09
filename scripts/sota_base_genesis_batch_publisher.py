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
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = Path("/home/mekaneeky/repos/.sota-base-testnet")
DEFAULT_SNAPSHOT_DIR = Path("/mnt/4tb/tao_fork_snapshot")
DEFAULT_MANIFEST = DEFAULT_ARTIFACTS_DIR / "base-sepolia-deployment-manifest.json"
DEFAULT_CLAIMS_API_URL = "https://gs4g5jntcn.eu-central-1.awsapprunner.com"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
) -> dict[str, Any]:
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
    if not isinstance(loaded, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return loaded


def _binding_hash(binding: dict[str, Any]) -> str:
    value = str(binding.get("binding_hash") or binding.get("bindingHash") or "").strip()
    if not value:
        raise RuntimeError("accepted binding is missing binding_hash")
    return value


def fetch_unincluded_bindings(args: argparse.Namespace, *, admin_token: str) -> list[dict[str, Any]]:
    payload = _request_json(
        "GET",
        _url(
            args.claims_api_url,
            "/api/v1/base/genesis/bindings",
            {"status": "accepted", "included": "false"},
        ),
        admin_token=admin_token,
        timeout=args.timeout,
    )
    bindings = payload.get("bindings")
    if not isinstance(bindings, list):
        raise RuntimeError("claims API did not return a bindings list")
    clean = [dict(item) for item in bindings if isinstance(item, dict) and not bool(dict(item).get("included"))]
    return clean[: int(args.batch_size)]


def _run(cmd: list[str], *, cwd: Path, timeout: float) -> dict[str, Any]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"command exited {result.returncode}: {' '.join(cmd)}")
    stdout = result.stdout.strip()
    if not stdout:
        return {}
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"stdout": stdout}


def _write_binding_files(bindings: list[dict[str, Any]], *, batch_dir: Path) -> list[Path]:
    binding_dir = batch_dir / "bindings"
    binding_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, binding in enumerate(bindings):
        path = binding_dir / f"binding-{index:04d}.json"
        _write_json(path, binding)
        paths.append(path)
    return paths


def _root_id_from_publish_result(path: Path) -> str:
    payload = _load_json(path)
    root_id = payload.get("root_id")
    if not root_id and isinstance(payload.get("root_published_event"), dict):
        root_id = dict(payload["root_published_event"]).get("root_id")
    if not root_id:
        raise RuntimeError(f"publish result {path} is missing root_id")
    return str(root_id)


def publish_batch(args: argparse.Namespace, bindings: list[dict[str, Any]], *, admin_token: str) -> dict[str, Any]:
    if len(bindings) < int(args.min_bindings):
        return {
            "schema": "sota-base-genesis-batch-publisher/v1",
            "generated_at": _utc_now(),
            "status": "idle",
            "ok": True,
            "binding_count": len(bindings),
            "message": f"waiting for at least {args.min_bindings} unbatched accepted binding(s)",
        }
    binding_hashes = [_binding_hash(binding) for binding in bindings]
    batch_seed = "|".join(binding_hashes)
    batch_id = f"genesis-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{abs(hash(batch_seed)) % 10_000_000:07d}"
    batch_dir = args.out_dir / "genesis-batches" / batch_id
    binding_paths = _write_binding_files(bindings, batch_dir=batch_dir)

    build_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sota_snapshot_claim_bridge.py"),
        "build",
        "--snapshot-dir",
        str(args.snapshot_dir),
        "--manifest",
        str(args.manifest),
        "--out-dir",
        str(batch_dir),
    ]
    for path in binding_paths:
        build_cmd.extend(["--binding", str(path)])
    if args.allow_local:
        build_cmd.append("--allow-local")
    build = _run(build_cmd, cwd=REPO_ROOT, timeout=args.command_timeout)

    root_artifact = batch_dir / "sota-snapshot-genesis-root-artifact.json"
    claim_template = batch_dir / "sota-snapshot-genesis-claim-template.json"
    publish_result = batch_dir / "sota-snapshot-genesis-root-publish-result.json"
    claim_artifact = batch_dir / "sota-snapshot-genesis-claim-artifact.json"

    publish_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sota_base_publish_root.py"),
        "--manifest",
        str(args.manifest),
        "--root-artifact",
        str(root_artifact),
        "--kind",
        "genesis",
        "--out",
        str(publish_result),
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

    result: dict[str, Any] = {
        "schema": "sota-base-genesis-batch-publisher/v1",
        "generated_at": _utc_now(),
        "status": "dry_run" if not args.broadcast else "published",
        "ok": True,
        "batch_id": batch_id,
        "binding_count": len(bindings),
        "binding_hashes": binding_hashes,
        "artifacts": {
            "batch_dir": str(batch_dir),
            "root_artifact": str(root_artifact),
            "claim_template": str(claim_template),
            "publish_result": str(publish_result),
            "claim_artifact": str(claim_artifact),
        },
        "build": build,
        "publish": publish,
        "does_not": ["touch_production_bittensor", "touch_base_mainnet", "custody_user_keys"],
    }
    if not args.broadcast:
        result["message"] = "dry run built a batch root publish request; no transaction broadcast, import, or inclusion mark performed"
        _write_json(args.report_out, result)
        return result

    finalize_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sota_snapshot_claim_bridge.py"),
        "finalize",
        "--claim-template",
        str(claim_template),
        "--publish-result",
        str(publish_result),
        "--out",
        str(claim_artifact),
    ]
    finalize = _run(finalize_cmd, cwd=REPO_ROOT, timeout=args.command_timeout)
    root_id = _root_id_from_publish_result(publish_result)
    result["root_id"] = root_id
    result["finalize"] = finalize

    if args.import_artifact:
        artifact = _load_json(claim_artifact)
        imported = _request_json(
            "POST",
            _url(args.claims_api_url, "/api/v1/base/index/artifact"),
            admin_token=admin_token,
            payload=artifact,
            timeout=args.timeout,
        )
        result["indexer_import"] = imported

    if args.mark_included:
        marked = _request_json(
            "POST",
            _url(args.claims_api_url, "/api/v1/base/genesis/bindings/included"),
            admin_token=admin_token,
            payload={"binding_hashes": binding_hashes, "root_id": root_id, "batch_id": batch_id},
            timeout=args.timeout,
        )
        result["bindings_included"] = marked
    _write_json(args.report_out, result)
    return result


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    admin_token = _admin_token(args.admin_token_env)
    if not admin_token:
        raise RuntimeError(f"{args.admin_token_env} is required")
    bindings = fetch_unincluded_bindings(args, admin_token=admin_token)
    return publish_batch(args, bindings, admin_token=admin_token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch accepted snapshot coldkey bindings into Base SOTA genesis roots.")
    parser.add_argument("--claims-api-url", default=os.environ.get("SOTA_CLAIMS_API_URL", DEFAULT_CLAIMS_API_URL))
    parser.add_argument("--admin-token-env", default="SOTA_BASE_INDEXER_ADMIN_TOKEN")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_ARTIFACTS_DIR / "base-sota-genesis-batch-publisher.json")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--min-bindings", type=int, default=1)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--command-timeout", type=float, default=300.0)
    parser.add_argument("--rpc-url", default="")
    parser.add_argument("--broadcast", action="store_true")
    parser.add_argument("--import-artifact", action="store_true")
    parser.add_argument("--mark-included", action="store_true")
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
            print(f"{result['status']}: {result.get('message') or result.get('batch_id')}")
        if args.once:
            return 0 if result.get("ok") else 1
        time.sleep(max(1, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())

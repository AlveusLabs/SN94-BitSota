from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_genesis_batch_publisher.py"


def _module():
    spec = importlib.util.spec_from_file_location("sota_base_genesis_batch_publisher", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        claims_api_url="https://claims.example.invalid",
        admin_token_env="SOTA_BASE_INDEXER_ADMIN_TOKEN",
        snapshot_dir=tmp_path / "snapshot",
        manifest=tmp_path / "manifest.json",
        out_dir=tmp_path / "out",
        report_out=tmp_path / "out" / "report.json",
        batch_size=500,
        min_bindings=1,
        interval_seconds=600,
        timeout=30.0,
        command_timeout=300.0,
        rpc_url="",
        broadcast=False,
        import_artifact=False,
        mark_included=False,
        allow_local=False,
        once=True,
        json=True,
    )


def _binding(index: int = 1) -> dict:
    return {
        "schema": "sota-snapshot-signed-binding/v1",
        "status": "accepted",
        "included": False,
        "binding_hash": "0x" + f"{index:064x}",
        "signature": "0x" + "11" * 64,
        "message": {
            "snapshot_id": "tao-finney-8549811",
            "claim_id": f"claim-{index}",
            "coldkey": f"coldkey-{index}",
            "reward_address": "0x0000000000000000000000000000000000000A11",
            "allocation_amount": "1",
        },
    }


def test_fetch_unincluded_bindings_filters_included_and_caps(monkeypatch, tmp_path) -> None:
    module = _module()
    args = _args(tmp_path)
    args.batch_size = 1
    seen = {}

    def fake_request(method, url, *, admin_token="", payload=None, timeout=0):
        seen["method"] = method
        seen["url"] = url
        seen["token"] = admin_token
        return {"bindings": [_binding(1), {**_binding(2), "included": True}]}

    monkeypatch.setattr(module, "_request_json", fake_request)
    bindings = module.fetch_unincluded_bindings(args, admin_token="secret")

    assert len(bindings) == 1
    assert bindings[0]["binding_hash"] == _binding(1)["binding_hash"]
    assert "included=false" in seen["url"]
    assert seen["token"] == "secret"


def test_publish_batch_idle_when_below_min_bindings(tmp_path) -> None:
    module = _module()
    args = _args(tmp_path)
    args.min_bindings = 2

    result = module.publish_batch(args, [_binding(1)], admin_token="secret")

    assert result["status"] == "idle"
    assert result["ok"] is True
    assert result["binding_count"] == 1


def test_publish_batch_dry_run_writes_single_batch_for_multiple_bindings(monkeypatch, tmp_path) -> None:
    module = _module()
    args = _args(tmp_path)
    calls = []

    def fake_run(cmd, *, cwd, timeout):
        calls.append(cmd)
        return {"ok": True, "cmd": Path(cmd[1]).name}

    monkeypatch.setattr(module, "_run", fake_run)
    result = module.publish_batch(args, [_binding(1), _binding(2)], admin_token="secret")

    assert result["status"] == "dry_run"
    assert result["binding_count"] == 2
    assert len(calls) == 2
    build_call = " ".join(calls[0])
    assert build_call.count("--binding") == 2
    assert (Path(result["artifacts"]["batch_dir"]) / "bindings" / "binding-0000.json").exists()
    assert (Path(result["artifacts"]["batch_dir"]) / "bindings" / "binding-0001.json").exists()

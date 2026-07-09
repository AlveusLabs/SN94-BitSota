from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_emission_batch_publisher.py"


def _module():
    spec = importlib.util.spec_from_file_location("sota_base_emission_batch_publisher", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        coordinator_url="https://coordinator.example.invalid",
        claims_api_url="https://claims.example.invalid",
        admin_token_env="SOTA_BASE_INDEXER_ADMIN_TOKEN",
        lane_id="base:sota-local",
        epoch=None,
        include_backlog=False,
        oldest_first=False,
        max_roots=1,
        manifest=tmp_path / "manifest.json",
        out_dir=tmp_path / "out",
        report_out=tmp_path / "out" / "report.json",
        min_accepted_count=3,
        min_committee_count=3,
        interval_seconds=600,
        timeout=30.0,
        command_timeout=300.0,
        rpc_url="",
        broadcast=False,
        import_artifact=False,
        sync_lane=False,
        sync_lane_broadcast=False,
        allow_local=False,
        once=True,
        json=True,
    )


def _root(epoch: int = 6, root: str | None = None) -> dict:
    return {
        "id": f"root-{epoch}",
        "subnet_id": "base:sota-local",
        "epoch": epoch,
        "root": root or "0x" + f"{epoch:064x}",
        "total_amount_units": 7200 * 10**18,
        "ready_for_attestation": True,
    }


def test_fetch_coordinator_roots_defaults_to_latest_ready(monkeypatch, tmp_path) -> None:
    module = _module()
    args = _args(tmp_path)

    def fake_request(method, url, *, admin_token="", payload=None, timeout=0):
        assert method == "GET"
        assert "emission-roots" in url
        return [_root(2), {**_root(8), "ready_for_attestation": False}, _root(6)]

    monkeypatch.setattr(module, "_request_json", fake_request)

    roots = module.fetch_coordinator_roots(args)

    assert [root["epoch"] for root in roots] == [6]


def test_run_once_skips_latest_when_indexed(monkeypatch, tmp_path) -> None:
    module = _module()
    args = _args(tmp_path)
    latest = _root(6)

    monkeypatch.setattr(module, "import_local_finalized_artifacts", lambda args, admin_token: [])
    monkeypatch.setattr(module, "fetch_indexed_roots", lambda args: {latest["root"].lower()})
    monkeypatch.setattr(module, "fetch_coordinator_roots", lambda args: [latest])

    result = module.run_once(args)

    assert result["status"] == "idle"
    assert result["skipped"][0]["reason"] == "already_indexed"
    assert result["published"] == []


def test_run_once_publishes_unindexed_latest(monkeypatch, tmp_path) -> None:
    module = _module()
    args = _args(tmp_path)
    latest = _root(7)
    published = {"ok": True, "status": "dry_run", "epoch": 7}

    monkeypatch.setattr(module, "import_local_finalized_artifacts", lambda args, admin_token: [])
    monkeypatch.setattr(module, "fetch_indexed_roots", lambda args: set())
    monkeypatch.setattr(module, "fetch_coordinator_roots", lambda args: [latest])
    monkeypatch.setattr(module, "publish_root", lambda args, root, admin_token: published)

    result = module.run_once(args)

    assert result["status"] == "dry_run"
    assert result["published"] == [published]
    assert result["skipped"] == []

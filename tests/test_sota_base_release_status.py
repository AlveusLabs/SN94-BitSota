from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_release_status.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_release_status", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_report(
    path: Path,
    *,
    schema: str,
    ok: bool,
    status: str = "green",
    checks: list[dict] | None = None,
    next_actions: list[str] | None = None,
) -> None:
    if checks is None and schema == "sota-base-testnet-browser-smoke/v1" and ok:
        checks = [
            {"name": "claims_binding_frontend", "status": "green"},
            {"name": "claims_page_text", "status": "green"},
            {"name": "genesis_binding_message", "status": "green"},
            {"name": "genesis_binding_submit_route", "status": "green"},
            {"name": "genesis_lookup", "status": "green"},
            {"name": "emission_lookup", "status": "green"},
            {"name": "genesis_calldata", "status": "green"},
            {"name": "emission_calldata", "status": "green"},
            {"name": "self_validation_evidence", "status": "green"},
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "ok": ok,
                "status": status,
                "summary": {"green": 1 if ok else 0, "yellow": 0, "red": 0 if ok else 1},
                "message": "ok" if ok else "blocked",
                "checks": checks or [],
                "next_actions": next_actions or [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _wallet_check(status: str = "green") -> dict:
    return {
        "name": "tester_wallet_rpc",
        "status": status,
        "detail": (
            "tester wallet RPC is browser-safe"
            if status == "green"
            else "tester wallet RPC may be rejected by MetaMask from another computer"
        ),
        "remediation": "" if status == "green" else "Relaunch with --share-mode tailscale-https.",
    }


def _write_seed_report(path: Path, *, wallet: str, genesis: str = "150", emission: str = "200") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "sota-base-testnet-seed-artifacts-finalized/v1",
                "ok": True,
                "status": "ready_to_import_claim_artifacts",
                "seeded_claims": {
                    "test_wallet_address": wallet,
                    "genesis_total_units": genesis,
                    "emission_total_units": emission,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_claim_evidence(path: Path, *, wallet: str, genesis: str = "150", emission: str = "200") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "sota-base-claim-tx-evidence/v1",
                "ok": True,
                "status": "green",
                "summary": {"green": 28, "yellow": 0, "red": 0},
                "message": "Claim transaction evidence verifies both genesis and emission SOTA claims.",
                "config": {"wallet_address": wallet},
                "transactions": {
                    "genesis": {"from": wallet, "claim_amount_raw": genesis},
                    "emission": {"from": wallet, "claim_amount_raw": emission},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_lane_sync(path: Path, *, ok: bool = True, status: str = "green") -> None:
    _write_report(path, schema="sota-base-lane-sync/v1", ok=ok, status=status)


def _write_miner_swarm(path: Path, *, count: int = 5, ok: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "sota-local-multi-miner/v1",
                "ok": ok,
                "miner_count": count,
                "accepted_count": count,
                "matching_claim_count": count,
                "miners": [
                    {
                        "name": f"miner-{index}",
                        "hotkey": f"hotkey-{index}",
                        "miner_address": f"0x{index:040x}",
                        "reward_address": f"0x{index + 100:040x}",
                    }
                    for index in range(1, count + 1)
                ],
                "claim_transactions": [
                    {
                        "reward_address": f"0x{index + 100:040x}",
                        "tx_hash": "0x" + f"{index:064x}",
                        "amount_units": "1",
                    }
                    for index in range(1, count + 1)
                ],
                "checks": {
                    "distinct_hotkeys": True,
                    "distinct_miner_addresses": True,
                    "distinct_reward_addresses": True,
                    "all_processes_exited_zero": True,
                    "all_self_validation_accepted": True,
                    "all_claims_submitted": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_snapshot_source(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "genesis_snapshot_block.json").write_text(
        json.dumps(
            {
                "schema": "sota-genesis-snapshot-block-lock/v1",
                "bittensor_block_number": 8549811,
                "bittensor_block_hash": "0x" + "ab" * 32,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "alpha_exposures.csv").write_text(
        "coldkey,netuid,included,total_alpha_units,tao_in_pool_rao\n"
        "5Alice,1,True,100,200\n",
        encoding="utf-8",
    )


def _write_snapshot_genesis_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "sota-base-claim-artifact/v1",
                "indexer_import_ready": True,
                "snapshot": {
                    "snapshot_id": "sota-genesis-8549811-test",
                    "bittensor_block_number": 8549811,
                    "bittensor_block_hash": "0x" + "ab" * 32,
                },
                "root": {
                    "root_id": "0x" + "12" * 32,
                    "subnet_id": "genesis",
                    "status": "finalized",
                    "validation_status": "accepted",
                    "total_amount_units": "350",
                },
                "allocations": [
                    {
                        "kind": "genesis",
                        "reward_address": "0x1111111111111111111111111111111111111111",
                        "amount_units": "350",
                        "tao_credit_rao": "100",
                        "alpha_synthetic_credit_rao": "250",
                        "alpha_credit_rao_by_netuid": {"1": "250"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_pending_binding_request(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "sota-snapshot-binding-message/v1",
                "status": "message_ready",
                "message": {"coldkey": "5Alice", "reward_address": "0x1111111111111111111111111111111111111111"},
                "signing_payload": "{}",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_publisher_report(path: Path, *, schema: str, status: str = "idle", ok: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "ok": ok,
                "status": status,
                "message": "ready",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fresh_emission_tester(path: Path, *, ok: bool = True, status: str = "green") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "sota-base-fresh-emission-tester/v1",
                "ok": ok,
                "status": status,
                "reward_address": "0x3333333333333333333333333333333333333333",
                "epoch": 12,
                "eligibility": {"eligible": ok, "claim_state": {"status": "claimable" if ok else "not_claimable"}},
                "claim_transaction": {"ok": ok, "chain_id": "84532", "to": "0x4444444444444444444444444444444444444444"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _args(tmp_path: Path, *, local_only: bool = False):
    return argparse.Namespace(
        local_report=tmp_path / "local" / "report.json",
        local_claim_proof=tmp_path / "local" / "claim-proof.json",
        local_miner_swarm=tmp_path / "local" / "miner-swarm.json",
        min_local_miners=3,
        local_tailscale_preflight=tmp_path / "local" / "tailscale-preflight.json",
        testnet_artifacts_dir=tmp_path / "testnet",
        snapshot_dir=tmp_path / "snapshot",
        snapshot_claim_bindings_url="",
        indexer_admin_token_env="SOTA_BASE_INDEXER_ADMIN_TOKEN",
        timeout=0.1,
        timer_check_timeout=0.1,
        check_publisher_timers=False,
        local_only=local_only,
        defer_real_holder_test=False,
    )


def _mock_publisher_systemctl(module, monkeypatch, *, emission_timer_active: bool = True) -> None:
    def fake_run(cmd, check, text, capture_output, timeout):
        assert cmd[:3] == ["systemctl", "--user", "show"]
        unit = cmd[3]
        if unit.endswith(".timer"):
            active_state = "active"
            if unit == "base-sota-emission-publisher.timer" and not emission_timer_active:
                active_state = "inactive"
            stdout = f"ActiveState={active_state}\nSubState=waiting\nUnitFileState=enabled\n"
        else:
            stdout = "ActiveState=inactive\nSubState=dead\nResult=success\nExecMainStatus=0\n"
        return type("Result", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

    monkeypatch.setattr(module.subprocess, "run", fake_run)


def test_release_status_local_only_green(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)

    report = module.run_status(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["local_stack_ok"] is True
    assert report["local_ok"] is True
    assert report["local_wallet_ok"] is True
    assert report["local_wallet"]["status"] == "green"
    assert report["local_remote_wallet_ok"] is False
    assert report["testnet_ok"] is None
    assert report["blocked_gates"] == []
    assert [gate["name"] for gate in report["gates"]] == ["local_demo", "local_claim_proof", "local_miner_swarm", "local_wallet"]


def test_release_status_local_only_requires_claim_proof(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True)

    report = module.run_status(args)

    assert report["ok"] is False
    assert report["status"] == "red"
    assert report["local_stack_ok"] is False
    assert report["local_ok"] is False
    assert {gate["name"] for gate in report["blocked_gates"]} == {"local_claim_proof", "local_miner_swarm", "local_wallet"}


def test_release_status_reports_local_wallet_readiness_from_local_smoke(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(
        args.local_report,
        schema="sota-local-claims-ui-smoke/v1",
        ok=True,
        status="green",
        checks=[
            {
                **_wallet_check("yellow"),
            }
        ],
    )
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)

    report = module.run_status(args)

    assert report["ok"] is False
    assert report["status"] == "yellow"
    assert report["local_stack_ok"] is True
    assert report["local_ok"] is False
    assert report["local_wallet_ok"] is False
    assert report["local_wallet"] == {
        "ok": False,
        "status": "yellow",
        "message": "tester wallet RPC may be rejected by MetaMask from another computer",
        "next_action": "Relaunch with --share-mode tailscale-https.",
    }
    assert report["local_remote_wallet_ok"] is False
    assert [gate["name"] for gate in report["blocked_gates"]] == ["local_wallet"]


def test_release_status_remote_wallet_comes_from_tailscale_preflight(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(args.local_tailscale_preflight, schema="sota-local-tailscale-preflight/v1", ok=True)

    report = module.run_status(args)

    assert report["ok"] is True
    assert report["local_ok"] is True
    assert report["local_wallet_ok"] is True
    assert report["local_remote_wallet_ok"] is True
    assert report["local_remote_wallet"]["status"] == "green"


def test_release_status_full_requires_all_testnet_gates(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(
        args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json",
        schema="sota-base-testnet-operator-run/v1",
        ok=False,
        status="red",
        next_actions=["Submit/export a signed snapshot coldkey binding."],
    )
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=False, status="red")
    _write_report(args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json", schema="sota-base-claim-tx-evidence/v1", ok=False, status="red")

    report = module.run_status(args)

    assert report["ok"] is False
    assert report["status"] == "red"
    assert report["local_stack_ok"] is True
    assert report["local_ok"] is True
    assert report["local_wallet_ok"] is True
    assert report["testnet_ok"] is False
    assert {gate["name"] for gate in report["blocked_gates"]} == {
        "testnet_operator_run",
        "testnet_snapshot_genesis",
        "testnet_blockers",
        "testnet_emission_lane_sync",
        "testnet_aws_inventory",
        "testnet_funding",
        "testnet_secret_handles",
        "testnet_apprunner_source_pack",
        "testnet_browser_smoke",
        "testnet_fresh_emission_tester",
        "claim_tx_evidence",
    }
    operator_gate = next(gate for gate in report["blocked_gates"] if gate["name"] == "testnet_operator_run")
    assert operator_gate["next_action"] == "Submit/export a signed snapshot coldkey binding."


def test_release_status_full_green_requires_operator_gate(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json", schema="sota-base-testnet-operator-run/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=True)
    _write_lane_sync(args.testnet_artifacts_dir / "base-sota-testnet-emission-lane-sync.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=True)
    _write_fresh_emission_tester(args.testnet_artifacts_dir / "base-sota-fresh-emission-tester.json")
    _write_snapshot_source(args.snapshot_dir)
    _write_snapshot_genesis_artifact(args.testnet_artifacts_dir / "base-sota-testnet-genesis-claim-artifact.json")
    _write_claim_evidence(
        args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json",
        wallet="0x1111111111111111111111111111111111111111",
        genesis="350",
        emission="200",
    )

    report = module.run_status(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["testnet_ok"] is True
    assert [gate["name"] for gate in report["gates"]] == [
        "local_demo",
        "local_claim_proof",
        "local_miner_swarm",
        "local_wallet",
        "testnet_operator_run",
        "testnet_snapshot_genesis",
        "testnet_blockers",
        "testnet_emission_lane_sync",
        "testnet_aws_inventory",
        "testnet_funding",
        "testnet_secret_handles",
        "testnet_apprunner_source_pack",
        "testnet_container_pack",
        "testnet_browser_smoke",
        "testnet_fresh_emission_tester",
        "claim_tx_evidence",
    ]


def test_release_status_checks_publisher_timers_when_enabled(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    _mock_publisher_systemctl(module, monkeypatch)
    args = _args(tmp_path)
    args.check_publisher_timers = True
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json", schema="sota-base-testnet-operator-run/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=True)
    _write_lane_sync(args.testnet_artifacts_dir / "base-sota-testnet-emission-lane-sync.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=True)
    _write_fresh_emission_tester(args.testnet_artifacts_dir / "base-sota-fresh-emission-tester.json")
    _write_snapshot_source(args.snapshot_dir)
    _write_snapshot_genesis_artifact(args.testnet_artifacts_dir / "base-sota-testnet-genesis-claim-artifact.json")
    _write_claim_evidence(
        args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json",
        wallet="0x1111111111111111111111111111111111111111",
        genesis="350",
        emission="200",
    )

    report = module.run_status(args)
    timer_gate = next(gate for gate in report["gates"] if gate["name"] == "testnet_publisher_timers")

    assert report["ok"] is True
    assert timer_gate["status"] == "green"
    assert timer_gate["timers"]["genesis"]["timer"]["ActiveState"] == "active"
    assert [gate["name"] for gate in report["gates"]] == [
        "local_demo",
        "local_claim_proof",
        "local_miner_swarm",
        "local_wallet",
        "testnet_operator_run",
        "testnet_snapshot_genesis",
        "testnet_publisher_timers",
        "testnet_blockers",
        "testnet_emission_lane_sync",
        "testnet_aws_inventory",
        "testnet_funding",
        "testnet_secret_handles",
        "testnet_apprunner_source_pack",
        "testnet_container_pack",
        "testnet_browser_smoke",
        "testnet_fresh_emission_tester",
        "claim_tx_evidence",
    ]


def test_release_status_blocks_when_publisher_timer_is_inactive(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    _mock_publisher_systemctl(module, monkeypatch, emission_timer_active=False)
    gate = module._publisher_timers_gate(timeout=0.1)

    assert gate["ok"] is False
    assert gate["status"] == "red"
    assert "emission" in gate["message"]
    assert "ActiveState=inactive" in gate["message"]


def test_release_status_can_defer_real_holder_test_when_binding_path_is_ready(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    args.defer_real_holder_test = True
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(
        args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json",
        schema="sota-base-testnet-operator-run/v1",
        ok=False,
        status="red",
        checks=[
            {"name": "snapshot_binding_export", "status": "red"},
            {"name": "snapshot_genesis_artifacts", "status": "red"},
            {"name": "publish_genesis_root", "status": "red"},
            {"name": "browser_smoke", "status": "red"},
            {"name": "finalize_claim_artifacts", "status": "yellow"},
            {"name": "import_claim_artifacts", "status": "yellow"},
        ],
    )
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=True)
    _write_lane_sync(args.testnet_artifacts_dir / "base-sota-testnet-emission-lane-sync.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=True)
    _write_fresh_emission_tester(args.testnet_artifacts_dir / "base-sota-fresh-emission-tester.json")
    _write_publisher_report(
        args.testnet_artifacts_dir / "base-sota-genesis-batch-publisher.json",
        schema="sota-base-genesis-batch-publisher/v1",
    )
    _write_publisher_report(
        args.testnet_artifacts_dir / "base-sota-emission-batch-publisher.json",
        schema="sota-base-emission-batch-publisher/v1",
    )
    _write_snapshot_source(args.snapshot_dir)
    _write_pending_binding_request(args.testnet_artifacts_dir / "snapshot-holder-binding-request.json")
    _write_claim_evidence(
        args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json",
        wallet="0x1111111111111111111111111111111111111111",
        genesis="150",
        emission="200",
    )

    report = module.run_status(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["testnet_ok"] is True
    assert report["real_holder_test_deferred"] is True
    assert report["blocked_gates"] == []
    operator_gate = next(gate for gate in report["gates"] if gate["name"] == "testnet_operator_run")
    snapshot_gate = next(gate for gate in report["gates"] if gate["name"] == "testnet_snapshot_genesis")
    assert operator_gate["holder_test_deferred"] is True
    assert snapshot_gate["holder_test_deferred"] is True


def test_release_status_rejects_stale_browser_smoke_without_binding_checks(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json", schema="sota-base-testnet-operator-run/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=True)
    _write_lane_sync(args.testnet_artifacts_dir / "base-sota-testnet-emission-lane-sync.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(
        args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json",
        schema="sota-base-testnet-browser-smoke/v1",
        ok=True,
        checks=[{"name": "claims_page_text", "status": "green"}],
    )
    _write_fresh_emission_tester(args.testnet_artifacts_dir / "base-sota-fresh-emission-tester.json")
    _write_snapshot_source(args.snapshot_dir)
    _write_snapshot_genesis_artifact(args.testnet_artifacts_dir / "base-sota-testnet-genesis-claim-artifact.json")
    _write_claim_evidence(
        args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json",
        wallet="0x1111111111111111111111111111111111111111",
        genesis="350",
        emission="200",
    )

    report = module.run_status(args)
    gate = next(gate for gate in report["gates"] if gate["name"] == "testnet_browser_smoke")

    assert report["ok"] is False
    assert gate["status"] == "red"
    assert "genesis_binding_message" in gate["message"]
    assert [gate["name"] for gate in report["blocked_gates"]] == ["testnet_browser_smoke"]


def test_release_status_marks_stale_claim_tx_evidence_red(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json", schema="sota-base-testnet-operator-run/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=True)
    _write_lane_sync(args.testnet_artifacts_dir / "base-sota-testnet-emission-lane-sync.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=True)
    _write_fresh_emission_tester(args.testnet_artifacts_dir / "base-sota-fresh-emission-tester.json")
    _write_snapshot_source(args.snapshot_dir)
    _write_snapshot_genesis_artifact(args.testnet_artifacts_dir / "base-sota-testnet-genesis-claim-artifact.json")
    _write_seed_report(
        args.testnet_artifacts_dir / "base-sota-testnet-seed-artifacts-finalized.json",
        wallet="0x2222222222222222222222222222222222222222",
    )
    _write_claim_evidence(
        args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json",
        wallet="0x1111111111111111111111111111111111111111",
    )

    report = module.run_status(args)
    claim_gate = next(gate for gate in report["gates"] if gate["name"] == "claim_tx_evidence")

    assert report["ok"] is False
    assert report["testnet_ok"] is False
    assert claim_gate["status"] == "red"
    assert "current finalized genesis artifact" in claim_gate["message"]
    assert [gate["name"] for gate in report["blocked_gates"]] == ["claim_tx_evidence"]


def test_release_status_claim_evidence_uses_artifact_wallet_over_seed_wallet(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json", schema="sota-base-testnet-operator-run/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=True)
    _write_lane_sync(args.testnet_artifacts_dir / "base-sota-testnet-emission-lane-sync.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=True)
    _write_fresh_emission_tester(args.testnet_artifacts_dir / "base-sota-fresh-emission-tester.json")
    _write_snapshot_source(args.snapshot_dir)
    _write_snapshot_genesis_artifact(args.testnet_artifacts_dir / "base-sota-testnet-genesis-claim-artifact.json")
    emission_artifact = {
        "schema": "sota-base-claim-artifact/v1",
        "root": {"root_id": "0x" + "13" * 32, "subnet_id": "base:sota-local", "total_amount_units": "200"},
    }
    (args.testnet_artifacts_dir / "base-sota-testnet-emission-claim-artifact.json").write_text(
        json.dumps(emission_artifact) + "\n",
        encoding="utf-8",
    )
    _write_seed_report(
        args.testnet_artifacts_dir / "base-sota-testnet-seed-artifacts-finalized.json",
        wallet="0x2222222222222222222222222222222222222222",
        genesis="999",
        emission="999",
    )
    _write_claim_evidence(
        args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json",
        wallet="0x1111111111111111111111111111111111111111",
        genesis="350",
        emission="200",
    )

    report = module.run_status(args)
    claim_gate = next(gate for gate in report["gates"] if gate["name"] == "claim_tx_evidence")

    assert report["ok"] is True
    assert claim_gate["status"] == "green"


def test_release_status_rejects_seeded_genesis_without_snapshot_alpha(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json", schema="sota-base-testnet-operator-run/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=True)
    _write_lane_sync(args.testnet_artifacts_dir / "base-sota-testnet-emission-lane-sync.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=True)
    _write_fresh_emission_tester(args.testnet_artifacts_dir / "base-sota-fresh-emission-tester.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json", schema="sota-base-claim-tx-evidence/v1", ok=True)
    _write_snapshot_source(args.snapshot_dir)
    seeded = {
        "schema": "sota-base-claim-artifact/v1",
        "root": {"root_id": "0x" + "12" * 32, "subnet_id": "genesis", "total_amount_units": "150"},
        "allocations": [
            {
                "reward_address": "0x1111111111111111111111111111111111111111",
                "amount_units": "150",
                "tao_credit": "100",
                "alpha_synthetic_credit": "50",
            }
        ],
    }
    path = args.testnet_artifacts_dir / "base-sota-testnet-genesis-claim-artifact.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seeded) + "\n", encoding="utf-8")
    _write_pending_binding_request(args.testnet_artifacts_dir / "snapshot-holder-binding-request.json")

    report = module.run_status(args)
    gate = next(gate for gate in report["gates"] if gate["name"] == "testnet_snapshot_genesis")

    assert report["ok"] is False
    assert gate["status"] == "red"
    assert gate["message"].startswith("accepted signed snapshot binding count is 0")
    assert "snapshot metadata is missing" in gate["message"]
    assert "TAO/alpha rao credit fields" in gate["message"]
    assert "accepted signed snapshot binding count is 0" in gate["message"]
    assert "pending unsigned binding request exists" in gate["message"]
    assert gate["snapshot_binding_evidence"]["accepted_signed_binding_count"] == 0
    assert gate["snapshot_binding_evidence"]["pending_unsigned_binding_request_count"] == 1
    assert gate["snapshot_binding_evidence"]["public_binding_export"]["status"] == "not_configured"


def test_release_status_reports_public_binding_export_count(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    args.snapshot_claim_bindings_url = "https://claims-api.example.invalid/api/v1/base/genesis/bindings"
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-operator-run.json", schema="sota-base-testnet-operator-run/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-blockers.json", schema="sota-base-testnet-blockers/v1", ok=True)
    _write_lane_sync(args.testnet_artifacts_dir / "base-sota-testnet-emission-lane-sync.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-aws-inventory.json", schema="sota-base-testnet-aws-inventory/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-funding.json", schema="sota-base-testnet-funding/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-secret-handles.json", schema="sota-base-testnet-secret-bootstrap/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-apprunner-source-pack.json", schema="sota-base-testnet-apprunner-source-pack/v1", ok=True)
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-container-pack.json", schema="sota-base-testnet-container-pack/v1", ok=False, status="yellow")
    _write_report(args.testnet_artifacts_dir / "base-sota-testnet-browser-smoke.json", schema="sota-base-testnet-browser-smoke/v1", ok=True)
    _write_fresh_emission_tester(args.testnet_artifacts_dir / "base-sota-fresh-emission-tester.json")
    _write_report(args.testnet_artifacts_dir / "base-sota-claim-tx-evidence.json", schema="sota-base-claim-tx-evidence/v1", ok=True)
    _write_snapshot_source(args.snapshot_dir)
    seeded = {
        "schema": "sota-base-claim-artifact/v1",
        "root": {"root_id": "0x" + "12" * 32, "subnet_id": "genesis", "total_amount_units": "150"},
        "allocations": [{"reward_address": "0x1111111111111111111111111111111111111111", "amount_units": "150"}],
    }
    path = args.testnet_artifacts_dir / "base-sota-testnet-genesis-claim-artifact.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seeded) + "\n", encoding="utf-8")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"schema":"sota-snapshot-bindings/v1","count":0,"bindings":[]}'

    monkeypatch.setenv("SOTA_BASE_INDEXER_ADMIN_TOKEN", '{"admin_token":"secret-token"}')
    monkeypatch.setattr(module, "urlopen", lambda request, timeout: FakeResponse())

    report = module.run_status(args)
    gate = next(gate for gate in report["gates"] if gate["name"] == "testnet_snapshot_genesis")
    export = gate["snapshot_binding_evidence"]["public_binding_export"]

    assert report["ok"] is False
    assert export["status"] == "green"
    assert export["accepted_signed_binding_count"] == 0
    assert export["used_auth_header"] is True
    assert "public claims API accepted binding count is 0" in gate["message"]


def test_release_status_rejects_schema_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="wrong-schema", ok=True)
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)

    report = module.run_status(args)
    gate = report["gates"][0]

    assert report["ok"] is False
    assert gate["status"] == "red"
    assert gate["schema"] == "wrong-schema"
    assert gate["expected_schema"] == "sota-local-claims-ui-smoke/v1"


def test_release_status_rejects_too_small_local_miner_swarm(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm, count=1)

    report = module.run_status(args)
    gate = next(gate for gate in report["gates"] if gate["name"] == "local_miner_swarm")

    assert report["ok"] is False
    assert gate["status"] == "red"
    assert "below required" in gate["message"]
    assert [gate["name"] for gate in report["blocked_gates"]] == ["local_miner_swarm"]


def test_release_status_missing_report_is_red(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)

    report = module.run_status(args)
    gates = {gate["name"]: gate for gate in report["gates"]}

    assert report["ok"] is False
    assert report["local_ok"] is False
    assert gates["local_demo"]["status"] == "red"
    assert gates["local_demo"]["message"] == "Report is missing."
    assert gates["local_claim_proof"]["status"] == "red"
    assert gates["local_claim_proof"]["message"] == "Report is missing."
    assert gates["local_miner_swarm"]["status"] == "red"
    assert gates["local_miner_swarm"]["message"] == "Local miner swarm report is missing."


def test_release_status_json_without_report_out_only_prints(tmp_path: Path, capsys) -> None:
    module = _load_module()
    args = _args(tmp_path, local_only=True)
    _write_report(args.local_report, schema="sota-local-claims-ui-smoke/v1", ok=True, checks=[_wallet_check()])
    _write_report(args.local_claim_proof, schema="sota-local-claim-proof/v1", ok=True)
    _write_miner_swarm(args.local_miner_swarm)

    exit_code = module.main(
        [
            "--local-only",
            "--local-report",
            str(args.local_report),
            "--local-claim-proof",
            str(args.local_claim_proof),
            "--local-miner-swarm",
            str(args.local_miner_swarm),
            "--json",
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["local_ok"] is True
    assert printed["local_wallet_ok"] is True
    assert printed["ok"] is True

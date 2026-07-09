from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_prepare_fresh_testnet_emission_tester.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_prepare_fresh_testnet_emission_tester", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        artifacts_dir=tmp_path / "artifacts",
        reward_key_file=tmp_path / "wallet.json",
        evidence_out=tmp_path / "evidence.json",
        seed_report_out=tmp_path / "seed.json",
        publisher_report_out=tmp_path / "publisher.json",
        report_out=tmp_path / "fresh-emission.json",
        deployment=tmp_path / "manifest.json",
        rpc_url="https://sepolia.base.org",
        claims_api_url="https://claims.example.test",
        coordinator_url="https://coordinator.example.test",
        lane_id="base:sota-local",
        aws_profile="test-profile",
        aws_region="eu-central-1",
        indexer_admin_secret_id="indexer-secret",
        root_publisher_secret_id="root-secret",
        deployer_secret_id="deployer-secret",
        sponsor_key_file=tmp_path / "sponsor.json",
        min_wallet_balance_eth="0.005",
        top_up_eth="0.01",
        max_priority_fee_gwei="0.001",
        metric_value=0.8,
        epoch=None,
        reuse_seed_report=False,
        timeout=1.0,
        command_timeout=2.0,
        operator_timeout=3.0,
    )


def _patch_happy_path(module, monkeypatch, args: argparse.Namespace, *, claimable: bool = True) -> list[list[str]]:
    commands: list[list[str]] = []
    reward_address = "0x1111111111111111111111111111111111111111"
    epoch = 12

    def fake_run_command(cmd, *, timeout, cwd=module.DOCS_REPO, env=None):
        commands.append(cmd)
        if any("sota_seed_testnet_autoresearch.py" in str(item) for item in cmd):
            _write_json(
                args.seed_report_out,
                {
                    "schema": "sota-base-testnet-autoresearch-seed/v1",
                    "reward_address": reward_address,
                    "epoch": epoch,
                    "claim_count": 1,
                },
            )
        if any("sota_base_emission_batch_publisher.py" in str(item) for item in cmd):
            assert env["SOTA_BASE_INDEXER_ADMIN_TOKEN"] == "indexer-token"
            assert env["SOTA_ROOT_PUBLISHER_PRIVATE_KEY"].startswith("0x")
            assert env["SOTA_DEPLOYER_PRIVATE_KEY"].startswith("0x")
            _write_json(
                args.publisher_report_out,
                {
                    "schema": "sota-base-emission-batch-publisher/v1",
                    "ok": True,
                    "status": "published",
                    "published": [{"epoch": epoch, "status": "published"}],
                    "skipped": [],
                },
            )
        return {"returncode": 0, "stdout": "{}", "stderr": "", "command": cmd}

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module, "_top_up_if_needed", lambda call_args, reward_key_file: {"status": "funded"})
    monkeypatch.setattr(module, "_load_indexer_admin_token", lambda call_args: "indexer-token")
    monkeypatch.setattr(module, "_secret_private_key", lambda *a, **k: "0x" + "12" * 32)
    monkeypatch.setattr(
        module,
        "_eligibility",
        lambda call_args, reward_address, lane_id: {
            "eligible": claimable,
            "claim_state": {"claimable": claimable},
            "credits": {
                "total_sota": {"raw": "7200000000000000000000" if claimable else "0"},
                "unclaimed_sota": {"raw": "7200000000000000000000" if claimable else "0"},
            },
        },
    )
    monkeypatch.setattr(
        module,
        "_claim_transaction",
        lambda call_args, reward_address, lane_id: {
            "transaction": {
                "to": "0x2222222222222222222222222222222222222222",
                "data": "0x090eb79900",
                "chainId": 84532,
            }
        },
    )
    return commands


def test_prepare_fresh_emission_tester_runs_real_seed_publish_shape(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    commands = _patch_happy_path(module, monkeypatch, args)

    report = module.prepare_emission_tester(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["summary"] == {"green": 1, "yellow": 0, "red": 0}
    assert report["private_key_printed"] is False
    assert report["reward_key_file"] == str(args.reward_key_file)
    assert report["epoch"] == 12
    assert report["claim_transaction"]["ok"] is True
    assert report["does_not"] == [
        "print_private_keys",
        "touch_production_bittensor",
        "touch_base_mainnet",
        "test_real_holder_claims",
    ]
    assert any(any("sota_seed_testnet_autoresearch.py" in str(item) for item in cmd) for cmd in commands)
    publisher_cmd = next(
        cmd for cmd in commands if any("sota_base_emission_batch_publisher.py" in str(item) for item in cmd)
    )
    assert "--broadcast" in publisher_cmd
    assert "--import-artifact" in publisher_cmd
    assert "--sync-lane-broadcast" in publisher_cmd
    assert ["--epoch", "12"] == publisher_cmd[publisher_cmd.index("--epoch") : publisher_cmd.index("--epoch") + 2]
    assert json.loads(args.report_out.read_text(encoding="utf-8"))["ok"] is True


def test_prepare_fresh_emission_tester_marks_unclaimable_red(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _patch_happy_path(module, monkeypatch, args, claimable=False)

    report = module.prepare_emission_tester(args)

    assert report["ok"] is False
    assert report["status"] == "red"
    assert report["eligibility"]["eligible"] is False


def test_secret_private_key_reads_json_field(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    monkeypatch.setattr(module, "_aws_secret_string", lambda *a, **k: '{"deployer_private_key":"' + ("34" * 32) + '"}')

    value = module._secret_private_key(
        "secret",
        fields=("deployer_private_key", "private_key"),
        args=args,
    )

    assert value == "0x" + "34" * 32


def test_prepare_fresh_emission_tester_can_resume_existing_seed(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    args.reuse_seed_report = True
    _write_json(
        args.seed_report_out,
        {
            "schema": "sota-base-testnet-autoresearch-seed/v1",
            "reward_address": "0x1111111111111111111111111111111111111111",
            "epoch": 12,
            "claim_count": 1,
        },
    )
    commands = _patch_happy_path(module, monkeypatch, args)

    report = module.prepare_emission_tester(args)

    assert report["ok"] is True
    assert report["reused_seed_report"] is True
    assert not any(any("sota_seed_testnet_autoresearch.py" in str(item) for item in cmd) for cmd in commands)

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_operator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_operator", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    values = {
        "artifacts_dir": tmp_path / "artifacts",
        "template": REPO / "docs" / "base" / "manifests" / "base-sepolia-deployment-manifest.template.json",
        "rpc_url": "https://sepolia.base.org",
        "aws_profile": "",
        "aws_region": "eu-central-1",
        "deployment": None,
        "deploy": False,
        "private_key_env": "SOTA_DEPLOYER_PRIVATE_KEY",
        "private_key_secret_id": "",
        "private_key_secret_json_key": "",
        "initial_vault_supply_sota": "1000000",
        "owner_address": "",
        "supply_authority_address": "",
        "emission_authority_address": "",
        "root_publisher_address": "",
        "default_lane_id": "base:sota-test",
        "emission_evidence": None,
        "local_state": tmp_path / "local-state.json",
        "test_wallet_address": "0x00000000000000000000000000000000000000aa",
        "test_old_coldkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "test_epoch": "1",
        "min_accepted_count": 3,
        "min_committee_count": 3,
        "claims_ui_url": "https://claims-test.example.invalid",
        "claims_ui_health_url": "",
        "claims_api_url": "https://claims-api-test.example.invalid",
        "claims_api_health_url": "",
        "coordinator_url": "https://coordinator-test.example.invalid",
        "attestation_url": "https://attestation-test.example.invalid",
        "attestation_health_url": "",
        "root_publisher_url": "https://root-test.example.invalid",
        "root_publisher_health_url": "",
        "claim_artifacts_url": "https://claims-test.example.invalid/base-sota-testnet-seed-artifacts-finalized.json",
        "monitoring_url": "https://monitoring-test.example.invalid",
        "monitoring_alert_policy_url": "",
        "monitoring_log_group_or_sink": "",
        "readiness_url": "https://claims-test.example.invalid/base-sota-testnet-readiness.json",
        "build_website": False,
        "broadcast_roots": False,
        "root_publisher_private_key_secret_id": "",
        "root_publisher_private_key_secret_json_key": "",
        "import_artifacts": False,
        "indexer_admin_token_env": "SOTA_INDEXER_ADMIN_TOKEN",
        "skip_browser_smoke": False,
        "timeout": 1.0,
        "command_timeout": 1.0,
        "json": False,
        "report_out": Path(""),
        "allow_blocked": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _command_result(cmd: list[str], returncode: int = 0) -> dict:
    return {"returncode": returncode, "stdout": "{}", "stderr": "", "command": cmd, "command_text": " ".join(cmd)}


def _has_cmd(cmd: list[str], name: str) -> bool:
    return any(name in item for item in cmd)


def _write_local_state(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "accounts": {"alice_reward": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"},
                "genesis": {"old_coldkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"},
                "autoresearch": {
                    "evidence": {
                        "root": {"root": "0x" + "11" * 32},
                        "bundle": {
                            "claim_list": [
                                {
                                    "index": 0,
                                    "reward_address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                                    "amount_units": "2000000000000000000",
                                    "reward_hash": "0x" + "22" * 32,
                                    "offchain_lane_id": "0x" + "33" * 32,
                                    "epoch": 1,
                                }
                            ],
                            "claim_evidence": [
                                {
                                    "index": 0,
                                    "evidence": {
                                        "self_validation_consensus": {
                                            "status": "accepted",
                                            "accepted_count": 3,
                                            "committee_count": 3,
                                        }
                                    },
                                }
                            ],
                        },
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_standard_reports(module, paths: dict[str, Path], cmd: list[str]) -> None:
    if _has_cmd(cmd, "sota_base_testnet_service_pack.py"):
        module._write_json(
            paths["service_pack_json"],
            {"schema": "sota-base-testnet-service-pack/v1", "ok": True, "status": "green"},
        )
    if _has_cmd(cmd, "sota_base_testnet_apprunner_source_pack.py"):
        module._write_json(
            paths["apprunner_source_pack"],
            {"schema": "sota-base-testnet-apprunner-source-pack/v1", "ok": True, "status": "green"},
        )
    if _has_cmd(cmd, "sota_base_testnet_blockers.py"):
        module._write_json(
            paths["blockers"],
            {"schema": "sota-base-testnet-blockers/v1", "ok": True, "status": "green"},
        )
    if _has_cmd(cmd, "sota_base_testnet_aws_inventory.py"):
        module._write_json(
            paths["aws_inventory"],
            {"schema": "sota-base-testnet-aws-inventory/v1", "ok": True, "status": "green"},
        )
    if _has_cmd(cmd, "sota_base_testnet_browser_smoke.py"):
        module._write_json(
            paths["browser_smoke"],
            {"schema": "sota-base-testnet-browser-smoke/v1", "ok": True, "status": "green"},
        )
    if _has_cmd(cmd, "sota_base_release_status.py"):
        module._write_json(
            paths["release_status"],
            {"schema": "sota-base-release-status/v1", "ok": True, "status": "green"},
        )


def test_operator_report_is_red_without_deployment_or_evidence(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    paths = module._paths(args.artifacts_dir)

    def fake_run(cmd: list[str], **kwargs) -> dict:
        _write_standard_reports(module, paths, cmd)
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)
    report = module.run_operator(args)
    steps = {step["name"]: step for step in report["steps"]}

    assert report["schema"] == "sota-base-testnet-operator-run/v1"
    assert report["ok"] is False
    assert report["status"] == "red"
    assert steps["rehearsal"]["status"] == "red"
    assert steps["seed_artifacts"]["status"] == "red"
    assert "touch_production_bittensor" in report["does_not"]
    assert "use_mock_claims" in report["does_not"]


def test_operator_fills_missing_seed_inputs_from_local_state(tmp_path: Path) -> None:
    module = _load_module()
    local_state = tmp_path / "local-state.json"
    _write_local_state(local_state)
    args = _args(
        tmp_path,
        local_state=local_state,
        test_wallet_address="",
        test_old_coldkey="",
        emission_evidence=None,
    )
    paths = module._paths(args.artifacts_dir)

    filled = module._fill_seed_inputs_from_local_state(args, paths)

    assert args.test_wallet_address == "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
    assert args.test_old_coldkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    assert args.emission_evidence == paths["artifacts_dir"] / "base-sota-testnet-emission-evidence-from-local.json"
    assert set(filled) == {"test_wallet_address", "test_old_coldkey", "emission_evidence"}
    evidence = json.loads(args.emission_evidence.read_text(encoding="utf-8"))
    assert evidence["bundle"]["claim_evidence"][0]["evidence"]["self_validation_consensus"]["accepted_count"] == 3


def test_operator_does_not_override_explicit_seed_inputs(tmp_path: Path) -> None:
    module = _load_module()
    local_state = tmp_path / "local-state.json"
    explicit_evidence = tmp_path / "explicit-evidence.json"
    explicit_evidence.write_text("{}\n", encoding="utf-8")
    _write_local_state(local_state)
    args = _args(
        tmp_path,
        local_state=local_state,
        test_wallet_address="0x00000000000000000000000000000000000000bb",
        test_old_coldkey="explicit-coldkey",
        emission_evidence=explicit_evidence,
    )
    paths = module._paths(args.artifacts_dir)

    filled = module._fill_seed_inputs_from_local_state(args, paths)

    assert filled == {}
    assert args.test_wallet_address == "0x00000000000000000000000000000000000000bb"
    assert args.test_old_coldkey == "explicit-coldkey"
    assert args.emission_evidence == explicit_evidence


def test_operator_passes_aws_profile_to_blocker_gate_and_inventory(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, aws_profile="moonrocklab-frankfurt", aws_region="us-west-2")
    paths = module._paths(args.artifacts_dir)
    seen = {}

    def fake_run(cmd: list[str], **kwargs) -> dict:
        _write_standard_reports(module, paths, cmd)
        if _has_cmd(cmd, "sota_base_testnet_blockers.py"):
            seen["blockers"] = cmd
        if _has_cmd(cmd, "sota_base_testnet_aws_inventory.py"):
            seen["inventory"] = cmd
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)
    module.run_operator(args)

    assert "--aws-profile" in seen["blockers"]
    assert seen["blockers"][seen["blockers"].index("--aws-profile") + 1] == "moonrocklab-frankfurt"
    assert "--gas-address" in seen["blockers"]
    assert (
        seen["blockers"][seen["blockers"].index("--gas-address") + 1]
        == "test_wallet=0x00000000000000000000000000000000000000aa"
    )
    assert "--aws-profile" in seen["inventory"]
    assert seen["inventory"][seen["inventory"].index("--aws-profile") + 1] == "moonrocklab-frankfurt"
    assert "--region" in seen["inventory"]
    assert seen["inventory"][seen["inventory"].index("--region") + 1] == "us-west-2"
    assert "--service-url" in seen["inventory"]


def test_operator_passes_configured_service_urls_to_blocker_gate(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(
        tmp_path,
        claims_ui_url="https://abc.awsapprunner.com",
        claims_api_url="https://def.awsapprunner.com",
        coordinator_url="https://ghi.awsapprunner.com",
        attestation_url="https://attestation.internal.example",
        root_publisher_url="https://root.internal.example",
        monitoring_url="https://monitoring.internal.example",
    )
    paths = module._paths(args.artifacts_dir)
    seen = {}

    def fake_run(cmd: list[str], **kwargs) -> dict:
        _write_standard_reports(module, paths, cmd)
        if _has_cmd(cmd, "sota_base_testnet_blockers.py"):
            seen["blockers"] = cmd
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)
    module.run_operator(args)

    host_values = [
        seen["blockers"][index + 1]
        for index, item in enumerate(seen["blockers"])
        if item == "--host"
    ]
    assert "claims_ui=https://abc.awsapprunner.com" in host_values
    assert "claims_api=https://def.awsapprunner.com" in host_values
    assert "coordinator=https://ghi.awsapprunner.com" in host_values
    assert "attestation=https://attestation.internal.example" in host_values
    assert "root_publisher=https://root.internal.example" in host_values
    assert "monitoring=https://monitoring.internal.example" not in host_values


def test_operator_passes_configured_app_runner_urls_to_aws_inventory(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(
        tmp_path,
        claims_ui_url="https://abc.awsapprunner.com",
        claims_api_url="https://def.awsapprunner.com",
        coordinator_url="https://ghi.awsapprunner.com",
        root_publisher_url="https://root.awsapprunner.com",
    )
    paths = module._paths(args.artifacts_dir)
    seen = {}

    def fake_run(cmd: list[str], **kwargs) -> dict:
        _write_standard_reports(module, paths, cmd)
        if _has_cmd(cmd, "sota_base_testnet_aws_inventory.py"):
            seen["inventory"] = cmd
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)
    module.run_operator(args)

    service_values = [
        seen["inventory"][index + 1]
        for index, item in enumerate(seen["inventory"])
        if item == "--service-url"
    ]
    assert "claims_ui=https://abc.awsapprunner.com" in service_values
    assert "claims_api=https://def.awsapprunner.com" in service_values
    assert "coordinator=https://ghi.awsapprunner.com" in service_values
    assert "root_publisher=https://root.awsapprunner.com" in service_values


def test_operator_dry_run_roots_keeps_claim_import_yellow(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, deployment=tmp_path / "compact.json", emission_evidence=tmp_path / "evidence.json")
    args.deployment.write_text("{}\n", encoding="utf-8")
    args.emission_evidence.write_text("{}\n", encoding="utf-8")
    paths = module._paths(args.artifacts_dir)

    def fake_run(cmd: list[str], **kwargs) -> dict:
        _write_standard_reports(module, paths, cmd)
        if _has_cmd(cmd, "sota_base_testnet_rehearsal.py"):
            module._write_json(paths["rehearsal_report"], {"ok": True, "status": "green"})
            module._write_json(paths["manifest"], {"environment": "base-sepolia", "chain": {"chain_id": 84532}})
            paths["env"].write_text("SOTA_CLAIMS_API_URL=https://claims-api-test.example.invalid\n", encoding="utf-8")
        if _has_cmd(cmd, "sota_base_testnet_seed_artifacts.py") and "build" in cmd:
            module._write_json(paths["genesis_root_artifact"], {"root": {"root": "0x" + "11" * 32}})
            module._write_json(paths["emission_root_artifact"], {"root": {"root": "0x" + "12" * 32}})
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)
    report = module.run_operator(args)
    steps = {step["name"]: step for step in report["steps"]}

    assert report["status"] == "yellow"
    assert steps["publish_genesis_root"]["status"] == "yellow"
    assert steps["publish_emission_root"]["status"] == "yellow"
    assert steps["finalize_claim_artifacts"]["status"] == "yellow"
    assert steps["import_claim_artifacts"]["status"] == "yellow"
    assert "--broadcast" not in " ".join(steps["publish_genesis_root"]["command"])


def test_operator_full_broadcast_finalize_import_path_can_be_green(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(
        tmp_path,
        deployment=tmp_path / "compact.json",
        emission_evidence=tmp_path / "evidence.json",
        broadcast_roots=True,
        import_artifacts=True,
    )
    args.deployment.write_text("{}\n", encoding="utf-8")
    args.emission_evidence.write_text("{}\n", encoding="utf-8")
    paths = module._paths(args.artifacts_dir)

    def fake_run(cmd: list[str], **kwargs) -> dict:
        _write_standard_reports(module, paths, cmd)
        if _has_cmd(cmd, "sota_base_testnet_rehearsal.py"):
            module._write_json(paths["rehearsal_report"], {"ok": True, "status": "green"})
            module._write_json(paths["manifest"], {"environment": "base-sepolia", "chain": {"chain_id": 84532}})
            paths["env"].write_text("SOTA_CLAIMS_API_URL=https://claims-api-test.example.invalid\n", encoding="utf-8")
        if _has_cmd(cmd, "sota_base_testnet_seed_artifacts.py") and "build" in cmd:
            module._write_json(paths["seed_report"], {"status": "ready_to_publish_roots"})
            module._write_json(paths["genesis_root_artifact"], {"root": {"root": "0x" + "11" * 32}})
            module._write_json(paths["emission_root_artifact"], {"root": {"root": "0x" + "12" * 32}})
        if _has_cmd(cmd, "sota_base_publish_root.py"):
            out = Path(cmd[cmd.index("--out") + 1])
            kind = cmd[cmd.index("--kind") + 1]
            root_id = "0x" + ("21" if kind == "genesis" else "22") * 32
            module._write_json(out, {"status": "broadcasted", "root_id": root_id})
        if _has_cmd(cmd, "sota_base_testnet_seed_artifacts.py") and "finalize" in cmd:
            module._write_json(paths["seed_finalized_report"], {"status": "ready_to_import_claim_artifacts"})
            module._write_json(paths["genesis_claim_artifact"], {"indexer_import_ready": True})
            module._write_json(paths["emission_claim_artifact"], {"indexer_import_ready": True})
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)
    monkeypatch.setattr(module, "_post_json", lambda *args, **kwargs: {"indexed": 1})

    report = module.run_operator(args)
    steps = {step["name"]: step for step in report["steps"]}

    assert report["ok"] is True
    assert report["status"] == "green"
    assert steps["publish_genesis_root"]["status"] == "green"
    assert steps["finalize_claim_artifacts"]["status"] == "green"
    assert steps["import_claim_artifacts"]["status"] == "green"
    assert report["read_only_default"] is False


def test_operator_loads_deployer_private_key_from_secret_handle(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(
        tmp_path,
        deploy=True,
        private_key_secret_id="base-sota/test/deployer",
        private_key_secret_json_key="private_key",
        emission_evidence=tmp_path / "evidence.json",
    )
    args.emission_evidence.write_text("{}\n", encoding="utf-8")
    paths = module._paths(args.artifacts_dir)
    seen_env = {}

    monkeypatch.delenv("SOTA_DEPLOYER_PRIVATE_KEY", raising=False)
    monkeypatch.setattr(module, "_aws_secret_string", lambda *args, **kwargs: '{"private_key":"0x' + '12' * 32 + '"}')

    def fake_run(cmd: list[str], **kwargs) -> dict:
        _write_standard_reports(module, paths, cmd)
        if _has_cmd(cmd, "sota_base_testnet_rehearsal.py"):
            seen_env.update(kwargs.get("env_overrides") or {})
            module._write_json(paths["rehearsal_report"], {"ok": True, "status": "green"})
            module._write_json(paths["manifest"], {"environment": "base-sepolia", "chain": {"chain_id": 84532}})
            paths["env"].write_text("SOTA_CLAIMS_API_URL=https://claims-api-test.example.invalid\n", encoding="utf-8")
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)

    report = module.run_operator(args)
    rehearsal = next(step for step in report["steps"] if step["name"] == "rehearsal")

    assert seen_env["SOTA_DEPLOYER_PRIVATE_KEY"] == "0x" + "12" * 32
    assert rehearsal["status"] == "green"
    assert "0x" + "12" * 32 not in json.dumps(rehearsal)


def test_operator_resolves_root_publisher_address_from_secret_tag(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(
        tmp_path,
        deploy=True,
        private_key_secret_id="base-sota/test/base-sepolia/deployer",
        root_publisher_private_key_secret_id="base-sota/test/base-sepolia/root-publisher",
        emission_evidence=tmp_path / "evidence.json",
    )
    args.emission_evidence.write_text("{}\n", encoding="utf-8")
    paths = module._paths(args.artifacts_dir)
    root_address = "0xec44b185Dc02bF1b0FC43c6914289c9310FE4dD0"
    rehearsal_commands: list[list[str]] = []

    monkeypatch.delenv("SOTA_DEPLOYER_PRIVATE_KEY", raising=False)
    monkeypatch.setattr(module, "_aws_secret_string", lambda *args, **kwargs: '{"private_key":"0x' + '12' * 32 + '"}')
    monkeypatch.setattr(module, "_aws_secret_tag", lambda *args, **kwargs: root_address)

    def fake_run(cmd: list[str], **kwargs) -> dict:
        _write_standard_reports(module, paths, cmd)
        if _has_cmd(cmd, "sota_base_testnet_rehearsal.py"):
            rehearsal_commands.append(cmd)
            module._write_json(paths["rehearsal_report"], {"ok": True, "status": "green"})
            module._write_json(paths["manifest"], {"environment": "base-sepolia", "chain": {"chain_id": 84532}})
            paths["env"].write_text("SOTA_CLAIMS_API_URL=https://claims-api-test.example.invalid\n", encoding="utf-8")
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)

    report = module.run_operator(args)
    rehearsal_cmd = rehearsal_commands[0]

    assert report["resolved_addresses"]["root_publisher"] == root_address
    assert "--root-publisher-address" in rehearsal_cmd
    assert rehearsal_cmd[rehearsal_cmd.index("--root-publisher-address") + 1] == root_address


def test_operator_reports_secret_handle_load_failure_without_running_deploy(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, deploy=True, private_key_secret_id="base-sota/test/deployer")
    paths = module._paths(args.artifacts_dir)

    monkeypatch.delenv("SOTA_DEPLOYER_PRIVATE_KEY", raising=False)
    monkeypatch.setattr(module, "_aws_secret_string", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("AccessDenied")))

    def fake_run(cmd: list[str], **kwargs) -> dict:
        assert not _has_cmd(cmd, "sota_base_testnet_rehearsal.py")
        _write_standard_reports(module, paths, cmd)
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)

    report = module.run_operator(args)
    rehearsal = next(step for step in report["steps"] if step["name"] == "rehearsal")

    assert rehearsal["status"] == "red"
    assert "Could not load SOTA_DEPLOYER_PRIVATE_KEY" in rehearsal["detail"]
    assert "AccessDenied" in rehearsal["detail"]


def test_operator_loads_root_publisher_private_key_from_secret_handle(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(
        tmp_path,
        deployment=tmp_path / "compact.json",
        emission_evidence=tmp_path / "evidence.json",
        broadcast_roots=True,
        root_publisher_private_key_secret_id="base-sota/test/root-publisher",
        root_publisher_private_key_secret_json_key="root_publisher_private_key",
    )
    args.deployment.write_text("{}\n", encoding="utf-8")
    args.emission_evidence.write_text("{}\n", encoding="utf-8")
    paths = module._paths(args.artifacts_dir)
    seen_publish_envs = []
    secret_value = "0x" + "34" * 32

    monkeypatch.delenv("SOTA_ROOT_PUBLISHER_PRIVATE_KEY", raising=False)
    monkeypatch.setattr(module, "_aws_secret_string", lambda *args, **kwargs: '{"root_publisher_private_key":"' + secret_value + '"}')

    def fake_run(cmd: list[str], **kwargs) -> dict:
        _write_standard_reports(module, paths, cmd)
        if _has_cmd(cmd, "sota_base_testnet_rehearsal.py"):
            module._write_json(paths["rehearsal_report"], {"ok": True, "status": "green"})
            module._write_json(paths["manifest"], {"environment": "base-sepolia", "chain": {"chain_id": 84532}})
            paths["env"].write_text("SOTA_CLAIMS_API_URL=https://claims-api-test.example.invalid\n", encoding="utf-8")
        if _has_cmd(cmd, "sota_base_testnet_seed_artifacts.py") and "build" in cmd:
            module._write_json(paths["seed_report"], {"status": "ready_to_publish_roots"})
            module._write_json(paths["genesis_root_artifact"], {"root": {"root": "0x" + "11" * 32}})
            module._write_json(paths["emission_root_artifact"], {"root": {"root": "0x" + "12" * 32}})
        if _has_cmd(cmd, "sota_base_publish_root.py"):
            seen_publish_envs.append(kwargs.get("env_overrides") or {})
            out = Path(cmd[cmd.index("--out") + 1])
            kind = cmd[cmd.index("--kind") + 1]
            root_id = "0x" + ("21" if kind == "genesis" else "22") * 32
            module._write_json(out, {"status": "broadcasted", "root_id": root_id})
        return _command_result(cmd)

    monkeypatch.setattr(module, "_run_command", fake_run)

    report = module.run_operator(args)
    steps = {step["name"]: step for step in report["steps"]}

    assert seen_publish_envs
    assert all(item["SOTA_ROOT_PUBLISHER_PRIVATE_KEY"] == secret_value for item in seen_publish_envs)
    assert steps["publish_genesis_root"]["status"] == "green"
    assert steps["publish_emission_root"]["status"] == "green"
    assert secret_value not in json.dumps(report)


def test_operator_main_without_report_out_does_not_write_to_current_directory(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module,
        "run_operator",
        lambda args: {
            "ok": False,
            "status": "red",
            "summary": {"green": 0, "yellow": 0, "red": 1},
            "paths": {"operator_report": str(tmp_path / "operator.json")},
            "next_actions": [],
        },
    )

    assert module.main(["--allow-blocked"]) == 0
    assert tmp_path.is_dir()


def test_run_command_timeout_returns_red_compatible_result(monkeypatch) -> None:
    module = _load_module()

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args") or args[0], timeout=1, output=b"partial")

    monkeypatch.setattr(module.subprocess, "run", timeout_run)

    result = module._run_command(["slow"], timeout=1)

    assert result["returncode"] == 124
    assert result["stdout"] == "partial"
    assert "timed out" in result["stderr"]
    assert result["command"] == ["slow"]


def test_step_from_result_turns_zero_gas_into_funding_remediation() -> None:
    module = _load_module()

    step = module._step_from_result(
        "rehearsal",
        {
            "returncode": 1,
            "stdout": "",
            "stderr": "deployer 0x1111111111111111111111111111111111111111 has no native gas balance on chain 84532",
            "command": ["deploy"],
        },
        success_detail="ok",
        failure_remediation="generic",
    )

    assert step.status == "red"
    assert "Fund the listed deployer address" in step.remediation

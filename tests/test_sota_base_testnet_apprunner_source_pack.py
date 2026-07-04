from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_apprunner_source_pack.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_apprunner_source_pack", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    service_pack = tmp_path / "service-pack.json"
    input_dir = tmp_path / "apprunner"
    input_dir.mkdir(parents=True)
    service_pack.write_text(
        json.dumps(
            {
                "schema": "sota-base-testnet-service-pack/v1",
                "services": [
                    {
                        "key": "indexer_api",
                        "deployment_recipe": {"service_name": "base-sota-indexer-api-test"},
                        "source": {
                            "path": str(tmp_path / "repo"),
                            "remote_url": "https://github.com/AlveusLabs/94-agent-community.git",
                            "branch": "base-sota-test",
                            "commit_sha": "a" * 40,
                        },
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "repo").mkdir()
    (input_dir / "base-sota-indexer-api-test.json").write_text(
        json.dumps(
            {
                "ServiceName": "base-sota-indexer-api-test",
                "SourceConfiguration": {
                    "AuthenticationConfiguration": {"ConnectionArn": "${SOTA_APPRUNNER_CONNECTION_ARN}"},
                    "CodeRepository": {
                        "RepositoryUrl": "https://github.com/AlveusLabs/94-agent-community.git",
                        "SourceCodeVersion": {"Type": "BRANCH", "Value": "base-sota-test"},
                        "CodeConfiguration": {
                            "ConfigurationSource": "API",
                            "CodeConfigurationValues": {
                                "Runtime": "PYTHON_311",
                                "BuildCommand": "python3 -m pip install -e .",
                                "StartCommand": "python3 -m uvicorn app:app",
                                "Port": "8010",
                                "RuntimeEnvironmentVariables": {},
                                "RuntimeEnvironmentSecrets": {
                                    "SOTA_BASE_INDEXER_ADMIN_TOKEN": "base-sota/test/base-sepolia/indexer-admin-token"
                                },
                            },
                        },
                        "SourceDirectory": "/",
                    },
                },
                "InstanceConfiguration": {
                    "Cpu": "1024",
                    "Memory": "2048",
                    "InstanceRoleArn": "${SOTA_APPRUNNER_INSTANCE_ROLE_ARN}",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return service_pack, input_dir


def _args(tmp_path: Path, **overrides):
    service_pack, input_dir = _write_inputs(tmp_path)
    values = {
        "service_pack": service_pack,
        "apprunner_input_dir": input_dir,
        "out_dir": tmp_path / "apprunner-source",
        "report_out": tmp_path / "source-pack.json",
        "aws_profile": "moonrocklab-frankfurt",
        "region": "eu-central-1",
        "connection_name": "bitsota",
        "connection_arn": "",
        "instance_role_arn": "arn:aws:iam::924380800822:role/AppRunnerReadSecrets",
        "no_resolve_connection_arn": False,
        "skip_remote_check": False,
        "timeout": 1.0,
        "json": False,
        "allow_blocked": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_source_pack_renders_aws_ready_apprunner_json(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    monkeypatch.setattr(
        module,
        "_run_aws",
        lambda *a, **k: {
            "ConnectionSummaryList": [
                {
                    "ConnectionName": "bitsota",
                    "ConnectionArn": "arn:aws:apprunner:eu-central-1:924380800822:connection/bitsota/abc",
                    "Status": "AVAILABLE",
                }
            ]
        },
    )
    monkeypatch.setattr(module, "_git_status", lambda path, timeout: "")
    monkeypatch.setattr(module, "_git_remote_head", lambda remote_url, branch, timeout: "a" * 40)

    report = module.build_pack(args)
    rendered = json.loads((args.out_dir / "base-sota-indexer-api-test.json").read_text(encoding="utf-8"))

    assert report["schema"] == "sota-base-testnet-apprunner-source-pack/v1"
    assert report["status"] == "green"
    assert report["ok"] is True
    assert rendered["SourceConfiguration"]["AuthenticationConfiguration"]["ConnectionArn"].endswith("/abc")
    assert rendered["InstanceConfiguration"]["InstanceRoleArn"].endswith("AppRunnerReadSecrets")
    assert report["rendered_services"][0]["create_service_command"][0:3] == ["aws", "apprunner", "create-service"]


def test_source_pack_marks_dirty_source_and_missing_connection_yellow(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, no_resolve_connection_arn=True, instance_role_arn="", skip_remote_check=True)
    monkeypatch.setattr(module, "_git_status", lambda path, timeout: " M experiments/base_protocol_design/sota_base_indexer/api.py\n M unrelated.md")

    report = module.build_pack(args)
    checks = {check["name"]: check for check in report["checks"]}
    publication = report["source_publication"][0]

    assert report["ok"] is False
    assert report["status"] == "yellow"
    assert checks["connection_arn"]["status"] == "yellow"
    assert checks["instance_role_arn"]["status"] == "yellow"
    assert checks["source_dirty_base-sota-indexer-api-test"]["status"] == "yellow"
    assert "Commit and push" in checks["source_dirty_base-sota-indexer-api-test"]["remediation"]
    assert publication["dirty_count"] == 2
    assert publication["deployment_relevant_dirty_paths"] == [
        "experiments/base_protocol_design/sota_base_indexer/api.py"
    ]


def test_dirty_paths_from_status_preserves_first_path_leading_dot() -> None:
    module = _load_module()
    status = " M .env.testnet.example\n M README.md\n?? .dockerignore"

    assert module._dirty_paths_from_status(status) == [
        ".env.testnet.example",
        "README.md",
        ".dockerignore",
    ]


def test_indexer_relevant_paths_ignore_generated_artifacts() -> None:
    module = _load_module()
    dirty_paths = [
        "experiments/base_protocol_design/sota_base_indexer/api.py",
        "experiments/base_protocol_design/sota_base_indexer/__pycache__/api.cpython-312.pyc",
        "experiments/base_protocol_design/out/tokenomics_sim.csv",
        "experiments/base_protocol_design/base_frontier_protocol_report.html",
        "tests/test_sota_base_indexer.py",
    ]

    assert module._deployment_relevant_paths(
        "base-sota-indexer-api-test",
        {"key": "indexer_api"},
        dirty_paths,
    ) == [
        "experiments/base_protocol_design/sota_base_indexer/api.py",
        "tests/test_sota_base_indexer.py",
    ]


def test_source_pack_ignores_unrelated_dirty_paths_for_readiness(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    monkeypatch.setattr(
        module,
        "_run_aws",
        lambda *a, **k: {
            "ConnectionSummaryList": [
                {
                    "ConnectionName": "bitsota",
                    "ConnectionArn": "arn:aws:apprunner:eu-central-1:924380800822:connection/bitsota/abc",
                    "Status": "AVAILABLE",
                }
            ]
        },
    )
    monkeypatch.setattr(module, "_git_status", lambda path, timeout: " M unrelated.md")
    monkeypatch.setattr(module, "_git_remote_head", lambda remote_url, branch, timeout: "a" * 40)

    report = module.build_pack(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["status"] == "green"
    assert checks["source_dirty_base-sota-indexer-api-test"]["status"] == "green"
    assert report["source_publication"][0]["deployment_relevant_dirty_count"] == 0

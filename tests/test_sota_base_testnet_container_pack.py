from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_container_pack.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_container_pack", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _service(key: str, *, env_values: dict | None = None, env_secrets: dict | None = None) -> dict:
    return {
        "key": key,
        "env_public_values": env_values or {},
        "env_secret_map": env_secrets or {},
    }


def _write_service_pack(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "sota-base-testnet-service-pack/v1",
                "services": [
                    _service(
                        "claims_ui",
                        env_values={
                            "NEXT_PUBLIC_SOTA_ENVIRONMENT": "testnet",
                            "NEXT_PUBLIC_SOTA_BASE_CHAIN_ID": "84532",
                            "NEXT_PUBLIC_SOTA_CLAIMS_API_URL": "https://claims-api-test.example.invalid",
                        },
                    ),
                    _service(
                        "indexer_api",
                        env_values={"SOTA_BASE_CHAIN_ID": "84532"},
                        env_secrets={"SOTA_BASE_INDEXER_ADMIN_TOKEN": "base-sota/test/base-sepolia/indexer-admin-token"},
                    ),
                    _service(
                        "autoresearch_coordinator",
                        env_values={"SOTA_DEFAULT_LANE_ID": "base:sota-local"},
                        env_secrets={"DATABASE_URL": "base-sota/test/base-sepolia/autoresearch-database-url"},
                    ),
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _args(tmp_path: Path, **overrides):
    values = {
        "service_pack": _write_service_pack(tmp_path / "service-pack.json"),
        "out": tmp_path / "container-pack.json",
        "apprunner_out_dir": tmp_path / "apprunner-image",
        "aws_profile": "moonrocklab-frankfurt",
        "region": "eu-central-1",
        "account_id": "924380800822",
        "tag": "test",
        "ensure_ecr": False,
        "apprunner_ecr_access_role_arn": "arn:aws:iam::924380800822:role/AppRunnerECRAccessRole",
        "apprunner_instance_role_arn": "arn:aws:iam::924380800822:role/AppRunnerReadSecrets",
        "timeout": 1.0,
        "json": False,
        "allow_blocked": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_container_pack_writes_ecr_image_apprunner_inputs(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)

    report = module.build_pack(args)
    claims_input = json.loads((args.apprunner_out_dir / "base-sota-claims-ui-test.json").read_text(encoding="utf-8"))
    indexer_input = json.loads((args.apprunner_out_dir / "base-sota-indexer-api-test.json").read_text(encoding="utf-8"))

    assert report["schema"] == "sota-base-testnet-container-pack/v1"
    assert report["status"] == "green"
    assert {image["key"] for image in report["images"]} == {"claims_ui", "indexer_api", "autoresearch_coordinator"}
    assert claims_input["SourceConfiguration"]["ImageRepository"]["ImageRepositoryType"] == "ECR"
    assert claims_input["SourceConfiguration"]["ImageRepository"]["ImageConfiguration"]["Port"] == "3000"
    assert claims_input["SourceConfiguration"]["AuthenticationConfiguration"]["AccessRoleArn"].endswith("AppRunnerECRAccessRole")
    assert "--build-arg" in next(image for image in report["images"] if image["key"] == "claims_ui")["build_command"]
    assert indexer_input["SourceConfiguration"]["ImageRepository"]["ImageConfiguration"]["RuntimeEnvironmentSecrets"]["SOTA_BASE_INDEXER_ADMIN_TOKEN"] == "base-sota/test/base-sepolia/indexer-admin-token"


def test_container_pack_can_ensure_ecr_without_printing_secret_values(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, ensure_ecr=True)
    created: list[str] = []

    def fake_ensure(name: str, **kwargs):
        created.append(name)
        return {"status": "green", "action": "created", "repository_uri": f"924380800822.dkr.ecr.eu-central-1.amazonaws.com/{name}"}

    monkeypatch.setattr(module, "_ensure_ecr_repo", fake_ensure)

    report = module.build_pack(args)
    serialized = json.dumps(report)

    assert set(created) == {"base-sota-claims-ui-test", "base-sota-indexer-api-test", "base-sota-autoresearch-coordinator-test"}
    assert report["status"] == "green"
    assert "database_url" not in serialized.lower()
    assert "private_key" not in serialized.lower()


def test_container_pack_marks_missing_ecr_access_role_as_yellow(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, apprunner_ecr_access_role_arn="")

    report = module.build_pack(args)
    check = next(item for item in report["checks"] if item["name"] == "apprunner_ecr_access_role")

    assert report["status"] == "yellow"
    assert check["status"] == "yellow"

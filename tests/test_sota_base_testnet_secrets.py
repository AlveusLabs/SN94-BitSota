from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_secrets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_secrets", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    values = {
        "command": "plan",
        "aws_profile": "moonrocklab-frankfurt",
        "region": "eu-central-1",
        "rpc_url": "https://sepolia.base.org",
        "timeout": 1.0,
        "out": tmp_path / "secrets.json",
        "json": False,
        "allow_blocked": False,
        "create_autoresearch_database": False,
        "source_autoresearch_db_secret_id": "bitsota/test/db",
        "autoresearch_database_name": "base_sota_testnet_autoresearch",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_secret_plan_is_read_only_and_lists_external_requirements(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)

    monkeypatch.setattr(module, "_safe_describe_secret", lambda *a, **k: (None, None))

    report = module.build_report(args)
    actions = {item["name"]: item for item in report["secret_handles"]}

    assert report["schema"] == "sota-base-testnet-secret-bootstrap/v1"
    assert report["status"] == "red"
    assert actions["base-sota/test/base-sepolia/deployer"]["action"] == "would_create"
    assert actions["base-sota/test/base-sepolia/autoresearch-database-url"]["action"] == "external_required"
    assert "placeholder" in actions["base-sota/test/base-sepolia/autoresearch-database-url"]["detail"]
    assert report["read_secret_values"] is False
    assert report["prints_secret_values"] is False


def test_secret_create_writes_only_managed_handles_and_redacts_values(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, command="create")
    created: list[tuple[str, dict, str | None]] = []

    monkeypatch.setattr(module, "_safe_describe_secret", lambda *a, **k: (None, None))

    def fake_create(spec, payload, *, address, profile, region, timeout):
        created.append((spec.name, payload, address))
        return {"ARN": f"arn:aws:secretsmanager:eu-central-1:924380800822:secret:{spec.name}"}

    monkeypatch.setattr(module, "_create_secret", fake_create)

    report = module.build_report(args)
    serialized = json.dumps(report)
    created_names = {name for name, _payload, _address in created}

    assert "base-sota/test/base-sepolia/deployer" in created_names
    assert "base-sota/test/base-sepolia/root-publisher" in created_names
    assert "base-sota/test/base-sepolia/indexer-admin-token" in created_names
    assert "base-sota/test/base-sepolia/autoresearch-database-url" not in created_names
    assert "private_key" not in serialized
    assert "root_publisher_private_key" not in serialized
    assert "admin_token" not in serialized
    deployer = next(item for item in report["secret_handles"] if item["name"].endswith("/deployer"))
    assert deployer["status"] == "green"
    assert str(deployer["address"]).startswith("0x")
    assert report["status"] == "red"


def test_secret_create_keeps_existing_handles_without_reading_values(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, command="create")

    def fake_describe(name: str, **kwargs):
        if name.endswith("/deployer"):
            return (
                {
                    "ARN": "arn:deployer",
                    "Tags": [{"Key": "sota-address", "Value": "0x1111111111111111111111111111111111111111"}],
                },
                None,
            )
        return (None, None)

    monkeypatch.setattr(module, "_safe_describe_secret", fake_describe)
    monkeypatch.setattr(module, "_create_secret", lambda spec, payload, **kwargs: {"ARN": f"arn:{spec.name}"})

    report = module.build_report(args)
    deployer = next(item for item in report["secret_handles"] if item["name"].endswith("/deployer"))

    assert deployer["action"] == "exists"
    assert deployer["address"] == "0x1111111111111111111111111111111111111111"


def test_autoresearch_database_bootstrap_refuses_prod_source(tmp_path: Path) -> None:
    module = _load_module()

    try:
        module._guard_source_database_secret("bitsota/prod/db", "base_sota_testnet_autoresearch")
    except SystemExit as exc:
        assert "production-looking" in str(exc)
    else:
        raise AssertionError("prod-looking source secret was not rejected")


def test_autoresearch_database_bootstrap_refuses_unsafe_database_name(tmp_path: Path) -> None:
    module = _load_module()

    try:
        module._guard_source_database_secret("bitsota/test/db", "postgres")
    except SystemExit as exc:
        assert "unsafe test database name" in str(exc)
    else:
        raise AssertionError("unsafe database name was not rejected")


def test_secret_create_can_create_autoresearch_database_without_leaking_values(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, command="create", create_autoresearch_database=True)
    created: list[tuple[str, dict, str | None]] = []

    monkeypatch.setattr(module, "_safe_describe_secret", lambda *a, **k: (None, None))
    monkeypatch.setattr(
        module,
        "_aws_secret_string",
        lambda *a, **k: '{"database_url":"postgresql://tester:super-secret@db.example.test:5432/bitsota"}',
    )
    monkeypatch.setattr(
        module,
        "_ensure_postgres_database",
        lambda source_url, database_name: {
            "database": database_name,
            "host": "db.example.test",
            "created_database": True,
            "database_url": "postgresql://tester:super-secret@db.example.test:5432/base_sota_testnet_autoresearch",
        },
    )

    def fake_create(spec, payload, *, address, profile, region, timeout):
        created.append((spec.name, payload, address))
        return {"ARN": f"arn:aws:secretsmanager:eu-central-1:924380800822:secret:{spec.name}"}

    monkeypatch.setattr(module, "_create_secret", fake_create)

    report = module.build_report(args)
    serialized = json.dumps(report)
    created_names = {name for name, _payload, _address in created}

    assert "base-sota/test/base-sepolia/autoresearch-database-url" in created_names
    assert report["read_secret_values"] is True
    assert report["prints_secret_values"] is False
    assert "super-secret" not in serialized
    assert "postgresql://tester" not in serialized
    assert "base_sota_testnet_autoresearch" in serialized
    database_action = next(item for item in report["secret_handles"] if item["name"].endswith("/autoresearch-database-url"))
    assert database_action["action"] == "database_created_secret_created"


def test_database_bootstrap_error_redacts_database_urls(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path, command="create", create_autoresearch_database=True)

    monkeypatch.setattr(module, "_safe_describe_secret", lambda *a, **k: (None, None))
    monkeypatch.setattr(module, "_aws_secret_string", lambda *a, **k: "postgresql://tester:super-secret@db.example.test:5432/bitsota")

    def fail_ensure(source_url, database_name):
        raise RuntimeError(f"could not connect to {source_url}")

    monkeypatch.setattr(module, "_ensure_postgres_database", fail_ensure)

    report = module.build_report(args)
    serialized = json.dumps(report)
    database_action = next(item for item in report["secret_handles"] if item["name"].endswith("/autoresearch-database-url"))

    assert database_action["action"] == "database_secret_create_failed"
    assert "postgresql://<redacted>" in database_action["detail"]
    assert "super-secret" not in serialized
    assert "postgresql://tester" not in serialized

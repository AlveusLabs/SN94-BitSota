from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_secret_value.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_secret_value", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "env_name": "SOTA_SECRET",
        "secret_id": "base-sota/test/secret",
        "field": ["admin_token"],
        "aws_profile": "moonrocklab-frankfurt",
        "aws_region": "eu-central-1",
        "timeout": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_secret_value_prefers_raw_env(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("SOTA_SECRET", "plain-token")

    assert module.load_value(_args()) == "plain-token"


def test_secret_value_reads_json_env_field(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("SOTA_SECRET", '{"admin_token":"json-token"}')

    assert module.load_value(_args()) == "json-token"


def test_secret_value_uses_fallback_fields(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("SOTA_SECRET", '{"private_key":"0xabc"}')

    assert module.load_value(_args(field=["root_publisher_private_key", "private_key"])) == "0xabc"


def test_secret_value_reads_aws_when_env_is_missing(monkeypatch) -> None:
    module = _load_module()
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout='{"admin_token":"aws-token"}', stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.load_value(_args()) == "aws-token"
    assert seen["command"][:3] == ["aws", "secretsmanager", "get-secret-value"]
    assert seen["kwargs"]["timeout"] == 1.0


def test_secret_value_reports_aws_failure(monkeypatch) -> None:
    module = _load_module()

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=255, stdout="", stderr="SSO token expired")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    try:
        module.load_value(_args())
    except RuntimeError as exc:
        assert "could not read AWS secret" in str(exc)
        assert "SSO token expired" in str(exc)
    else:
        raise AssertionError("expected AWS failure to raise")

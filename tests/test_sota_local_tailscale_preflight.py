from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_local_tailscale_preflight.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_local_tailscale_preflight", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path):
    return argparse.Namespace(
        state=tmp_path / "state.json",
        report_out=tmp_path / "tailscale-preflight.json",
        timeout=0.1,
    )


def _write_state(path: Path, *, mode: str = "tailscale-https", wallet_safe: bool = True) -> None:
    path.write_text(
        json.dumps(
            {
                "urls": {"anvil_rpc": "https://sota-host.example.ts.net:8545"},
                "sharing": {
                    "mode": mode,
                    "wallet_rpc_browser_safe": wallet_safe,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_tailscale_preflight_green_when_https_and_serve_are_configured(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_state(args.state)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/tailscale")

    def fake_run(command, **kwargs):
        if command[:2] == ["/usr/bin/tailscale", "status"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "BackendState": "Running",
                        "MagicDNSSuffix": "example.ts.net",
                        "CertDomains": ["sota-host.example.ts.net"],
                        "Self": {
                            "ID": "node123",
                            "HostName": "sota-host",
                            "DNSName": "sota-host.example.ts.net.",
                        },
                    }
                ),
                stderr="",
            )
        if command[:3] == ["/usr/bin/tailscale", "serve", "status"]:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"TCP": {"443": {}}}), stderr="")
        if command[:2] == ["/usr/bin/tailscale", "cert"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    report = module.run_preflight(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["serve_enable_url"] == "https://login.tailscale.com/f/serve?node=node123"
    assert (args.report_out).exists()


def test_tailscale_preflight_reports_disabled_https_and_http_share(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_state(args.state, mode="http", wallet_safe=False)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/tailscale")

    def fake_run(command, **kwargs):
        if command[:2] == ["/usr/bin/tailscale", "status"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "BackendState": "Running",
                        "MagicDNSSuffix": "example.ts.net",
                        "CertDomains": None,
                        "Self": {
                            "ID": "node123",
                            "HostName": "sota-host",
                            "DNSName": "sota-host.example.ts.net.",
                        },
                    }
                ),
                stderr="",
            )
        if command[:3] == ["/usr/bin/tailscale", "serve", "status"]:
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")
        if command[:2] == ["/usr/bin/tailscale", "cert"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="Access denied: cert access denied")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    report = module.run_preflight(args)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is False
    assert report["status"] == "red"
    assert checks["tailscale_https_certificates"]["status"] == "red"
    assert checks["tailscale_local_operator"]["status"] == "red"
    assert "sudo tailscale set --operator" in checks["tailscale_local_operator"]["remediation"]
    assert checks["local_wallet_rpc_share"]["status"] == "yellow"
    assert "https://login.tailscale.com/f/serve?node=node123" in checks["tailscale_https_certificates"]["remediation"]

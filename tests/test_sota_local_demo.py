from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_local_demo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_local_demo", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launch_claim_proof_runs_reset_after(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="proof ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_local_claim_proof_reset()

    command, kwargs = calls[0]
    assert command[0] == sys.executable
    assert command[1].endswith("scripts/sota_local_claim_proof.py")
    assert "--reset-after" in command
    assert str(module.RUN_DIR / "claim-proof" / "latest.json") in command
    assert str(module.RUN_DIR / "claim-proof" / "local-claim-tx-evidence.json") in command
    assert kwargs["cwd"] == module.DOCS_REPO
    assert kwargs["timeout"] == 420


def test_plan_public_share_uses_tailscale_https_urls(monkeypatch) -> None:
    module = _load_module()

    monkeypatch.setattr(module, "_primary_host", lambda: "100.64.0.10")
    monkeypatch.setattr(
        module,
        "_tailscale_status",
        lambda: {
            "MagicDNSSuffix": "example.ts.net",
            "Self": {"HostName": "sota-host", "DNSName": "sota-host.example.ts.net."},
        },
    )

    urls, sharing = module._plan_public_share("auto", require_remote_wallet=True)

    assert urls["claims_ui"] == "https://sota-host.example.ts.net:3000/claims"
    assert urls["anvil_rpc"] == "https://sota-host.example.ts.net:8545"
    assert sharing["mode"] == "tailscale-https"
    assert sharing["status"] == "pending"
    assert sharing["wallet_rpc_browser_safe"] is True


def test_plan_public_share_forced_tailscale_https_fails_without_dns(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_primary_host", lambda: "100.64.0.10")
    monkeypatch.setattr(module, "_tailscale_status", lambda: {})

    try:
        module._plan_public_share("tailscale-https", require_remote_wallet=True)
    except RuntimeError as exc:
        assert "MagicDNS" in str(exc)
    else:
        raise AssertionError("expected forced tailscale-https to fail without MagicDNS")


def test_activate_public_share_configures_tailscale_ports(monkeypatch) -> None:
    module = _load_module()
    ports = []

    monkeypatch.setattr(module, "_run_tailscale_serve_https", lambda port: (ports.append(port) or (True, "")))

    sharing = module._activate_public_share(
        {
            "mode": "tailscale-https",
            "status": "pending",
            "wallet_rpc_browser_safe": True,
        }
    )

    assert sharing["status"] == "green"
    assert sorted(sharing["configured_https_ports"]) == sorted(set(module.PUBLIC_SERVICE_PORTS.values()))
    assert sorted(ports) == sorted(set(module.PUBLIC_SERVICE_PORTS.values()))


def test_start_website_uses_browser_rpc_and_local_api_proxy(monkeypatch) -> None:
    module = _load_module()
    captured = {}
    state = {
        "accounts": {"alice_reward": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"},
        "genesis": {"old_coldkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"},
        "contracts": {
            "sota_token": "0x0000000000000000000000000000000000000001",
            "genesis_distributor": "0x0000000000000000000000000000000000000002",
            "emission_distributor": "0x0000000000000000000000000000000000000003",
        },
    }

    monkeypatch.setattr(module, "_is_port_open", lambda host, port: False)
    monkeypatch.setattr(module, "_wait_http", lambda url, timeout_seconds=60.0: None)

    def fake_start_process(name, args, *, cwd, env=None):
        captured["name"] = name
        captured["args"] = args
        captured["env"] = env or {}
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(module, "_start_process", fake_start_process)

    module._start_website(state, browser_rpc_url="https://sota-host.example.ts.net:8545")

    assert captured["name"] == "website"
    assert captured["env"]["NEXT_PUBLIC_SOTA_BASE_RPC_URL"] == "https://sota-host.example.ts.net:8545"
    assert captured["env"]["NEXT_PUBLIC_SOTA_CLAIMS_API_URL"] == "http://127.0.0.1:8010"
    assert captured["env"]["NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL"] == "http://127.0.0.1:8000"

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


def test_claim_proof_reset_relaunch_skips_recursive_miner_swarm(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="proof ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._run_local_claim_proof_reset()

    command, _kwargs = calls[0]
    assert "--reset-after" in command

    proof_module_spec = importlib.util.spec_from_file_location(
        "sota_local_claim_proof_for_reset_test",
        REPO / "scripts" / "sota_local_claim_proof.py",
    )
    assert proof_module_spec and proof_module_spec.loader
    proof_module = importlib.util.module_from_spec(proof_module_spec)
    sys.modules[proof_module_spec.name] = proof_module
    proof_module_spec.loader.exec_module(proof_module)

    reset_command = []

    def fake_reset_run(command, **kwargs):
        reset_command.extend(command)
        return SimpleNamespace(returncode=0, stdout="reset ok", stderr="")

    monkeypatch.setattr(proof_module.subprocess, "run", fake_reset_run)

    proof_module._restart_local_stack(timeout=1)

    assert reset_command[-3:] == ["launch", "--skip-claim-proof", "--skip-miner-swarm-proof"]


def test_launch_runs_claim_and_miner_swarm_proofs(monkeypatch) -> None:
    module = _load_module()
    captured = {}

    def fake_start_stack(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "start_stack", fake_start_stack)

    assert module.main(["launch"]) == 0

    assert captured["claim_proof"] is True
    assert captured["miner_swarm_proof"] is True


def test_launch_can_skip_claim_and_miner_swarm_proofs(monkeypatch) -> None:
    module = _load_module()
    captured = {}

    def fake_start_stack(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "start_stack", fake_start_stack)

    assert module.main(["launch", "--skip-claim-proof", "--skip-miner-swarm-proof"]) == 0

    assert captured["claim_proof"] is False
    assert captured["miner_swarm_proof"] is False


def test_refresh_tester_artifacts_runs_tailscale_preflight(monkeypatch) -> None:
    module = _load_module()
    calls = []

    monkeypatch.setattr(module, "_run_local_ui_smoke_report", lambda: calls.append("ui"))
    monkeypatch.setattr(module, "_run_tailscale_preflight_report", lambda: calls.append("tailscale"))
    monkeypatch.setattr(module, "_generate_release_status_report", lambda: calls.append("release"))
    monkeypatch.setattr(module, "_generate_handoff", lambda: calls.append("handoff"))

    module._refresh_tester_artifacts()

    assert calls == ["ui", "tailscale", "release", "handoff"]


def test_local_release_status_report_is_local_only(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._generate_release_status_report()

    command, kwargs = calls[0]
    assert "--local-only" in command
    assert str(module.LOCAL_RELEASE_STATUS_PATH) in command
    assert str(module.TESTNET_RUN_DIR / "base-sota-release-status.json") not in command
    assert kwargs["cwd"] == module.DOCS_REPO


def test_local_handoff_uses_local_release_status(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._generate_handoff()

    command, kwargs = calls[0]
    assert "--environment" in command
    assert "local" in command
    assert "--release-status" in command
    assert str(module.LOCAL_RELEASE_STATUS_PATH) in command
    assert kwargs["cwd"] == module.DOCS_REPO


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


def test_plan_public_share_auto_falls_back_to_wallet_safe_localhost(monkeypatch) -> None:
    module = _load_module()

    monkeypatch.setattr(module, "_primary_host", lambda: "100.64.0.10")
    monkeypatch.setattr(module, "_tailscale_status", lambda: {})

    urls, sharing = module._plan_public_share("auto", require_remote_wallet=True)

    assert urls["claims_ui"] == "http://127.0.0.1:3000/claims"
    assert urls["anvil_rpc"] == "http://127.0.0.1:8545"
    assert sharing["mode"] == "localhost"
    assert sharing["status"] == "green"
    assert sharing["wallet_rpc_browser_safe"] is True
    assert "only work on the computer running the demo" in sharing["warning"]


def test_local_swarm_miners_are_deterministic_distinct_and_private() -> None:
    module = _load_module()

    first = module._demo_swarm_miner(1)
    first_again = module._demo_swarm_miner(1)
    second = module._demo_swarm_miner(2)

    assert first["hotkey"] == first_again["hotkey"]
    assert first["miner_address"] == first_again["miner_address"]
    assert first["reward_address"] == first_again["reward_address"]
    assert first["hotkey"] != second["hotkey"]
    assert first["miner_address"] != second["miner_address"]
    assert first["reward_address"] != second["reward_address"]

    public = {key: value for key, value in first.items() if not key.endswith("_private_key")}
    assert "miner_private_key" not in public
    assert "reward_private_key" not in public


def test_extract_last_json_object_reads_miner_cli_result() -> None:
    module = _load_module()

    payload = module._extract_last_json_object(
        "[research-agent] selected task\n"
        "[research-agent] submitted task\n"
        '{\n  "claim": {"id": "claim-1"},\n  "submission": {"id": "sub-1"}\n}\n'
    )

    assert payload["claim"]["id"] == "claim-1"
    assert payload["submission"]["id"] == "sub-1"


def test_plan_public_share_http_can_expose_tailscale_ip_with_wallet_warning(monkeypatch) -> None:
    module = _load_module()

    monkeypatch.setattr(module, "_primary_host", lambda: "100.64.0.10")

    urls, sharing = module._plan_public_share("http", require_remote_wallet=True)

    assert urls["claims_ui"] == "http://100.64.0.10:3000/claims"
    assert sharing["mode"] == "http"
    assert sharing["status"] == "yellow"
    assert sharing["wallet_rpc_browser_safe"] is False


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

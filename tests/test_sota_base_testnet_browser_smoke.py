from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_testnet_browser_smoke.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_testnet_browser_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    artifacts_dir = tmp_path / "artifacts"
    values = {
        "artifacts_dir": artifacts_dir,
        "manifest": artifacts_dir / "base-sepolia-deployment-manifest.json",
        "env_file": artifacts_dir / "base-sota.env.testnet",
        "readiness_file": artifacts_dir / "base-sota-testnet-readiness.json",
        "claims_url": "",
        "claims_api_url": "",
        "autoresearch_url": "",
        "readiness_url": "",
        "test_wallet_address": "",
        "test_old_coldkey": "",
        "test_genesis_wallet_address": "",
        "test_genesis_coldkey": "",
        "test_snapshot_coldkey": "",
        "lane_id": "",
        "epoch": "",
        "timeout": 0.1,
        "allow_yellow": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_artifacts(args: argparse.Namespace, *, readiness_ok: bool = True, env_chain_id: str = "84532") -> None:
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "environment": "base-sepolia",
                "chain": {"chain_id": 84532},
                "services": {
                    "claims_ui": {"public_url": "https://claims-test.example.invalid"},
                    "indexer_api": {"public_base_url": "https://claims-api-test.example.invalid"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args.env_file.write_text(
        "\n".join(
            [
                f"NEXT_PUBLIC_SOTA_BASE_CHAIN_ID={env_chain_id}",
                "NEXT_PUBLIC_SOTA_CLAIMS_API_URL=https://claims-api-test.example.invalid",
                "NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL=https://coordinator-test.example.invalid",
                "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID=base:sota-local",
                "NEXT_PUBLIC_SOTA_READINESS_URL=https://claims-test.example.invalid/base-sota-testnet-readiness.json",
                "SOTA_TEST_WALLET_ADDRESS=0x5555555555555555555555555555555555555555",
                "SOTA_TEST_OLD_COLDKEY=5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "SOTA_TEST_GENESIS_WALLET_ADDRESS=0x6666666666666666666666666666666666666666",
                "SOTA_TEST_GENESIS_COLDKEY=5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
                "SOTA_TEST_SNAPSHOT_COLDKEY=5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
                "SOTA_TEST_EPOCH=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.readiness_file.write_text(
        json.dumps(
            {
                "schema": "sota-base-testnet-readiness/v1",
                "ok": readiness_ok,
                "status": "green" if readiness_ok else "red",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _install_green_http(module, monkeypatch) -> None:
    page_text = (
        " ".join(module.EXPECTED_TESTNET_CLAIMS_TEXT)
        + '<script src="/_next/static/chunks/claims.js"></script>'
    )
    binding_asset_text = " ".join(module.EXPECTED_BINDING_FRONTEND_TEXT)

    def fake_text(url: str, timeout: float):
        if url.endswith("/_next/static/chunks/claims.js"):
            return binding_asset_text
        return page_text

    monkeypatch.setattr(module, "_http_text", fake_text)

    def fake_json(method: str, url: str, *, payload=None, timeout: float):
        if url.endswith("/health"):
            return {"status": "ok", "chain_id": 84532}
        if url.endswith("/api/v1/base/index/sync"):
            return {"ok": True}
        if url.endswith("/api/v1/base/index/status"):
            return {
                "chain_id": 84532,
                "contracts_configured": [
                    "root_registry",
                    "lane_registry",
                    "genesis_distributor",
                    "emission_distributor",
                ],
                "lag_blocks": 0,
                "last_sync_error": None,
            }
        if url.endswith("/api/v1/base/genesis/binding-message"):
            return {
                "schema": "sota-snapshot-binding-message/v1",
                "status": "message_ready",
                "message": {
                    "coldkey": payload["coldkey"],
                    "reward_address": payload["reward_address"],
                    "base_chain_id": 84532,
                    "allocation_amount": 1500000000000000000,
                },
                "signing_payload": "{}",
                "snapshot_claim": {
                    "direct_tao_rao": "100",
                    "alpha_credit_rao": "50",
                    "alpha_credit_rao_by_netuid": {"1": "50"},
                },
            }
        if "/api/v1/base/eligibility/" in url and "subnet_id=genesis" in url:
            return {
                "eligible": True,
                "credits": {"total_sota": {"raw": "1500000000000000000"}},
            }
        if "/api/v1/base/eligibility/" in url and "subnet_id=base%3Asota-local" in url:
            return {
                "eligible": True,
                "credits": {"total_sota": {"raw": "2000000000000000000"}},
            }
        if url.endswith("/api/v1/base/claims/transaction"):
            return {"transaction": {"chainId": 84532, "data": "0x1234"}}
        if "/api/v1/sota/subnets/" in url:
            return {
                "root": {"root": "0x" + "1" * 64},
                "bundle": {
                    "claim_evidence": [
                        {
                            "evidence": {
                                "self_validation_consensus": {"status": "accepted"}
                            }
                        }
                    ]
                },
            }
        raise AssertionError(f"unexpected {method} {url}")

    monkeypatch.setattr(module, "_http_json", fake_json)
    monkeypatch.setattr(
        module,
        "_http_json_response",
        lambda method, url, payload=None, timeout=0.1: (
            422,
            {"detail": {"code": "invalid_binding_signature"}},
        ),
    )
    monkeypatch.setattr(module, "_http_status", lambda method, url, timeout: 204)


def test_browser_smoke_green_for_public_testnet_fixture(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    _install_green_http(module, monkeypatch)

    report = module.run_browser_smoke(args)

    assert report["ok"] is True
    assert report["status"] == "green"
    assert report["summary"]["red"] == 0
    names = {check["name"]: check for check in report["checks"]}
    assert names["claims_page_text"]["status"] == "green"
    assert names["claims_binding_frontend"]["status"] == "green"
    assert names["genesis_binding_message"]["status"] == "green"
    assert names["genesis_binding_submit_route"]["status"] == "green"
    assert names["genesis_calldata"]["status"] == "green"
    assert names["emission_calldata"]["status"] == "green"
    assert names["self_validation_evidence"]["status"] == "green"
    assert report["targets"]["test_wallet_address"] == "0x5555555555555555555555555555555555555555"
    assert report["targets"]["test_genesis_wallet_address"] == "0x6666666666666666666666666666666666666666"


def test_browser_smoke_accepts_already_claimed_seeded_wallet(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    _install_green_http(module, monkeypatch)

    original_http_json = module._http_json

    def wrapped_json(method: str, url: str, *, payload=None, timeout: float):
        if "/api/v1/base/eligibility/" in url:
            return {
                "eligible": True,
                "credits": {
                    "total_sota": {"raw": "1500000000000000000"},
                    "claimed_sota": {"raw": "1500000000000000000"},
                    "unclaimed_sota": {"raw": "0"},
                },
                "claim_state": {"status": "claimed", "claimable": False},
            }
        if url.endswith("/api/v1/base/claims/transaction"):
            raise RuntimeError('POST failed with HTTP 409: {"detail":{"code":"already_claimed"}}')
        return original_http_json(method, url, payload=payload, timeout=timeout)

    monkeypatch.setattr(module, "_http_json", wrapped_json)

    report = module.run_browser_smoke(args)
    names = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is True
    assert names["genesis_calldata"]["status"] == "green"
    assert "already complete" in names["genesis_calldata"]["detail"]
    assert names["emission_calldata"]["status"] == "green"


def test_browser_smoke_rejects_readiness_that_is_not_green(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args, readiness_ok=False)
    _install_green_http(module, monkeypatch)

    report = module.run_browser_smoke(args)
    readiness = next(check for check in report["checks"] if check["name"] == "readiness_report")

    assert report["ok"] is False
    assert readiness["status"] == "red"


def test_browser_smoke_rejects_base_mainnet_chain_id(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args, env_chain_id="8453")
    _install_green_http(module, monkeypatch)

    report = module.run_browser_smoke(args)
    chain = next(check for check in report["checks"] if check["name"] == "chain_config")

    assert report["ok"] is False
    assert chain["status"] == "red"
    assert "mainnet" in chain["detail"].lower()


def test_browser_smoke_requires_seeded_old_coldkey(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    env = args.env_file.read_text(encoding="utf-8").replace(
        "SOTA_TEST_OLD_COLDKEY=5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY\n",
        "",
    )
    args.env_file.write_text(env, encoding="utf-8")
    _install_green_http(module, monkeypatch)

    report = module.run_browser_smoke(args)
    old_coldkey = next(check for check in report["checks"] if check["name"] == "test_old_coldkey")

    assert report["ok"] is False
    assert old_coldkey["status"] == "red"


def test_browser_smoke_defaults_genesis_inputs_from_seeded_fixture(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    env = args.env_file.read_text(encoding="utf-8")
    env = env.replace("SOTA_TEST_GENESIS_WALLET_ADDRESS=0x6666666666666666666666666666666666666666\n", "")
    env = env.replace("SOTA_TEST_GENESIS_COLDKEY=5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM\n", "")
    env = env.replace("SOTA_TEST_SNAPSHOT_COLDKEY=5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM\n", "")
    args.env_file.write_text(env, encoding="utf-8")
    args.test_genesis_wallet_address = ""
    args.test_genesis_coldkey = ""
    _install_green_http(module, monkeypatch)

    report = module.run_browser_smoke(args)
    names = {check["name"]: check for check in report["checks"]}

    assert report["ok"] is True
    assert names["test_genesis_wallet_address"]["status"] == "green"
    assert names["test_genesis_coldkey"]["status"] == "green"
    assert report["targets"]["test_genesis_wallet_address"] == report["targets"]["test_wallet_address"]


def test_browser_smoke_rejects_claims_page_missing_wallet_copy(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_artifacts(args)
    _install_green_http(module, monkeypatch)
    monkeypatch.setattr(module, "_http_text", lambda url, timeout: "plain shell")

    report = module.run_browser_smoke(args)
    page = next(check for check in report["checks"] if check["name"] == "claims_page_text")

    assert report["ok"] is False
    assert page["status"] == "red"
    assert "Base Sepolia claims" in page["remediation"]


def test_browser_smoke_json_without_report_out_only_prints(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module,
        "run_browser_smoke",
        lambda args: {
            "schema": "sota-base-testnet-browser-smoke/v1",
            "ok": True,
            "status": "green",
            "checks": [],
            "summary": {"green": 0, "yellow": 0, "red": 0},
        },
    )

    exit_code = module.main(["--json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "green"
    assert tmp_path.is_dir()

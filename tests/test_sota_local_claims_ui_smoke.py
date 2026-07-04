from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_local_claims_ui_smoke.py"


def _load_module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("sota_local_claims_ui_smoke", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state() -> dict:
    return {
        "chain_id": 31337,
        "accounts": {"alice_reward": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"},
        "contracts": {
            "genesis_distributor": "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9",
            "emission_distributor": "0x5FC8d32690cc91D4c39d9d3abcBD16989F875707",
        },
        "genesis": {
            "old_coldkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "tao_credit": 1000000000000000000,
            "alpha_synthetic_credit": 500000000000000000,
            "amount": 1500000000000000000,
        },
        "emission_onchain": {"amount": 2000000000000000000},
        "autoresearch": {
            "subnet": {"id": "base:sota-local"},
            "emission_root": {
                "root": "0x9485d52e9c3f33701e38158e997c9a0d64c07b002126d2a4e5810d63ad268594"
            },
        },
        "urls": {"docs": "http://127.0.0.1:9002/base/"},
    }


def test_build_targets_uses_claims_proxy_and_encodes_lane() -> None:
    module = _load_module()

    targets = module.build_targets(_state(), claims_url="http://127.0.0.1:3000/claims")

    assert targets["claims_page"] == "http://127.0.0.1:3000/claims"
    assert targets["genesis_lookup"].startswith("http://127.0.0.1:3000/api/sota-claims/")
    assert "subnet_id=genesis" in targets["genesis_lookup"]
    assert "base%3Asota-local" in targets["self_validation_evidence"]
    assert targets["docs_base"] == "http://127.0.0.1:9002/base/"
    assert targets["docs_new_users"] == "http://127.0.0.1:9002/base/new-users/"


def test_build_targets_includes_optional_handoff_url() -> None:
    module = _load_module()
    state = _state()
    state["urls"]["handoff"] = "http://127.0.0.1:9003/"

    targets = module.build_targets(state, claims_url="http://127.0.0.1:3000/claims")

    assert targets["handoff"] == "http://127.0.0.1:9003/"


def test_validate_page_html_requires_local_demo_copy() -> None:
    module = _load_module()

    checks = module.validate_page_html("plain shell")

    assert checks[0]["status"] == "red"
    assert "Load genesis claim" in checks[0]["detail"]


def test_validate_page_html_normalizes_react_comment_boundaries() -> None:
    module = _load_module()
    html = " ".join(text.replace("SOTA Local Base claims", "SOTA Local Base<!-- --> claims") for text in module.EXPECTED_PAGE_TEXT)

    checks = module.validate_page_html(html)

    assert checks[0]["status"] == "green"


def test_validate_docs_htmls_requires_tester_entry_points() -> None:
    module = _load_module()

    checks = module.validate_docs_htmls({"docs_base": "plain docs"})

    assert checks[0]["status"] == "red"
    assert "New user guide" in checks[0]["detail"]


def test_validate_docs_htmls_accepts_required_copy() -> None:
    module = _load_module()
    docs = {name: " ".join(expected) for name, expected in module.EXPECTED_DOC_PAGE_TEXT.items()}

    checks = module.validate_docs_htmls(docs)

    assert checks
    assert {check["status"] for check in checks} == {"green"}


def test_validate_handoff_page_accepts_required_copy(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_http_text",
            lambda url, timeout: (
                "SOTA Base Tester Handoff Local demo ready Local Demo Local-only private key "
                "Add SOTA Local Base network Open claims UI Mined emission claim Self-validation "
                "Peer validators State-changing claim proof Base Sepolia "
                "Base Sepolia is not ready for a nontechnical MetaMask tester yet."
            ),
        )

    checks = module.validate_handoff_page({"handoff": "http://127.0.0.1:9003/"}, request_timeout=1)

    assert checks[0]["status"] == "green"


def test_validate_api_payloads_accepts_complete_local_demo_payloads() -> None:
    module = _load_module()
    state = _state()
    account = "0x3c44cdddb6a900fa2b585dd299e03d12fa4293bc"

    checks = module.validate_api_payloads(
        state,
        genesis_lookup={
            "account": account,
            "eligible": True,
            "credits": {
                "tao": {"raw": "1000000000000000000"},
                "alpha_synthetic": {"raw": "500000000000000000"},
                "total_sota": {"raw": "1500000000000000000"},
            },
        },
        emission_lookup={
            "account": account,
            "eligible": True,
            "credits": {"total_sota": {"raw": "2000000000000000000"}},
        },
        evidence={
            "root": {
                "root": "0x9485d52e9c3f33701e38158e997c9a0d64c07b002126d2a4e5810d63ad268594"
            },
            "bundle": {
                "claim_evidence": [
                    {
                        "evidence": {
                            "self_validation_consensus": {
                                "status": "accepted",
                                "accepted_count": 3,
                                "committee_count": 3,
                                "committee_size": 3,
                                "frontier_gate_passed": True,
                                "quorum_gate_passed": True,
                            }
                        }
                    }
                ]
            },
        },
        genesis_transaction={
            "transaction": {
                "to": "0xdc64a140aa3e981100a9beca4e685f962f0cf6c9",
                "data": "0x1234",
                "chainId": 31337,
            }
        },
        emission_transaction={
            "transaction": {
                "to": "0x5fc8d32690cc91d4c39d9d3abcbd16989f875707",
                "data": "0x5678",
                "chainId": 31337,
            }
        },
        claims_health={"status": "ok", "chain_id": 31337},
        index_status={
            "chain_id": 31337,
            "contracts_configured": [
                "root_registry",
                "lane_registry",
                "genesis_distributor",
                "emission_distributor",
            ],
            "lag_blocks": 0,
            "last_sync_error": None,
        },
        autoresearch_ready_status=204,
    )

    assert {check["status"] for check in checks} == {"green"}

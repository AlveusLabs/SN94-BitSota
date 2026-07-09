from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sota_base_tester_handoff.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sota_base_tester_handoff", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _args(tmp_path: Path, *, environment: str = "both"):
    return argparse.Namespace(
        environment=environment,
        state=tmp_path / "state.json",
        local_report=tmp_path / "local-report.json",
        release_status=tmp_path / "release-status.json",
        json_out=tmp_path / "handoff.json",
        markdown_out=tmp_path / "handoff.md",
        html_out=tmp_path / "handoff.html",
        print_markdown=False,
    )


def _write_inputs(args: argparse.Namespace) -> None:
    claim_proof_report = args.local_report.parent / "local-claim-proof.json"
    miner_swarm_report = args.local_report.parent / "miner-swarm.json"
    blocker_report = args.local_report.parent / "blockers.json"
    funding_report = args.local_report.parent / "funding.json"
    _write_json(
        args.state,
        {
            "chain_id": 31337,
            "urls": {
                "claims_ui": "http://100.0.0.1:3000/claims",
                "docs": "http://100.0.0.1:9002/base/",
                "autoresearch_dashboard": "http://100.0.0.1:8000/dashboard",
                "anvil_rpc": "http://100.0.0.1:8545",
            },
            "sharing": {
                "mode": "http",
                "status": "yellow",
                "wallet_rpc_browser_safe": False,
                "warning": "Tailscale Serve HTTPS is unavailable.",
            },
            "accounts": {"alice_reward": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"},
            "genesis": {
                "old_coldkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "tao_credit": 1000000000000000000,
                "alpha_synthetic_credit": 500000000000000000,
                "amount": 1500000000000000000,
            },
            "emission_onchain": {"amount": 2000000000000000000},
            "autoresearch": {
                "consensus": {
                    "status": "accepted",
                    "accepted_count": 3,
                    "committee_count": 3,
                    "committee_size": 3,
                },
                "emission_root": {"total_amount_units": 2000000000000000000},
                "participants": {
                    "validators": [
                        {"name": "Bob", "hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstT88xwZtWK7q95dYbF"},
                        {"name": "Charlie", "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"},
                        {"name": "Dave", "hotkey": "5DAAnrj7VHTznn2T6V2HxWjZqZPspK7r4sSm3FvdDEAjZykT"},
                    ]
                },
            },
        },
    )
    _write_json(
        args.local_report,
        {
            "schema": "sota-local-claims-ui-smoke/v1",
            "ok": True,
            "status": "green",
            "summary": {"green": 13, "yellow": 1, "red": 0},
        },
    )
    _write_json(
        args.release_status,
        {
            "schema": "sota-base-release-status/v1",
            "ok": False,
            "status": "red",
            "local_stack_ok": True,
            "local_ok": False,
            "local_wallet_ok": False,
            "local_wallet": {
                "ok": False,
                "status": "yellow",
                "message": "tester wallet RPC may be rejected by MetaMask from another computer",
                "next_action": "Relaunch with --share-mode tailscale-https.",
            },
            "local_remote_wallet_ok": False,
            "local_remote_wallet": {
                "ok": False,
                "status": "red",
                "message": "Tailscale HTTPS sharing is not ready for remote MetaMask testing.",
                "next_action": "Enable Tailscale Serve/HTTPS.",
            },
            "local_tailscale_preflight": {
                "path": str(args.local_report.parent / "tailscale-preflight.json"),
                "schema": "sota-local-tailscale-preflight/v1",
                "ok": False,
                "status": "red",
                "message": "Tailscale HTTPS sharing is not ready for remote MetaMask testing.",
                "summary": {"green": 3, "yellow": 2, "red": 1},
                "next_actions": [
                    "Run `sudo tailscale set --operator=$USER` once on this host, then rerun "
                    "the local demo with `--share-mode tailscale-https`."
                ],
            },
            "testnet_ok": False,
            "summary": {"green": 2, "yellow": 0, "red": 3},
            "gates": [
                {
                    "name": "local_demo",
                    "phase": "local",
                    "status": "green",
                    "summary": {"green": 13, "yellow": 1, "red": 0},
                    "path": "/tmp/local-report.json",
                },
                {
                    "name": "local_claim_proof",
                    "phase": "local",
                    "status": "green",
                    "summary": {"green": 8, "yellow": 0, "red": 0},
                    "path": str(claim_proof_report),
                },
                {
                    "name": "local_miner_swarm",
                    "phase": "local",
                    "status": "green",
                    "summary": {"green": 1, "yellow": 0, "red": 0},
                    "path": str(miner_swarm_report),
                },
                {
                    "name": "testnet_blockers",
                    "phase": "base_sepolia",
                    "status": "red",
                    "summary": {"green": 1, "yellow": 0, "red": 12},
                    "path": str(blocker_report),
                },
                {
                    "name": "testnet_funding",
                    "phase": "base_sepolia",
                    "status": "red",
                    "summary": {"green": 2, "yellow": 0, "red": 3},
                    "path": str(funding_report),
                },
            ],
            "blocked_gates": [
                {
                    "name": "testnet_blockers",
                    "phase": "base_sepolia",
                    "status": "red",
                    "next_action": "Clear AWS/DNS blockers.",
                },
                {
                    "name": "testnet_funding",
                    "phase": "base_sepolia",
                    "status": "red",
                    "next_action": "Fund Base Sepolia roles.",
                }
            ],
        },
    )
    _write_json(
        claim_proof_report,
        {
            "schema": "sota-local-claim-proof/v1",
            "ok": True,
            "status": "green",
            "summary": {"green": 8, "yellow": 0, "red": 0},
            "reset_after": True,
        },
    )
    _write_json(
        miner_swarm_report,
        {
            "schema": "sota-local-multi-miner/v1",
            "ok": True,
            "miner_count": 5,
            "accepted_count": 5,
            "matching_claim_count": 5,
            "claim_transactions": [{"tx_hash": "0x" + str(index) * 64} for index in range(5)],
        },
    )
    _write_json(
        blocker_report,
        {
            "schema": "sota-base-testnet-blockers/v1",
            "ok": False,
            "status": "red",
            "checks": [
                {
                    "name": "gas_deployer",
                    "status": "red",
                    "detail": "deployer 0x00000000000000000000000000000000000000aa has 0 ETH on Base Sepolia.",
                    "remediation": "Fund 0x00000000000000000000000000000000000000aa with Base Sepolia ETH before deployment/browser smoke.",
                },
                {
                    "name": "base_sepolia_rpc",
                    "status": "green",
                    "detail": "RPC returned Base Sepolia chain id 84532.",
                },
            ],
        },
    )
    _write_json(
        funding_report,
        {
            "schema": "sota-base-testnet-funding/v1",
            "ok": False,
            "status": "red",
            "funding_targets": [
                {
                    "label": "deployer",
                    "status": "red",
                    "address": "0x00000000000000000000000000000000000000aa",
                    "balance_eth": "0.00000000",
                    "minimum_balance_eth": "0.02000000",
                    "needed_eth": "0.02000000",
                    "explorer_url": "https://sepolia.basescan.org/address/0x00000000000000000000000000000000000000aa",
                    "remediation": "Fund the deployer.",
                }
            ],
            "faucet_sources": [
                {
                    "name": "Base network faucets",
                    "url": "https://docs.base.org/base-chain/network-information/network-faucets",
                    "note": "Use native Base Sepolia ETH.",
                }
            ],
        },
    )


def test_tester_handoff_contains_local_urls_and_warning(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_inputs(args)

    handoff = module.build_handoff(args)
    markdown = module.render_markdown(handoff)
    html = module.render_html(handoff)

    assert handoff["schema"] == "sota-base-tester-handoff/v1"
    assert handoff["release_status"]["local_stack_ok"] is True
    assert handoff["release_status"]["local_ok"] is False
    assert handoff["release_status"]["local_wallet_ok"] is False
    assert handoff["release_status"]["local_wallet"]["status"] == "yellow"
    assert handoff["release_status"]["local_remote_wallet_ok"] is False
    assert handoff["release_status"]["local_remote_wallet"]["status"] == "red"
    assert handoff["release_status"]["local_tailscale_preflight"]["status"] == "red"
    assert handoff["local"]["ready"] is False
    assert handoff["local"]["status"] == "green"
    assert handoff["local"]["claims_ui_url"] == "http://100.0.0.1:3000/claims"
    assert handoff["local"]["chain_id_hex"] == "0x7a69"
    assert handoff["local"]["share_mode"] == "http"
    assert handoff["local"]["wallet_rpc_browser_safe"] is False
    assert [item["name"] for item in handoff["local"]["local_gates"]] == ["local_demo", "local_claim_proof", "local_miner_swarm"]
    assert handoff["local"]["smoke_status"] == "green"
    assert handoff["local"]["claim_proof_status"] == "green"
    assert handoff["local"]["claim_proof_report"] == str(args.local_report.parent / "local-claim-proof.json")
    assert "archived pre-reset evidence" in handoff["local"]["claim_proof_scope"]
    assert handoff["local"]["miner_swarm_status"] == "green"
    assert handoff["local"]["miner_swarm_miner_count"] == 5
    assert handoff["local"]["miner_swarm_accepted_count"] == 5
    assert handoff["local"]["miner_swarm_claim_tx_count"] == 5
    assert handoff["local"]["local_only_private_key"].startswith("0x5de411")
    assert handoff["local"]["genesis_claim_amount"] == "1.5 SOTA"
    assert handoff["local"]["emission_claim_amount"] == "2 SOTA"
    assert handoff["local"]["expected_final_balance"] == "3.5 SOTA"
    assert handoff["local"]["manual_metamask_fields"]["chain_id"] == "31337"
    assert handoff["local"]["self_validation_status"] == "accepted"
    assert handoff["local"]["self_validation_summary"] == "3/3 accepted"
    assert handoff["testnet"]["immediate_blockers"][0]["name"] == "gas_deployer"
    assert handoff["testnet"]["funding_targets"][0]["label"] == "deployer"
    assert handoff["testnet"]["funding_targets"][0]["needed_eth"] == "0.02000000"
    assert handoff["testnet"]["faucet_sources"][0]["name"] == "Base network faucets"
    assert [item["name"] for item in handoff["local"]["peer_validators"]] == ["Bob", "Charlie", "Dave"]
    assert "Never paste a real seed phrase" in markdown
    assert "A real holder may sign the binding with the snapshot coldkey" in markdown
    assert "production Bittensor wallets" not in markdown
    assert "Local stack ready: true" in markdown
    assert "Local ready: false" in markdown
    assert "Local MetaMask ready: false" in markdown
    assert "Local MetaMask detail: tester wallet RPC may be rejected" in markdown
    assert "Remote Tailscale MetaMask ready: false" in markdown
    assert "Remote Tailscale MetaMask detail: Tailscale HTTPS sharing is not ready" in markdown
    assert "Remote Tailscale MetaMask next action: Enable Tailscale Serve/HTTPS." in markdown
    assert "Tailscale preflight: red" in markdown
    assert "Tailscale next action: Run `sudo tailscale set --operator=$USER`" in markdown
    assert "Tester Decision" in markdown
    assert "Local same-machine: false" not in markdown
    assert "Local-only private key" in markdown
    assert "State-changing claim proof: green" in markdown
    assert "Local miner swarm: green (5 miners, 5 accepted, 5 claim txs)" in markdown
    assert f"Claim proof report: {args.local_report.parent / 'local-claim-proof.json'}" in markdown
    assert "archived pre-reset evidence" in markdown
    assert "Genesis claim amount: 1.5 SOTA" in markdown
    assert "Mined emission claim amount: 2 SOTA" in markdown
    assert "Expected final local SOTA balance after both claims: 3.5 SOTA" in markdown
    assert "Manual MetaMask Network Fields" in markdown
    assert "Local Evidence To Send Back" in markdown
    assert "Operator Only: Local Claim Evidence Command" in markdown
    assert "Block explorer URL: leave blank" in markdown
    assert "Peer validators: Bob" in markdown
    assert "Base Sepolia infrastructure is not ready" in markdown
    assert "Immediate Base Sepolia Blockers" in markdown
    assert "gas_deployer" in markdown
    assert "has 0 ETH on Base Sepolia" in markdown
    assert "Base Sepolia Funding Targets" in markdown
    assert "needs 0.02000000 ETH" in markdown
    assert "Base Sepolia Faucet Sources" in markdown
    assert "<title>SOTA Base Tester Handoff</title>" in html
    assert "SOTA SOTA" not in html
    assert "Local demo blocked" in html
    assert "Local MetaMask ready" in html
    assert "Remote Tailscale MetaMask ready" in html
    assert "Tailscale preflight" in html
    assert "Remote Tailscale MetaMask not ready" in html
    assert "Next action: Enable Tailscale Serve/HTTPS." in html
    assert "Aggregate status: red" not in html
    assert "http://100.0.0.1:3000/claims" in html
    assert "Add SOTA Local Base network" in html
    assert "Wallet RPC browser-safe: false" in html
    assert "Copy local-only key" in html
    assert "Open autoresearch dashboard" in html
    assert "wallet_addEthereumChain" in html
    assert "Self-validation" in html
    assert "Peer validators" in html
    assert "State-changing claim proof" in html
    assert "Local miner swarm" in html
    assert "Expected final balance" in html
    assert "Manual MetaMask Network Fields" in html
    assert "Immediate Base Sepolia Blockers" in html
    assert "gas_deployer" in html
    assert "Base Sepolia Funding Targets" in html
    assert "Copy address" in html
    assert "Base Sepolia Faucet Sources" in html
    assert str(args.local_report.parent / "local-claim-proof.json") in html
    assert "archived pre-reset evidence" in html


def test_tester_handoff_marks_testnet_ready_for_claim_tester_when_only_tx_evidence_is_red(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_inputs(args)
    artifacts_dir = args.release_status.parent
    release = json.loads(args.release_status.read_text(encoding="utf-8"))
    release["testnet_ok"] = False
    release["gates"] = [
        gate
        for gate in release["gates"]
        if gate["phase"] == "local"
    ] + [
        {
            "name": "testnet_operator_run",
            "phase": "base_sepolia",
            "status": "green",
            "required": True,
            "summary": {"green": 14, "yellow": 0, "red": 0},
            "path": str(artifacts_dir / "operator.json"),
        },
        {
            "name": "testnet_blockers",
            "phase": "base_sepolia",
            "status": "green",
            "required": True,
            "summary": {"green": 15, "yellow": 0, "red": 0},
            "path": str(artifacts_dir / "blockers-green.json"),
        },
        {
            "name": "testnet_funding",
            "phase": "base_sepolia",
            "status": "green",
            "required": True,
            "summary": {"green": 5, "yellow": 0, "red": 0},
            "path": str(artifacts_dir / "funding-green.json"),
        },
        {
            "name": "testnet_container_pack",
            "phase": "base_sepolia",
            "status": "yellow",
            "required": False,
            "summary": {"green": 3, "yellow": 1, "red": 0},
            "path": str(artifacts_dir / "container.json"),
        },
        {
            "name": "testnet_browser_smoke",
            "phase": "base_sepolia",
            "status": "green",
            "required": True,
            "summary": {"green": 21, "yellow": 0, "red": 0},
            "path": str(artifacts_dir / "browser.json"),
        },
        {
            "name": "claim_tx_evidence",
            "phase": "base_sepolia",
            "status": "red",
            "required": True,
            "summary": {"green": 4, "yellow": 0, "red": 5},
            "path": str(artifacts_dir / "base-sota-claim-tx-evidence.json"),
        },
    ]
    release["blocked_gates"] = [
        {
            "name": "claim_tx_evidence",
            "phase": "base_sepolia",
            "status": "red",
            "next_action": "Submit both MetaMask claims and rerun evidence verification.",
        }
    ]
    _write_json(args.release_status, release)
    _write_json(artifacts_dir / "blockers-green.json", {"checks": []})
    _write_json(
        artifacts_dir / "funding-green.json",
        {
            "funding_targets": [
                {
                    "label": "test_wallet",
                    "status": "green",
                    "address": "0xE93daE9Bb94aa2f2abA57C7CadEC822b800461Fc",
                    "balance_eth": "0.03999904",
                    "minimum_balance_eth": "0.00500000",
                    "needed_eth": "0.00000000",
                }
            ],
            "faucet_sources": [],
        },
    )
    _write_json(
        artifacts_dir / "base-sota-testnet-seed-artifacts-finalized.json",
        {
            "seeded_claims": {
                "test_wallet_address": "0xe93dae9bb94aa2f2aba57c7cadec822b800461fc",
                "test_old_coldkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "lane_id": "base:sota-local",
                "epoch": 2,
                "genesis_total_units": "1500000000000000000",
                "emission_total_units": "2000000000000000000",
            },
            "root_ids": {
                "genesis": "0x" + "11" * 32,
                "emission": "0x" + "22" * 32,
            },
        },
    )
    _write_json(
        artifacts_dir / "base-sepolia-deployment-manifest.json",
        {
            "services": {
                "claims_ui": {
                    "public_url": "https://claims.example.test",
                    "browser_safe_env": {
                        "NEXT_PUBLIC_SOTA_READINESS_URL": "https://claims.example.test/base-sota-testnet-readiness.json"
                    },
                },
                "indexer_api": {"public_base_url": "https://api.example.test"},
            }
        },
    )
    _write_json(
        artifacts_dir / "base-sota-testnet-browser-smoke.json",
        {"status": "green", "summary": {"green": 21, "yellow": 0, "red": 0}},
    )
    _write_json(
        artifacts_dir / "base-sota-fresh-testnet-tester.json",
        {
            "status": "green",
            "ok": True,
            "reward_address": "0xe93dae9bb94aa2f2aba57c7cadec822b800461fc",
            "reward_key_file": str(artifacts_dir / "fresh-claim-wallet.json"),
            "private_key_printed": False,
            "epoch": 2,
            "funding": {
                "status": "green",
                "tx_hash": "0x" + "33" * 32,
                "balance_after_eth": "0.00500000",
            },
            "next_action": "Open the public Base Sepolia claims UI with this wallet.",
        },
    )

    handoff = module.build_handoff(args)
    markdown = module.render_markdown(handoff)
    html = module.render_html(handoff)

    assert handoff["testnet"]["ready"] is True
    assert handoff["testnet"]["release_ready"] is False
    assert handoff["testnet"]["claims_ui_url"] == "https://claims.example.test/claims"
    assert handoff["testnet"]["test_wallet_address"] == "0xe93dae9bb94aa2f2aba57c7cadec822b800461fc"
    assert handoff["testnet"]["genesis_claim_amount"] == "1.5 SOTA"
    assert handoff["testnet"]["emission_claim_amount"] == "2 SOTA"
    assert handoff["testnet"]["fresh_tester"]["status"] == "green"
    assert handoff["testnet"]["fresh_tester"]["private_key_printed"] is False
    assert handoff["testnet"]["fresh_tester"]["reward_key_file"] == str(artifacts_dir / "fresh-claim-wallet.json")
    assert "Base Sepolia is ready for a MetaMask claim tester" in markdown
    assert "operator-provided seeded Base Sepolia test wallet" in markdown
    assert "Base Sepolia claim test: ready only with the operator-provided seeded wallet" in markdown
    assert "Base Sepolia MetaMask Network Fields" in markdown
    assert "Chain ID: 84532" in markdown
    assert "Expected selected account: `0xe93dae9bb94aa2f2aba57c7cadec822b800461fc`" in markdown
    assert "Expected final SOTA balance after both claims: 3.5 SOTA" in markdown
    assert "Fresh Tester Prep" in markdown
    assert "Operator-only wallet key file" in markdown
    assert "Private key printed by prep command: false" in markdown
    assert "Base Sepolia Evidence To Send Back" in markdown
    assert "Remaining Evidence Gate" in markdown
    assert "BASE_SEPOLIA_GENESIS_TX_HASH" in markdown
    assert "Operator Only: Refresh Release/Handoff After Evidence" in markdown
    assert "https://claims.example.test/claims" in html
    assert "Testnet wallet access" in html
    assert "Tester Decision" in html
    assert "Base Sepolia MetaMask Network Fields" in html
    assert "Fresh Tester Prep" in html
    assert "Private key printed by prep command: false" in html
    assert "Base Sepolia Evidence To Send Back" in html
    assert "Copy testnet wallet" in html
    assert "Remaining Evidence Gate" in html

    _write_json(
        artifacts_dir / "base-sota-submitted-claim-txs.json",
        {
            "schema": "sota-base-submitted-claim-txs/v1",
            "wallet_address": "0xE93daE9Bb94aa2f2abA57C7CadEC822b800461Fc",
            "transactions": [
                {"program": "genesis", "tx_hash": "0x" + "44" * 32},
                {"program": "emission", "tx_hash": "0x" + "55" * 32},
            ],
        },
    )
    stale_handoff = module.build_handoff(args)
    stale_markdown = module.render_markdown(stale_handoff)
    stale_html = module.render_html(stale_handoff)

    assert stale_handoff["testnet"]["fresh_tester"]["status"] == "claimed"
    assert stale_handoff["testnet"]["fresh_tester"]["ok"] is False
    assert stale_handoff["testnet"]["fresh_tester"]["already_claimed"] is True
    assert stale_handoff["testnet"]["fresh_tester"]["claimed_programs"] == ["emission", "genesis"]
    assert "Already claimed by this seeded wallet: emission, genesis" in stale_markdown
    assert "Prepare a new real signed snapshot binding and fresh root cycle" in stale_markdown
    assert "Already claimed by this seeded wallet: emission, genesis" in stale_html


def test_tester_handoff_testnet_only_omits_local_private_key(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, environment="testnet")
    _write_inputs(args)

    handoff = module.build_handoff(args)
    markdown = module.render_markdown(handoff)

    assert "local" not in handoff
    assert "testnet" in handoff
    assert module.LOCAL_PRIVATE_KEY not in markdown
    assert handoff["testnet"]["blocked_gates"][0]["name"] == "testnet_blockers"


def test_tester_handoff_deferred_holder_uses_own_wallet_language(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, environment="testnet")
    artifacts_dir = args.release_status.parent
    seeded_wallet = "0x13761911b9be377680a664fa0cd153864a2878bc"
    _write_json(
        args.release_status,
        {
            "schema": "sota-base-release-status/v1",
            "ok": True,
            "status": "green",
            "testnet_ok": True,
            "real_holder_test_deferred": True,
            "local_stack_ok": True,
            "local_ok": True,
            "local_wallet_ok": True,
            "local_remote_wallet_ok": False,
            "summary": {"green": 3, "yellow": 0, "red": 0},
            "gates": [
                {
                    "name": "testnet_operator_run",
                    "phase": "base_sepolia",
                    "status": "green",
                    "required": True,
                    "summary": {"green": 1, "yellow": 0, "red": 0},
                    "message": "scheduled genesis and emission publishers are active and idle/ready",
                },
                {
                    "name": "claim_tx_evidence",
                    "phase": "base_sepolia",
                    "status": "green",
                    "required": True,
                    "summary": {"green": 1, "yellow": 0, "red": 0},
                },
            ],
            "blocked_gates": [],
        },
    )
    _write_json(
        artifacts_dir / "base-sota-testnet-seed-artifacts-finalized.json",
        {
            "seeded_claims": {
                "test_wallet_address": seeded_wallet,
                "test_old_coldkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
                "lane_id": "base:sota-local",
                "epoch": 6,
                "genesis_total_units": "1500000000000000000",
                "emission_total_units": "7200000000000000000000",
            },
            "root_ids": {"genesis": "0x" + "11" * 32, "emission": "0x" + "22" * 32},
        },
    )
    _write_json(
        artifacts_dir / "base-sepolia-deployment-manifest.json",
        {
            "services": {
                "claims_ui": {"public_url": "https://claims.example.test"},
                "indexer_api": {"public_base_url": "https://api.example.test"},
            }
        },
    )
    _write_json(
        artifacts_dir / "base-sota-testnet-browser-smoke.json",
        {"status": "green", "summary": {"green": 27, "yellow": 0, "red": 0}},
    )

    handoff = module.build_handoff(args)
    markdown = module.render_markdown(handoff)
    html = module.render_html(handoff)

    assert handoff["testnet"]["ready"] is True
    assert "Seeded evidence wallet" in markdown
    assert "use your own Base Sepolia wallet for a real holder test" in markdown
    assert "Expected selected account: your own Base Sepolia wallet with test ETH" in markdown
    assert f"Expected selected account: `{seeded_wallet}`" not in markdown
    assert "Seeded evidence wallet" in html
    assert "Expected selected account: your own Base Sepolia wallet with test ETH" in html
    assert f"Expected selected account: {seeded_wallet}" not in html
    assert "Copy evidence wallet" in html


def test_tester_handoff_local_not_ready_when_claim_proof_gate_is_red(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path, environment="local")
    _write_inputs(args)
    release = json.loads(args.release_status.read_text(encoding="utf-8"))
    release["local_ok"] = False
    release["summary"] = {"green": 1, "yellow": 0, "red": 1}
    for gate in release["gates"]:
        if gate["name"] == "local_claim_proof":
            gate["status"] = "red"
            gate["summary"] = {"green": 0, "yellow": 0, "red": 1}
    _write_json(args.release_status, release)

    handoff = module.build_handoff(args)

    assert handoff["local"]["ready"] is False
    assert handoff["local"]["status"] == "red"
    assert handoff["local"]["smoke_status"] == "green"
    assert handoff["local"]["claim_proof_status"] == "red"


def test_tester_handoff_main_writes_json_and_markdown(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_inputs(args)

    exit_code = module.main(
        [
            "--state",
            str(args.state),
            "--local-report",
            str(args.local_report),
            "--release-status",
            str(args.release_status),
            "--json-out",
            str(args.json_out),
            "--markdown-out",
            str(args.markdown_out),
            "--html-out",
            str(args.html_out),
        ]
    )

    assert exit_code == 0
    assert args.json_out.exists()
    assert args.markdown_out.exists()
    assert args.html_out.exists()
    assert "SOTA Base Tester Handoff" in args.markdown_out.read_text(encoding="utf-8")
    assert "SOTA Base Tester Handoff" in args.html_out.read_text(encoding="utf-8")


def test_tester_handoff_default_outputs_refresh_local_served_copy(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_inputs(args)
    local_handoff_dir = tmp_path / ".sota-base-local" / "handoff"
    testnet_dir = tmp_path / ".sota-base-testnet"
    monkeypatch.setattr(module, "LOCAL_HANDOFF_DIR", local_handoff_dir)
    monkeypatch.setattr(module, "TESTNET_RUN_DIR", testnet_dir)

    exit_code = module.main(
        [
            "--state",
            str(args.state),
            "--local-report",
            str(args.local_report),
            "--release-status",
            str(args.release_status),
        ]
    )

    assert exit_code == 0
    assert (testnet_dir / "base-sota-tester-handoff.json").exists()
    assert (testnet_dir / "base-sota-tester-handoff.md").exists()
    assert (testnet_dir / "base-sota-tester-handoff.html").exists()
    assert (local_handoff_dir / "handoff.json").exists()
    assert (local_handoff_dir / "handoff.md").exists()
    assert (local_handoff_dir / "index.html").exists()
    local_html = (local_handoff_dir / "index.html").read_text(encoding="utf-8")
    assert "Local demo blocked" in local_html
    assert "local demo only" in local_html
    assert "Base Sepolia" not in local_html


def test_tester_handoff_can_mirror_local_with_explicit_outputs(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    args = _args(tmp_path)
    _write_inputs(args)
    local_handoff_dir = tmp_path / ".sota-base-local" / "handoff"
    monkeypatch.setattr(module, "LOCAL_HANDOFF_DIR", local_handoff_dir)

    exit_code = module.main(
        [
            "--state",
            str(args.state),
            "--local-report",
            str(args.local_report),
            "--release-status",
            str(args.release_status),
            "--json-out",
            str(args.json_out),
            "--markdown-out",
            str(args.markdown_out),
            "--html-out",
            str(args.html_out),
            "--mirror-local",
        ]
    )

    assert exit_code == 0
    assert args.html_out.exists()
    assert (local_handoff_dir / "index.html").exists()
    local_html = (local_handoff_dir / "index.html").read_text(encoding="utf-8")
    assert "Local demo blocked" in local_html
    assert "Base Sepolia" not in local_html

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any


REPOS = Path("/home/mekaneeky/repos")
LOCAL_RUN_DIR = REPOS / ".sota-base-local"
TESTNET_RUN_DIR = REPOS / ".sota-base-testnet"
LOCAL_HANDOFF_DIR = LOCAL_RUN_DIR / "handoff"
LOCAL_PRIVATE_KEY = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
CLAIM_TX_EVIDENCE_GATE = "claim_tx_evidence"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _summary_text(summary: dict[str, Any] | None) -> str:
    summary = summary or {}
    return f"{int(summary.get('green') or 0)} green / {int(summary.get('yellow') or 0)} yellow / {int(summary.get('red') or 0)} red"


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(status, 2)


def _worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "unknown"
    return max(statuses, key=_status_rank)


def _format_sota_units(value: Any) -> str:
    try:
        raw = int(value or 0)
    except (TypeError, ValueError):
        raw = 0
    whole, fraction = divmod(raw, 10**18)
    if fraction == 0:
        return f"{whole} SOTA"
    fraction_text = str(fraction).rjust(18, "0").rstrip("0")
    return f"{whole}.{fraction_text} SOTA"


def _append_claims_path(url: str) -> str:
    url = str(url or "").strip().rstrip("/")
    if not url:
        return ""
    return url if url.endswith("/claims") else f"{url}/claims"


def _funding_target(funding_targets: list[dict[str, str]], label: str) -> dict[str, str]:
    for target in funding_targets:
        if str(target.get("label") or "") == label:
            return target
    return {}


def _local_section(state: dict[str, Any], release: dict[str, Any], local_report: dict[str, Any]) -> dict[str, Any]:
    urls = dict(state.get("urls") or {})
    sharing = dict(state.get("sharing") or {})
    genesis = dict(state.get("genesis") or {})
    accounts = dict(state.get("accounts") or {})
    autoresearch = dict(state.get("autoresearch") or {})
    consensus = dict(autoresearch.get("consensus") or {})
    participants = dict(autoresearch.get("participants") or {})
    peer_validators = [
        {
            "name": str(dict(item).get("name") or "validator"),
            "hotkey": str(dict(item).get("hotkey") or ""),
        }
        for item in list(participants.get("validators") or [])
        if isinstance(item, dict)
    ]
    emission_root = dict(autoresearch.get("emission_root") or {})
    emission_onchain = dict(state.get("emission_onchain") or {})
    local_gates = [
        dict(gate)
        for gate in list(release.get("gates") or [])
        if isinstance(gate, dict) and dict(gate).get("phase") == "local"
    ]
    smoke_gate = next((gate for gate in local_gates if gate.get("name") == "local_demo"), {})
    claim_proof_gate = next((gate for gate in local_gates if gate.get("name") == "local_claim_proof"), {})
    claim_proof_payload = {}
    if claim_proof_gate.get("path"):
        claim_proof_payload = _load_json(Path(str(claim_proof_gate.get("path"))))
    claim_proof_reset_after = bool(claim_proof_payload.get("reset_after"))
    claim_proof_scope = (
        "archived pre-reset evidence; the current local stack was reset for the next tester"
        if claim_proof_reset_after
        else "current local stack evidence"
    )
    local_ok = bool(release.get("local_ok")) if release else bool(local_report.get("ok"))
    local_status = _worst_status([str(gate.get("status") or "red") for gate in local_gates])
    if not local_gates:
        local_status = str(local_report.get("status") or ("green" if local_ok else "red"))
    chain_id = int(state.get("chain_id") or 31337)
    emission_amount = emission_onchain.get("amount") or emission_root.get("total_amount_units") or 0
    return {
        "ready": local_ok,
        "status": local_status,
        "summary": local_report.get("summary") or {},
        "local_gates": [
            {
                "name": gate.get("name"),
                "status": gate.get("status"),
                "summary": gate.get("summary") or {},
                "path": gate.get("path"),
            }
            for gate in local_gates
        ],
        "smoke_status": smoke_gate.get("status") or local_report.get("status") or "unknown",
        "smoke_summary": smoke_gate.get("summary") or local_report.get("summary") or {},
        "smoke_report": smoke_gate.get("path") or "",
        "claim_proof_status": claim_proof_gate.get("status") or "missing",
        "claim_proof_summary": claim_proof_gate.get("summary") or {},
        "claim_proof_report": claim_proof_gate.get("path") or str(LOCAL_RUN_DIR / "claim-proof" / "latest.json"),
        "claim_proof_scope": claim_proof_scope,
        "claim_proof_reset_after": claim_proof_reset_after,
        "claims_ui_url": urls.get("claims_ui") or "",
        "docs_url": urls.get("docs") or "",
        "autoresearch_dashboard_url": urls.get("autoresearch_dashboard") or "",
        "anvil_rpc_url": urls.get("anvil_rpc") or "",
        "share_mode": sharing.get("mode") or "",
        "share_status": sharing.get("status") or "",
        "share_warning": sharing.get("warning") or "",
        "wallet_rpc_browser_safe": bool(sharing.get("wallet_rpc_browser_safe")),
        "tailscale_dns_name": sharing.get("tailscale_dns_name") or "",
        "chain_id": chain_id,
        "chain_id_hex": hex(chain_id),
        "network_name": "SOTA Local Base",
        "native_currency_symbol": "ETH",
        "wallet_address": accounts.get("alice_reward") or genesis.get("reward_address") or "",
        "local_only_private_key": LOCAL_PRIVATE_KEY,
        "old_coldkey": genesis.get("old_coldkey") or "",
        "genesis_claim_amount": _format_sota_units(genesis.get("amount")),
        "genesis_tao_credit": _format_sota_units(genesis.get("tao_credit")),
        "genesis_alpha_credit": _format_sota_units(genesis.get("alpha_synthetic_credit")),
        "emission_claim_amount": _format_sota_units(emission_amount),
        "self_validation_status": consensus.get("status") or "",
        "self_validation_accepted_count": int(consensus.get("accepted_count") or 0),
        "self_validation_committee_count": int(consensus.get("committee_count") or consensus.get("committee_size") or 0),
        "self_validation_committee_size": int(consensus.get("committee_size") or 0),
        "self_validation_summary": (
            f"{int(consensus.get('accepted_count') or 0)}/"
            f"{int(consensus.get('committee_count') or consensus.get('committee_size') or 0)} accepted"
        ),
        "peer_validators": peer_validators,
        "expected_flow": [
            "Import the local-only private key into a throwaway MetaMask profile.",
            "Click Add SOTA Local Base network, or add the Anvil RPC URL manually with chain id 31337.",
            "Open the claims UI URL and connect the imported wallet.",
            "Confirm the Local readiness panel is green.",
            "Click Load genesis claim, then Claim. This claims local SOTA based on seeded TAO plus alpha accounting credit; TAO and alpha are not transferred.",
            "Click Load mined emission and confirm Mining and self-validation shows accepted consensus from the other local peer validators.",
            "Click Claim for the mined emission.",
            "Confirm the SOTA balance increases after each claim.",
        ],
        "manual_evidence_checklist": [
            "Record the genesis claim transaction hash shown after MetaMask confirms.",
            "Record the mined emission claim transaction hash shown after MetaMask confirms.",
            "Confirm both claims show claimed or an increased SOTA balance in the claims UI.",
            "Run the local claim transaction evidence verifier with both hashes.",
        ],
        "local_tx_evidence_command": (
            "python3 scripts/sota_base_claim_tx_evidence.py --environment local "
            "--state /home/mekaneeky/repos/.sota-base-local/state.json "
            '--genesis-tx "$LOCAL_GENESIS_TX_HASH" '
            '--emission-tx "$LOCAL_EMISSION_TX_HASH" '
            "--report-out /home/mekaneeky/repos/.sota-base-local/claim-proof/manual-claim-tx-evidence.json"
        ),
    }


def _required_testnet_infra_ready(testnet_gates: list[dict[str, Any]]) -> bool:
    required_infra = []
    for gate in testnet_gates:
        name = str(dict(gate).get("name") or "")
        if name == CLAIM_TX_EVIDENCE_GATE:
            continue
        required = bool(dict(gate).get("required", name != "testnet_container_pack"))
        if required:
            required_infra.append(gate)
    return bool(required_infra) and all(str(dict(gate).get("status") or "red") == "green" for gate in required_infra)


def _testnet_section(release: dict[str, Any], testnet_dir: Path = TESTNET_RUN_DIR) -> dict[str, Any]:
    gates = list(release.get("gates") or [])
    blocked = list(release.get("blocked_gates") or [])
    testnet_gates = [gate for gate in gates if dict(gate).get("phase") == "base_sepolia"]
    claim_tester_ready = _required_testnet_infra_ready(testnet_gates)
    blocker_gate = next((dict(gate) for gate in testnet_gates if dict(gate).get("name") == "testnet_blockers"), {})
    funding_gate = next((dict(gate) for gate in testnet_gates if dict(gate).get("name") == "testnet_funding"), {})
    seed_report = _load_json(testnet_dir / "base-sota-testnet-seed-artifacts-finalized.json")
    manifest = _load_json(testnet_dir / "base-sepolia-deployment-manifest.json")
    env = _load_env(testnet_dir / "base-sota.env.testnet")
    browser_smoke = _load_json(testnet_dir / "base-sota-testnet-browser-smoke.json")
    services = dict(manifest.get("services") or {})
    claims_ui_service = dict(services.get("claims_ui") or {})
    claims_api_service = dict(services.get("indexer_api") or {})
    seeded_claims = dict(seed_report.get("seeded_claims") or {})
    root_ids = dict(seed_report.get("root_ids") or {})
    immediate_blockers: list[dict[str, str]] = []
    if blocker_gate.get("path"):
        blocker_report = _load_json(Path(str(blocker_gate.get("path"))))
        for check in blocker_report.get("checks") or []:
            if not isinstance(check, dict) or check.get("status") == "green":
                continue
            immediate_blockers.append(
                {
                    "name": str(check.get("name") or ""),
                    "status": str(check.get("status") or "unknown"),
                    "detail": str(check.get("detail") or ""),
                    "remediation": str(check.get("remediation") or ""),
                }
            )
    funding_targets: list[dict[str, str]] = []
    faucet_sources: list[dict[str, str]] = []
    if funding_gate.get("path"):
        funding_report = _load_json(Path(str(funding_gate.get("path"))))
        for target in funding_report.get("funding_targets") or []:
            if not isinstance(target, dict):
                continue
            funding_targets.append(
                {
                    "label": str(target.get("label") or ""),
                    "status": str(target.get("status") or "unknown"),
                    "address": str(target.get("address") or ""),
                    "balance_eth": str(target.get("balance_eth") or ""),
                    "minimum_balance_eth": str(target.get("minimum_balance_eth") or ""),
                    "needed_eth": str(target.get("needed_eth") or ""),
                    "explorer_url": str(target.get("explorer_url") or ""),
                    "remediation": str(target.get("remediation") or ""),
                }
            )
        for source in funding_report.get("faucet_sources") or []:
            if not isinstance(source, dict):
                continue
            faucet_sources.append(
                {
                    "name": str(source.get("name") or ""),
                    "url": str(source.get("url") or ""),
                    "note": str(source.get("note") or ""),
                }
            )
    test_wallet = (
        str(seeded_claims.get("test_wallet_address") or "")
        or str(_funding_target(funding_targets, "test_wallet").get("address") or "")
    )
    claims_ui_url = _append_claims_path(
        env.get("SOTA_CLAIMS_UI_URL")
        or env.get("NEXT_PUBLIC_SOTA_CLAIMS_UI_URL")
        or str(claims_ui_service.get("public_url") or "")
    )
    claims_api_url = (
        env.get("SOTA_CLAIMS_API_URL")
        or env.get("NEXT_PUBLIC_SOTA_CLAIMS_API_URL")
        or str(claims_api_service.get("public_base_url") or "")
    )
    readiness_url = (
        env.get("NEXT_PUBLIC_SOTA_READINESS_URL")
        or str(claims_ui_service.get("browser_safe_env", {}).get("NEXT_PUBLIC_SOTA_READINESS_URL") or "")
    )
    evidence_command = (
        "python3 scripts/sota_base_claim_tx_evidence.py --environment testnet "
        f"--artifacts-dir {testnet_dir} "
        '--genesis-tx "$BASE_SEPOLIA_GENESIS_TX_HASH" '
        '--emission-tx "$BASE_SEPOLIA_EMISSION_TX_HASH" '
        f"--report-out {testnet_dir / 'base-sota-claim-tx-evidence.json'}"
    )
    post_evidence_refresh_command = (
        "python3 scripts/sota_base_release_status.py "
        f"--testnet-artifacts-dir {testnet_dir} "
        f"--report-out {testnet_dir / 'base-sota-release-status.json'} && "
        "python3 scripts/sota_base_tester_handoff.py --environment both "
        f"--release-status {testnet_dir / 'base-sota-release-status.json'} --mirror-local"
    )
    claim_gate = next((dict(gate) for gate in testnet_gates if dict(gate).get("name") == CLAIM_TX_EVIDENCE_GATE), {})
    return {
        "ready": claim_tester_ready,
        "release_ready": bool(release.get("testnet_ok")),
        "status": "green" if claim_tester_ready else "red",
        "release_status": "green" if release.get("testnet_ok") else "red",
        "claims_ui_url": claims_ui_url,
        "claims_api_url": claims_api_url,
        "readiness_url": readiness_url,
        "browser_smoke_status": str(browser_smoke.get("status") or "missing"),
        "browser_smoke_summary": browser_smoke.get("summary") or {},
        "test_wallet_address": test_wallet,
        "old_coldkey": str(seeded_claims.get("test_old_coldkey") or ""),
        "lane_id": str(seeded_claims.get("lane_id") or ""),
        "epoch": str(seeded_claims.get("epoch") or ""),
        "genesis_claim_amount": _format_sota_units(seeded_claims.get("genesis_total_units")),
        "emission_claim_amount": _format_sota_units(seeded_claims.get("emission_total_units")),
        "genesis_root_id": str(root_ids.get("genesis") or ""),
        "emission_root_id": str(root_ids.get("emission") or ""),
        "claim_tx_evidence_status": str(claim_gate.get("status") or "missing"),
        "claim_tx_evidence_command": evidence_command,
        "post_evidence_refresh_command": post_evidence_refresh_command,
        "gates": [
            {
                "name": dict(gate).get("name"),
                "status": dict(gate).get("status"),
                "summary": dict(gate).get("summary") or {},
                "path": dict(gate).get("path"),
            }
            for gate in testnet_gates
        ],
        "blocked_gates": [
            {
                "name": dict(gate).get("name"),
                "status": dict(gate).get("status"),
                "next_action": dict(gate).get("next_action"),
            }
            for gate in blocked
            if dict(gate).get("phase") == "base_sepolia"
        ],
        "immediate_blockers": immediate_blockers,
        "funding_targets": funding_targets,
        "faucet_sources": faucet_sources,
        "tester_message": (
            "Base Sepolia is fully verified, including human MetaMask claim transaction evidence."
            if release.get("testnet_ok")
            else "Base Sepolia is ready for a MetaMask claim tester; full release turns green after the two claim transaction hashes verify."
            if claim_tester_ready
            else "Base Sepolia infrastructure is not ready for a nontechnical MetaMask tester yet."
        ),
        "expected_flow_when_ready": [
            "Open the public Base Sepolia claims URL.",
            "Connect the seeded test MetaMask wallet.",
            "Switch MetaMask to Base Sepolia.",
            "Submit the genesis claim and copy the transaction hash from MetaMask activity or Basescan.",
            "Submit the mined emission claim and copy the transaction hash.",
            "Run the claim transaction evidence verifier against both hashes.",
            "Refresh release status and this handoff after the verifier is green.",
        ]
        if claim_tester_ready
        else [
            "Clear the infrastructure blockers listed above.",
            "Rerun the Base Sepolia operator and browser smoke.",
            "Return to this handoff when the claim tester readiness is green.",
        ],
    }


def build_handoff(args: argparse.Namespace) -> dict[str, Any]:
    state = _load_json(args.state)
    release = _load_json(args.release_status)
    local_report = _load_json(args.local_report)
    payload: dict[str, Any] = {
        "schema": "sota-base-tester-handoff/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": args.environment,
        "release_status_path": str(args.release_status),
        "release_status": {
            "ok": bool(release.get("ok")),
            "status": str(release.get("status") or "unknown"),
            "local_stack_ok": bool(release.get("local_stack_ok")),
            "local_ok": bool(release.get("local_ok")),
            "local_wallet_ok": bool(release.get("local_wallet_ok")),
            "local_wallet": release.get("local_wallet") or {},
            "local_remote_wallet_ok": bool(release.get("local_remote_wallet_ok")),
            "local_remote_wallet": release.get("local_remote_wallet") or {},
            "local_tailscale_preflight": release.get("local_tailscale_preflight") or {},
            "testnet_ok": release.get("testnet_ok"),
            "summary": release.get("summary") or {},
        },
        "warnings": [
            "Use the printed local-only private key only on the local Anvil network.",
            "Never paste a real seed phrase or production private key into the local demo.",
            "Do not invite a Base Sepolia tester while infrastructure gates are red; claim transaction evidence is the tester action that completes full verification.",
        ],
    }
    if args.environment in {"local", "both"}:
        payload["local"] = _local_section(state, release, local_report)
    if args.environment in {"testnet", "both"}:
        payload["testnet"] = _testnet_section(release, args.release_status.parent)
    return payload


def render_markdown(handoff: dict[str, Any]) -> str:
    lines: list[str] = []
    release = dict(handoff.get("release_status") or {})
    lines.append("# SOTA Base Tester Handoff")
    lines.append("")
    lines.append(f"Generated: {handoff.get('generated_at')}")
    lines.append("")
    lines.append("## Overall Status")
    lines.append("")
    lines.append(f"- Local stack ready: {str(release.get('local_stack_ok')).lower()}")
    lines.append(f"- Local ready: {str(release.get('local_ok')).lower()}")
    lines.append(f"- Local MetaMask ready: {str(release.get('local_wallet_ok')).lower()}")
    lines.append(f"- Remote Tailscale MetaMask ready: {str(release.get('local_remote_wallet_ok')).lower()}")
    local_wallet = dict(release.get("local_wallet") or {})
    remote_wallet = dict(release.get("local_remote_wallet") or {})
    if local_wallet:
        lines.append(f"- Local MetaMask status: {local_wallet.get('status') or 'unknown'}")
        if local_wallet.get("message"):
            lines.append(f"- Local MetaMask detail: {local_wallet.get('message')}")
    tailscale_preflight = dict(release.get("local_tailscale_preflight") or {})
    if remote_wallet:
        lines.append(f"- Remote Tailscale MetaMask status: {remote_wallet.get('status') or 'unknown'}")
        if remote_wallet.get("message"):
            lines.append(f"- Remote Tailscale MetaMask detail: {remote_wallet.get('message')}")
    if tailscale_preflight:
        lines.append(f"- Tailscale preflight: {tailscale_preflight.get('status') or 'unknown'}")
        if tailscale_preflight.get("path"):
            lines.append(f"- Tailscale preflight report: {tailscale_preflight.get('path')}")
    testnet = handoff.get("testnet") if isinstance(handoff.get("testnet"), dict) else {}
    lines.append(f"- Base Sepolia claim test ready: {str(dict(testnet).get('ready')).lower() if testnet else 'unknown'}")
    lines.append(f"- Base Sepolia full release ready: {str(release.get('testnet_ok')).lower()}")
    lines.append(f"- Full local + Base Sepolia status: {release.get('status')}")
    lines.append(f"- Gate summary: {_summary_text(dict(release.get('summary') or {}))}")
    lines.append("")
    lines.append("## Safety")
    lines.append("")
    for warning in handoff.get("warnings") or []:
        lines.append(f"- {warning}")
    lines.append("")
    local = handoff.get("local")
    if isinstance(local, dict):
        lines.append("## Local Demo")
        lines.append("")
        lines.append(f"- Ready: {str(local.get('ready')).lower()}")
        lines.append(f"- Status: {local.get('status')}")
        lines.append(f"- UI smoke: {local.get('smoke_status')} ({_summary_text(dict(local.get('smoke_summary') or {}))})")
        lines.append(f"- State-changing claim proof: {local.get('claim_proof_status')} ({_summary_text(dict(local.get('claim_proof_summary') or {}))}); {local.get('claim_proof_scope')}")
        lines.append(f"- Claim proof report: {local.get('claim_proof_report')}")
        lines.append(f"- Claims UI: {local.get('claims_ui_url')}")
        lines.append(f"- Docs: {local.get('docs_url')}")
        lines.append(f"- Autoresearch dashboard: {local.get('autoresearch_dashboard_url')}")
        lines.append(f"- MetaMask RPC URL: {local.get('anvil_rpc_url')}")
        lines.append(f"- Share mode: {local.get('share_mode') or 'unknown'} ({local.get('share_status') or 'unknown'})")
        lines.append(f"- Wallet RPC browser-safe: {str(local.get('wallet_rpc_browser_safe')).lower()}")
        if local.get("tailscale_dns_name"):
            lines.append(f"- Tailscale DNS: {local.get('tailscale_dns_name')}")
        if local.get("share_warning"):
            lines.append(f"- Share warning: {local.get('share_warning')}")
        lines.append(f"- MetaMask chain ID: {local.get('chain_id')}")
        lines.append(f"- Wallet address: {local.get('wallet_address')}")
        lines.append(f"- Local-only private key: `{local.get('local_only_private_key')}`")
        lines.append(f"- Old coldkey lookup: `{local.get('old_coldkey')}`")
        lines.append(f"- Genesis claim amount: {local.get('genesis_claim_amount')}")
        lines.append(f"- TAO credit in genesis claim: {local.get('genesis_tao_credit')}")
        lines.append(f"- Alpha liquidation credit in genesis claim: {local.get('genesis_alpha_credit')}")
        lines.append(f"- Mined emission claim amount: {local.get('emission_claim_amount')}")
        lines.append(f"- Self-validation: {local.get('self_validation_status')} ({local.get('self_validation_summary')})")
        validators = [
            f"{dict(item).get('name')} `{dict(item).get('hotkey')}`"
            for item in local.get("peer_validators") or []
            if isinstance(item, dict)
        ]
        if validators:
            lines.append(f"- Peer validators: {', '.join(validators)}")
        lines.append("")
        lines.append("### Local Steps")
        lines.append("")
        for index, step in enumerate(local.get("expected_flow") or [], start=1):
            lines.append(f"{index}. {step}")
        lines.append("")
        lines.append("### Local MetaMask Evidence To Record")
        lines.append("")
        for item in local.get("manual_evidence_checklist") or []:
            lines.append(f"- {item}")
        if local.get("local_tx_evidence_command"):
            lines.append("")
            lines.append("```bash")
            lines.append(str(local.get("local_tx_evidence_command")))
            lines.append("```")
        lines.append("")
    if isinstance(testnet, dict):
        lines.append("## Base Sepolia")
        lines.append("")
        lines.append(f"- Claim test ready: {str(testnet.get('ready')).lower()}")
        lines.append(f"- Full release ready: {str(testnet.get('release_ready')).lower()}")
        lines.append(f"- Claim test status: {testnet.get('status')}")
        lines.append(f"- Release status: {testnet.get('release_status')}")
        lines.append(f"- Tester message: {testnet.get('tester_message')}")
        if testnet.get("claims_ui_url"):
            lines.append(f"- Claims UI: {testnet.get('claims_ui_url')}")
        if testnet.get("claims_api_url"):
            lines.append(f"- Claims API: {testnet.get('claims_api_url')}")
        if testnet.get("readiness_url"):
            lines.append(f"- Readiness: {testnet.get('readiness_url')}")
        lines.append(f"- Browser smoke: {testnet.get('browser_smoke_status')} ({_summary_text(dict(testnet.get('browser_smoke_summary') or {}))})")
        if testnet.get("test_wallet_address"):
            lines.append(f"- Test wallet: `{testnet.get('test_wallet_address')}`")
        if testnet.get("old_coldkey"):
            lines.append(f"- Old coldkey lookup: `{testnet.get('old_coldkey')}`")
        if testnet.get("lane_id"):
            lines.append(f"- Emission lane: `{testnet.get('lane_id')}` epoch {testnet.get('epoch')}")
        lines.append(f"- Genesis claim amount: {testnet.get('genesis_claim_amount')}")
        lines.append(f"- Mined emission claim amount: {testnet.get('emission_claim_amount')}")
        if testnet.get("genesis_root_id"):
            lines.append(f"- Genesis root id: `{testnet.get('genesis_root_id')}`")
        if testnet.get("emission_root_id"):
            lines.append(f"- Emission root id: `{testnet.get('emission_root_id')}`")
        lines.append("")
        lines.append("### Gates")
        lines.append("")
        gates = testnet.get("gates") or []
        if gates:
            for gate in gates:
                gate = dict(gate)
                lines.append(f"- {gate.get('name')}: {gate.get('status')} ({_summary_text(dict(gate.get('summary') or {}))})")
        else:
            lines.append("- No Base Sepolia gate reports found.")
        blocked = testnet.get("blocked_gates") or []
        if blocked:
            lines.append("")
            lines.append("### Remaining Evidence Gate" if testnet.get("ready") else "### Blocked Gates")
            lines.append("")
            for gate in blocked:
                gate = dict(gate)
                lines.append(f"- {gate.get('name')}: {gate.get('next_action')}")
        immediate = testnet.get("immediate_blockers") or []
        if immediate:
            lines.append("")
            lines.append("### Immediate Base Sepolia Blockers")
            lines.append("")
            for check in immediate:
                check = dict(check)
                detail = f"{check.get('name')}: {check.get('detail')}"
                if check.get("remediation"):
                    detail += f" Next: {check.get('remediation')}"
                lines.append(f"- {detail}")
        funding_targets = testnet.get("funding_targets") or []
        if funding_targets:
            lines.append("")
            lines.append("### Base Sepolia Funding Targets")
            lines.append("")
            for target in funding_targets:
                target = dict(target)
                lines.append(
                    "- "
                    f"{target.get('label')}: {target.get('status')}; "
                    f"address `{target.get('address')}`; "
                    f"balance {target.get('balance_eth') or 'unknown'} ETH; "
                    f"minimum {target.get('minimum_balance_eth') or 'unknown'} ETH; "
                    f"needs {target.get('needed_eth') or 'unknown'} ETH"
                )
        faucet_sources = testnet.get("faucet_sources") or []
        if faucet_sources:
            lines.append("")
            lines.append("### Base Sepolia Faucet Sources")
            lines.append("")
            for source in faucet_sources:
                source = dict(source)
                note = f" - {source.get('note')}" if source.get("note") else ""
                lines.append(f"- [{source.get('name')}]({source.get('url')}){note}")
        lines.append("")
        lines.append("### Testnet Claim Steps")
        lines.append("")
        for index, step in enumerate(testnet.get("expected_flow_when_ready") or [], start=1):
            lines.append(f"{index}. {step}")
        if testnet.get("claim_tx_evidence_command"):
            lines.append("")
            lines.append("### Base Sepolia Claim Evidence Command")
            lines.append("")
            lines.append("```bash")
            lines.append(str(testnet.get("claim_tx_evidence_command")))
            lines.append("```")
        if testnet.get("post_evidence_refresh_command"):
            lines.append("")
            lines.append("### Refresh Release/Handoff After Evidence")
            lines.append("")
            lines.append("```bash")
            lines.append(str(testnet.get("post_evidence_refresh_command")))
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _html_list(items: list[str]) -> str:
    return "\n".join(f"<li>{escape(str(item))}</li>" for item in items)


def _json_for_script(value: Any) -> str:
    return json.dumps(value, sort_keys=True).replace("</", "<\\/")


def render_html(handoff: dict[str, Any]) -> str:
    release = dict(handoff.get("release_status") or {})
    local = handoff.get("local") if isinstance(handoff.get("local"), dict) else None
    testnet = handoff.get("testnet") if isinstance(handoff.get("testnet"), dict) else None
    if local:
        primary_status_class = "ok" if local.get("ready") else "blocked"
        primary_status_text = "Local demo ready" if local.get("ready") else "Local demo blocked"
    else:
        primary_status_class = "ok" if release.get("ok") else "blocked"
        primary_status_text = f"Aggregate status: {release.get('status')}"
    blocks: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>SOTA Base Tester Handoff</title>",
        "<style>",
        ":root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#13282f;background:#f4f8f8}",
        "body{margin:0;line-height:1.55;background:#f4f8f8}",
        "main{max-width:1120px;margin:0 auto;padding:28px}",
        ".hero{border-bottom:1px solid #cfd9dd;padding:28px 0 24px;margin-bottom:20px}",
        ".kicker{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;font-weight:800;text-transform:uppercase;color:#5b6770;letter-spacing:0}",
        "h1{max-width:840px;font-size:46px;line-height:1.04;margin:8px 0 10px;letter-spacing:0}",
        ".lede{max-width:780px;margin:0;color:#5b6770;font-size:17px}",
        "h2{font-size:22px;margin:32px 0 12px}",
        "h3{font-size:16px;margin:24px 0 8px}",
        ".status{display:inline-flex;border-radius:6px;padding:6px 10px;font-weight:700}",
        ".ok{background:#dcfce7;color:#166534}",
        ".blocked{background:#fee2e2;color:#991b1b}",
        ".grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}",
        ".summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0 0}",
        ".card{border:1px solid #cfd9dd;background:white;border-radius:8px;padding:16px;box-shadow:0 1px 0 rgba(19,40,47,.05)}",
        ".label{font-size:12px;text-transform:uppercase;color:#5b6770;font-weight:700}",
        ".value{word-break:break-word;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}",
        ".audience{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;border:1px solid #cfd9dd;background:#cfd9dd;border-radius:8px;overflow:hidden;margin:18px 0}",
        ".audience div{background:white;padding:16px;min-height:110px}",
        ".audience strong{display:block;font-size:16px}",
        ".audience span{display:block;margin-top:8px;color:#5b6770}",
        ".journey{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;border:1px solid #cfd9dd;background:#cfd9dd;border-radius:8px;overflow:hidden;margin:16px 0}",
        ".journey div{background:white;padding:16px;min-height:104px}",
        ".journey b{display:block;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#b45309;font-size:12px;text-transform:uppercase}",
        ".journey span{display:block;margin-top:8px;color:#5b6770}",
        ".actions{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}",
        "button,.button{appearance:none;border:1px solid #13282f;background:#13282f;color:white;border-radius:6px;padding:10px 12px;font-weight:700;text-decoration:none;cursor:pointer}",
        "button.secondary,.button.secondary{background:white;color:#13282f}",
        ".action-status{min-height:24px;color:#5b6770;font-size:14px}",
        ".flow{border:1px solid #b7e4c7;background:#f0fdf4;border-radius:6px;padding:16px;margin:16px 0}",
        "code{word-break:break-all;background:#edf5f5;border-radius:5px;padding:2px 5px}",
        "ol,ul{padding-left:22px}",
        "@media(max-width:860px){main{padding:20px}.grid,.summary,.audience,.journey{grid-template-columns:1fr}h1{font-size:32px}}",
        "</style>",
        "</head>",
        "<body><main>",
        '<section class="hero">',
        '<div class="kicker">SOTA Base local tester</div>',
        "<h1>SOTA Base Tester Handoff</h1>",
        '<p class="lede">Use this page to run the local fork demo from a normal browser: add the local Base network, import the local-only account, claim genesis SOTA, inspect the mined emission, and claim again after self-validation.</p>',
        f'<p><span class="status {primary_status_class}">{escape(primary_status_text)}</span></p>',
        f"<p>Generated: {escape(str(handoff.get('generated_at')))}</p>",
        "</section>",
        '<section class="summary">',
        f'<div class="card"><div class="label">Local stack ready</div><div class="value">{escape(str(release.get("local_stack_ok")).lower())}</div></div>',
        f'<div class="card"><div class="label">Local ready</div><div class="value">{escape(str(release.get("local_ok")).lower())}</div></div>',
        f'<div class="card"><div class="label">Local MetaMask ready</div><div class="value">{escape(str(release.get("local_wallet_ok")).lower())}</div></div>',
        f'<div class="card"><div class="label">Remote Tailscale MetaMask ready</div><div class="value">{escape(str(release.get("local_remote_wallet_ok")).lower())}</div></div>',
        f'<div class="card"><div class="label">Tailscale preflight</div><div class="value">{escape(str(dict(release.get("local_tailscale_preflight") or {}).get("status") or "unknown"))}</div></div>',
        f'<div class="card"><div class="label">Base Sepolia claim test ready</div><div class="value">{escape(str(dict(testnet or {}).get("ready")).lower() if testnet else "unknown")}</div></div>',
        f'<div class="card"><div class="label">Base Sepolia full release ready</div><div class="value">{escape(str(release.get("testnet_ok")).lower())}</div></div>',
        "</section>",
        '<section class="audience">',
        "<div><strong>I am new</strong><span>Follow Local Steps and use only the printed local-only MetaMask account. You do not need TAO, Base ETH, or a Bittensor wallet.</span></div>",
        "<div><strong>I am migrating</strong><span>Genesis shows TAO 1:1 plus synthetic alpha accounting credit. Mining uses an EVM identity and SOTA-only emissions.</span></div>",
        "<div><strong>I am reviewing</strong><span>Check the green UI smoke, state-changing claim proof, self-validation evidence, and local transaction receipts.</span></div>",
        "</section>",
        "<h2>Safety</h2>",
        f"<ul>{_html_list([str(item) for item in handoff.get('warnings') or []])}</ul>",
    ]
    if local:
        blocks.extend(
            [
                "<h2>Local Demo</h2>",
                '<section class="grid">',
                f'<div class="card"><div class="label">Ready</div><div class="value">{escape(str(local.get("ready")).lower())}</div></div>',
                f'<div class="card"><div class="label">UI smoke</div><div class="value">{escape(str(local.get("smoke_status") or "unknown"))} ({escape(_summary_text(dict(local.get("smoke_summary") or {})))})</div></div>',
                f'<div class="card"><div class="label">State-changing claim proof</div><div class="value">{escape(str(local.get("claim_proof_status") or "missing"))} ({escape(_summary_text(dict(local.get("claim_proof_summary") or {})))})<br>{escape(str(local.get("claim_proof_scope") or ""))}</div></div>',
                f'<div class="card"><div class="label">Claim proof report</div><div class="value">{escape(str(local.get("claim_proof_report") or ""))}</div></div>',
                f'<div class="card"><div class="label">Claims UI</div><div class="value"><a href="{escape(str(local.get("claims_ui_url")))}">{escape(str(local.get("claims_ui_url")))}</a></div></div>',
                f'<div class="card"><div class="label">Docs</div><div class="value"><a href="{escape(str(local.get("docs_url")))}">{escape(str(local.get("docs_url")))}</a></div></div>',
                f'<div class="card"><div class="label">Autoresearch dashboard</div><div class="value"><a href="{escape(str(local.get("autoresearch_dashboard_url")))}">{escape(str(local.get("autoresearch_dashboard_url")))}</a></div></div>',
                f'<div class="card"><div class="label">MetaMask RPC URL</div><div class="value">{escape(str(local.get("anvil_rpc_url")))}</div></div>',
                f'<div class="card"><div class="label">Share mode</div><div class="value">{escape(str(local.get("share_mode") or "unknown"))} ({escape(str(local.get("share_status") or "unknown"))})<br>Wallet RPC browser-safe: {escape(str(local.get("wallet_rpc_browser_safe")).lower())}</div></div>',
                f'<div class="card"><div class="label">MetaMask chain ID</div><div class="value">{escape(str(local.get("chain_id")))}</div></div>',
                f'<div class="card"><div class="label">Wallet address</div><div class="value">{escape(str(local.get("wallet_address")))}</div></div>',
                f'<div class="card"><div class="label">Local-only private key</div><div class="value"><code>{escape(str(local.get("local_only_private_key")))}</code></div></div>',
                f'<div class="card"><div class="label">Old coldkey lookup</div><div class="value"><code>{escape(str(local.get("old_coldkey")))}</code></div></div>',
                f'<div class="card"><div class="label">Genesis SOTA claim</div><div class="value">{escape(str(local.get("genesis_claim_amount")))}</div></div>',
                f'<div class="card"><div class="label">TAO + alpha accounting</div><div class="value">{escape(str(local.get("genesis_tao_credit")))} + {escape(str(local.get("genesis_alpha_credit")))}</div></div>',
                f'<div class="card"><div class="label">Mined emission claim</div><div class="value">{escape(str(local.get("emission_claim_amount")))}</div></div>',
                f'<div class="card"><div class="label">Self-validation</div><div class="value">{escape(str(local.get("self_validation_status") or "unknown"))} ({escape(str(local.get("self_validation_summary") or "0/0 accepted"))})</div></div>',
                f'<div class="card"><div class="label">Peer validators</div><div class="value">{escape(", ".join(str(dict(item).get("name") or "validator") for item in local.get("peer_validators") or [] if isinstance(item, dict)) or "Not loaded")}</div></div>',
                "</section>",
                '<section class="journey">',
                f'<div><b>Claim genesis</b><span>{escape(str(local.get("genesis_claim_amount")))} from seeded TAO plus alpha accounting credit.</span></div>',
                f'<div><b>Mine locally</b><span>Alice miner submission creates a {escape(str(local.get("emission_claim_amount")))} SOTA emission.</span></div>',
                f'<div><b>Self-validate</b><span>{escape(str(local.get("self_validation_summary") or "0/0 accepted"))} from the local peer committee before claiming.</span></div>',
                "</section>",
                '<div class="flow">',
                "<h3>What this local test covers</h3>",
                "<ul>",
                "<li>Claim local SOTA based on seeded local-node TAO plus alpha accounting credit. TAO and alpha are not transferred by this local demo.</li>",
                "<li>Inspect a mined local SOTA emission generated by the autoresearch backend.</li>",
                "<li>Verify the emission is backed by accepted peer self-validation evidence from other local users before claiming.</li>",
                "</ul>",
                "</div>",
            ]
        )
        if local.get("share_warning"):
            blocks.extend(
                [
                    '<div class="flow">',
                    "<h3>Remote wallet note</h3>",
                    f"<p>{escape(str(local.get('share_warning')))}</p>",
                    "</div>",
                ]
            )
        blocks.extend(
            [
                '<div class="actions">',
                '<button id="add-local-network" type="button">Add SOTA Local Base network</button>',
                '<button id="copy-local-key" class="secondary" type="button">Copy local-only key</button>',
                '<button id="copy-wallet-address" class="secondary" type="button">Copy wallet address</button>',
                f'<a class="button secondary" href="{escape(str(local.get("claims_ui_url")))}">Open claims UI</a>',
                f'<a class="button secondary" href="{escape(str(local.get("autoresearch_dashboard_url")))}">Open autoresearch dashboard</a>',
                "</div>",
                '<p id="handoff-action-status" class="action-status" aria-live="polite"></p>',
                "<h3>Local Steps</h3>",
                f"<ol>{_html_list([str(item) for item in local.get('expected_flow') or []])}</ol>",
                "<h3>Local MetaMask Evidence To Record</h3>",
                f"<ul>{_html_list([str(item) for item in local.get('manual_evidence_checklist') or []])}</ul>",
                f"<p><code>{escape(str(local.get('local_tx_evidence_command') or ''))}</code></p>",
            ]
        )
    if testnet:
        gate_lines = []
        for gate in testnet.get("gates") or []:
            gate = dict(gate)
            gate_lines.append(
                f"{gate.get('name')}: {gate.get('status')} ({_summary_text(dict(gate.get('summary') or {}))})"
            )
        blocked_lines = [
            f"{dict(gate).get('name')}: {dict(gate).get('next_action')}"
            for gate in testnet.get("blocked_gates") or []
        ]
        immediate_lines = []
        for check in testnet.get("immediate_blockers") or []:
            check = dict(check)
            text = f"{check.get('name')}: {check.get('detail')}"
            if check.get("remediation"):
                text += f" Next: {check.get('remediation')}"
            immediate_lines.append(text)
        blocks.extend(
            [
                "<h2>Base Sepolia</h2>",
                f"<p>{escape(str(testnet.get('tester_message')))}</p>",
                '<section class="grid">',
                f'<div class="card"><div class="label">Claim test ready</div><div class="value">{escape(str(testnet.get("ready")).lower())}</div></div>',
                f'<div class="card"><div class="label">Full release ready</div><div class="value">{escape(str(testnet.get("release_ready")).lower())}</div></div>',
                f'<div class="card"><div class="label">Browser smoke</div><div class="value">{escape(str(testnet.get("browser_smoke_status") or "missing"))} ({escape(_summary_text(dict(testnet.get("browser_smoke_summary") or {})))})</div></div>',
                f'<div class="card"><div class="label">Claims UI</div><div class="value"><a href="{escape(str(testnet.get("claims_ui_url") or ""))}">{escape(str(testnet.get("claims_ui_url") or ""))}</a></div></div>',
                f'<div class="card"><div class="label">Claims API</div><div class="value">{escape(str(testnet.get("claims_api_url") or ""))}</div></div>',
                f'<div class="card"><div class="label">Readiness</div><div class="value"><a href="{escape(str(testnet.get("readiness_url") or ""))}">{escape(str(testnet.get("readiness_url") or ""))}</a></div></div>',
                f'<div class="card"><div class="label">Test wallet</div><div class="value"><code>{escape(str(testnet.get("test_wallet_address") or ""))}</code></div></div>',
                f'<div class="card"><div class="label">Old coldkey lookup</div><div class="value"><code>{escape(str(testnet.get("old_coldkey") or ""))}</code></div></div>',
                f'<div class="card"><div class="label">Emission lane</div><div class="value"><code>{escape(str(testnet.get("lane_id") or ""))}</code><br>Epoch {escape(str(testnet.get("epoch") or ""))}</div></div>',
                f'<div class="card"><div class="label">Genesis claim</div><div class="value">{escape(str(testnet.get("genesis_claim_amount") or ""))}</div></div>',
                f'<div class="card"><div class="label">Mined emission claim</div><div class="value">{escape(str(testnet.get("emission_claim_amount") or ""))}</div></div>',
                f'<div class="card"><div class="label">Root IDs</div><div class="value">Genesis <code>{escape(str(testnet.get("genesis_root_id") or ""))}</code><br>Emission <code>{escape(str(testnet.get("emission_root_id") or ""))}</code></div></div>',
                "</section>",
                '<div class="actions">',
                f'<a class="button secondary" href="{escape(str(testnet.get("claims_ui_url") or ""))}">Open Base Sepolia claims UI</a>',
                '<button id="copy-testnet-wallet" class="secondary" type="button">Copy testnet wallet</button>',
                "</div>",
                "<h3>Gates</h3>",
                f"<ul>{_html_list(gate_lines) if gate_lines else '<li>No Base Sepolia gate reports found.</li>'}</ul>",
            ]
        )
        if blocked_lines:
            heading = "Remaining Evidence Gate" if testnet.get("ready") else "Blocked Gates"
            blocks.extend([f"<h3>{heading}</h3>", f"<ul>{_html_list(blocked_lines)}</ul>"])
        if immediate_lines:
            blocks.extend(["<h3>Immediate Base Sepolia Blockers</h3>", f"<ul>{_html_list(immediate_lines)}</ul>"])
        funding_targets = [
            dict(target)
            for target in testnet.get("funding_targets") or []
            if isinstance(target, dict)
        ]
        if funding_targets:
            funding_cards = []
            for target in funding_targets:
                address = str(target.get("address") or "")
                explorer_url = str(target.get("explorer_url") or "")
                explorer_link = (
                    f'<a class="button secondary" href="{escape(explorer_url)}">Open explorer</a>'
                    if explorer_url
                    else ""
                )
                funding_cards.append(
                    '<div class="card">'
                    f'<div class="label">{escape(str(target.get("label") or "target"))} funding</div>'
                    f'<div class="value">{escape(str(target.get("status") or "unknown"))}<br>'
                    f'Balance: {escape(str(target.get("balance_eth") or "unknown"))} ETH<br>'
                    f'Minimum: {escape(str(target.get("minimum_balance_eth") or "unknown"))} ETH<br>'
                    f'Needed: {escape(str(target.get("needed_eth") or "unknown"))} ETH<br>'
                    f'<code>{escape(address)}</code></div>'
                    '<div class="actions">'
                    f'<button class="secondary funding-copy" type="button" data-address="{escape(address)}">Copy address</button>'
                    f"{explorer_link}"
                    "</div>"
                    "</div>"
                )
            blocks.extend(
                [
                    "<h3>Base Sepolia Funding Targets</h3>",
                    '<section class="grid">',
                    *funding_cards,
                    "</section>",
                ]
            )
        faucet_sources = [
            dict(source)
            for source in testnet.get("faucet_sources") or []
            if isinstance(source, dict)
        ]
        if faucet_sources:
            faucet_lines = []
            for source in faucet_sources:
                name = escape(str(source.get("name") or "Faucet source"))
                url = escape(str(source.get("url") or ""))
                note = escape(str(source.get("note") or ""))
                link = f'<a href="{url}">{name}</a>' if url else name
                faucet_lines.append(f"{link}: {note}" if note else link)
            blocks.extend(
                [
                    "<h3>Base Sepolia Faucet Sources</h3>",
                    "<ul>" + "\n".join(f"<li>{line}</li>" for line in faucet_lines) + "</ul>",
                ]
            )
        blocks.extend(
            [
                "<h3>Testnet Claim Steps</h3>",
                f"<ol>{_html_list([str(item) for item in testnet.get('expected_flow_when_ready') or []])}</ol>",
                "<h3>Base Sepolia Claim Evidence Command</h3>",
                f"<p><code>{escape(str(testnet.get('claim_tx_evidence_command') or ''))}</code></p>",
                "<h3>Refresh Release/Handoff After Evidence</h3>",
                f"<p><code>{escape(str(testnet.get('post_evidence_refresh_command') or ''))}</code></p>",
            ]
        )
    local_script = {}
    if local:
        local_script = {
            "chainId": local.get("chain_id_hex"),
            "chainName": local.get("network_name"),
            "rpcUrls": [local.get("anvil_rpc_url")],
            "nativeCurrency": {
                "name": "Ether",
                "symbol": local.get("native_currency_symbol") or "ETH",
                "decimals": 18,
            },
            "walletAddress": local.get("wallet_address"),
            "localOnlyPrivateKey": local.get("local_only_private_key"),
        }
    testnet_script = {}
    if testnet:
        testnet_script = {
            "walletAddress": testnet.get("test_wallet_address"),
        }
    blocks.extend(
        [
            "<script>",
            f"const localHandoff = {_json_for_script(local_script)};",
            f"const testnetHandoff = {_json_for_script(testnet_script)};",
            "const statusNode = document.getElementById('handoff-action-status');",
            "function setStatus(message){ if(statusNode){ statusNode.textContent = message; } }",
            "async function copyText(value, label){",
            "  if(!value){ setStatus(label + ' is missing.'); return; }",
            "  try {",
            "    if(navigator.clipboard && window.isSecureContext){ await navigator.clipboard.writeText(value); }",
            "    else { const textarea = document.createElement('textarea'); textarea.value = value; textarea.setAttribute('readonly', ''); textarea.style.position = 'fixed'; textarea.style.opacity = '0'; document.body.appendChild(textarea); textarea.select(); document.execCommand('copy'); document.body.removeChild(textarea); }",
            "    setStatus(label + ' copied.');",
            "  } catch (error) { setStatus(label + ' copy failed. Select the value on the page and copy it manually.'); }",
            "}",
            "document.getElementById('copy-local-key')?.addEventListener('click', () => copyText(localHandoff.localOnlyPrivateKey, 'Local-only key'));",
            "document.getElementById('copy-wallet-address')?.addEventListener('click', () => copyText(localHandoff.walletAddress, 'Wallet address'));",
            "document.getElementById('copy-testnet-wallet')?.addEventListener('click', () => copyText(testnetHandoff.walletAddress, 'Testnet wallet'));",
            "document.querySelectorAll('.funding-copy').forEach((button) => button.addEventListener('click', () => copyText(button.dataset.address, 'Funding address')));",
            "document.getElementById('add-local-network')?.addEventListener('click', async () => {",
            "  try {",
            "    if(!window.ethereum){ setStatus('MetaMask is not available in this browser.'); return; }",
            "    if(!localHandoff.chainId || !localHandoff.rpcUrls || !localHandoff.rpcUrls[0]){ setStatus('Local network details are missing.'); return; }",
            "    await window.ethereum.request({ method: 'wallet_addEthereumChain', params: [{ chainId: localHandoff.chainId, chainName: localHandoff.chainName, rpcUrls: localHandoff.rpcUrls, nativeCurrency: localHandoff.nativeCurrency }] });",
            "    setStatus('SOTA Local Base network added or selected in MetaMask.');",
            "  } catch (error) { setStatus(error && error.message ? error.message : 'MetaMask network add failed.'); }",
            "});",
            "</script>",
            "</main></body></html>",
        ]
    )
    return "\n".join(blocks)


def _write_handoff_artifacts(
    *,
    handoff: dict[str, Any],
    markdown: str,
    html: str,
    json_out: Path,
    markdown_out: Path,
    html_out: Path,
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_out.write_text(markdown, encoding="utf-8")
    html_out.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a nontechnical SOTA Base tester handoff from live report artifacts.")
    parser.add_argument("--environment", choices=("local", "testnet", "both"), default="both")
    parser.add_argument("--state", type=Path, default=LOCAL_RUN_DIR / "state.json")
    parser.add_argument("--local-report", type=Path, default=LOCAL_RUN_DIR / "ui-smoke" / "report.json")
    parser.add_argument("--release-status", type=Path, default=TESTNET_RUN_DIR / "base-sota-release-status.json")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--html-out", type=Path)
    parser.add_argument("--mirror-local", action="store_true", help="Also refresh the locally served handoff copy when explicit outputs are supplied.")
    parser.add_argument("--print-markdown", action="store_true")
    args = parser.parse_args(argv)
    default_outputs = not (args.json_out or args.markdown_out or args.html_out)
    args.json_out = args.json_out or TESTNET_RUN_DIR / "base-sota-tester-handoff.json"
    args.markdown_out = args.markdown_out or TESTNET_RUN_DIR / "base-sota-tester-handoff.md"
    args.html_out = args.html_out or TESTNET_RUN_DIR / "base-sota-tester-handoff.html"
    handoff = build_handoff(args)
    markdown = render_markdown(handoff)
    html = render_html(handoff)
    _write_handoff_artifacts(
        handoff=handoff,
        markdown=markdown,
        html=html,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
        html_out=args.html_out,
    )
    local_mirror: dict[str, str] | None = None
    if (default_outputs or args.mirror_local) and args.environment in {"local", "both"}:
        mirror_json = LOCAL_HANDOFF_DIR / "handoff.json"
        mirror_markdown = LOCAL_HANDOFF_DIR / "handoff.md"
        mirror_html = LOCAL_HANDOFF_DIR / "index.html"
        _write_handoff_artifacts(
            handoff=handoff,
            markdown=markdown,
            html=html,
            json_out=mirror_json,
            markdown_out=mirror_markdown,
            html_out=mirror_html,
        )
        local_mirror = {
            "json": str(mirror_json),
            "markdown": str(mirror_markdown),
            "html": str(mirror_html),
        }
    if args.print_markdown:
        print(markdown, end="")
    else:
        print(
            json.dumps(
                {
                    "ok": True,
                    "json": str(args.json_out),
                    "markdown": str(args.markdown_out),
                    "html": str(args.html_out),
                    "local_mirror": local_mirror,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

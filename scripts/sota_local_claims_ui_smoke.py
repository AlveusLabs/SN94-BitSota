#!/usr/bin/env python3
from __future__ import annotations

import argparse
from html import unescape
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


REPOS = Path("/home/mekaneeky/repos")
RUN_DIR = REPOS / ".sota-base-local"
DEFAULT_STATE_PATH = RUN_DIR / "state.json"
DEFAULT_REPORT_PATH = RUN_DIR / "ui-smoke" / "report.json"
DEFAULT_SCREENSHOT_PATH = RUN_DIR / "ui-smoke" / "claims-page.png"
LANE_ID = "base:sota-local"
MIN_LOCAL_SELF_VALIDATION_COMMITTEE = 3
HANDOFF_BASE_SEPOLIA_STATUS_TEXTS = (
    "Base Sepolia is not ready for a nontechnical MetaMask tester yet.",
    "Base Sepolia is fully verified, including human MetaMask claim transaction evidence.",
)

EXPECTED_PAGE_TEXT = (
    "SOTA Local Base claims",
    "Keep TAO and alpha. Claim SOTA separately.",
    "Local demo path",
    "Bob, Charlie, and Dave peer validation",
    "Load genesis claim",
    "Load mined emission",
    "Local readiness",
    "Claim sources",
    "TAO credit",
    "Alpha synthetic credit",
    "Total SOTA",
    "Unclaimed SOTA",
)

EXPECTED_DOC_PAGE_TEXT = {
    "docs_base": (
        "Claim and mine SOTA on a local Base chain.",
        "New user guide",
        "Migrating from Bittensor",
        "One-Command Demo",
        "Base Sepolia",
    ),
    "docs_new_users": (
        "Try Base SOTA without knowing Bittensor.",
        "Terms In Plain English",
        "Genesis claim",
        "Emission claim",
        "Do not use real keys in the local demo.",
    ),
    "docs_bittensor_migrants": (
        "Move from subnet thinking to a Base-settled SOTA fork.",
        "Coldkey",
        "Hotkey",
        "Validator weights/Yuma emissions",
        "Mining Uses EVM Identity",
    ),
    "docs_local_e2e": (
        "Run the whole Base SOTA loop on this machine.",
        "Local readiness",
        "Import the printed local-only MetaMask account",
        "Submit the local genesis claim",
        "Submit the local emission claim",
    ),
}


def _print(message: str) -> None:
    print(message, flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _join_url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _visible_text(html: str) -> str:
    without_comments = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    without_tags = re.sub(r"<[^>]+>", " ", without_comments)
    return unescape(re.sub(r"\s+", " ", without_tags))


def _check(name: str, ok: bool, detail: str, *, remediation: str | None = None) -> dict[str, str]:
    payload = {
        "name": name,
        "status": "green" if ok else "red",
        "detail": detail,
    }
    if remediation:
        payload["remediation"] = remediation
    return payload


def _yellow(name: str, detail: str, *, remediation: str | None = None) -> dict[str, str]:
    payload = {"name": name, "status": "yellow", "detail": detail}
    if remediation:
        payload["remediation"] = remediation
    return payload


def _summary(checks: list[dict[str, str]]) -> dict[str, int]:
    return {
        "green": sum(1 for check in checks if check["status"] == "green"),
        "yellow": sum(1 for check in checks if check["status"] == "yellow"),
        "red": sum(1 for check in checks if check["status"] == "red"),
    }


def _http_text(url: str, *, timeout: float) -> str:
    request = Request(url, headers={"Accept": "text/html"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc


def _http_json(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def _http_status(method: str, url: str, *, timeout: float) -> int:
    request = Request(url, headers={"Accept": "application/json"}, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except HTTPError as exc:
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def wait_for_claims_page(url: str, *, timeout_seconds: float, request_timeout: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            _http_text(url, timeout=request_timeout)
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for claims page at {url}: {last_error}")


def build_targets(state: dict[str, Any], *, claims_url: str | None = None) -> dict[str, str]:
    alice = str(state["accounts"]["alice_reward"])
    old_coldkey = str(state["genesis"]["old_coldkey"])
    lane_id = str(state.get("autoresearch", {}).get("subnet", {}).get("id") or LANE_ID)
    if claims_url:
        base_url = claims_url.rsplit("/claims", 1)[0].rstrip("/")
        page_url = claims_url if claims_url.endswith("/claims") else _join_url(claims_url, "/claims")
    else:
        base_url = "http://127.0.0.1:3000"
        page_url = _join_url(base_url, "/claims")
    docs_base = str(dict(state.get("urls") or {}).get("docs") or "http://127.0.0.1:9002/base/")

    genesis_query = urlencode(
        {
            "old_coldkey": old_coldkey,
            "reward_address": alice,
            "subnet_id": "genesis",
        }
    )
    emission_query = urlencode({"evm_address": alice, "subnet_id": lane_id})
    encoded_lane = quote(lane_id, safe="")
    targets = {
        "claims_page": page_url,
        "genesis_lookup": f"{base_url}/api/sota-claims/api/v1/base/eligibility/{quote(alice)}?{genesis_query}",
        "emission_lookup": f"{base_url}/api/sota-claims/api/v1/base/eligibility/{quote(alice)}?{emission_query}",
        "genesis_transaction": f"{base_url}/api/sota-claims/api/v1/base/claims/transaction",
        "emission_transaction": f"{base_url}/api/sota-claims/api/v1/base/claims/transaction",
        "claims_health": f"{base_url}/api/sota-claims/health",
        "index_sync": f"{base_url}/api/sota-claims/api/v1/base/index/sync",
        "index_status": f"{base_url}/api/sota-claims/api/v1/base/index/status",
        "autoresearch_ready": f"{base_url}/api/sota-autoresearch/readyz",
        "self_validation_evidence": (
            f"{base_url}/api/sota-autoresearch/api/v1/sota/subnets/{encoded_lane}/epochs/1/evidence"
        ),
        "docs_base": docs_base,
        "docs_new_users": _join_url(docs_base, "new-users/"),
        "docs_bittensor_migrants": _join_url(docs_base, "bittensor-migrants/"),
        "docs_local_e2e": _join_url(docs_base, "local-e2e/"),
    }
    handoff_url = str(dict(state.get("urls") or {}).get("handoff") or "")
    if handoff_url:
        targets["handoff"] = handoff_url
    return targets


def validate_page_html(html: str) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    visible_text = _visible_text(html)
    missing = [text for text in EXPECTED_PAGE_TEXT if text not in visible_text]
    checks.append(
        _check(
            "claims_page_text",
            not missing,
            "claims page exposes local demo, lookup, source, and credit text"
            if not missing
            else f"claims page is missing: {', '.join(missing)}",
            remediation="Restart the local demo website and rebuild if the page shell is stale.",
        )
    )
    return checks


def validate_docs_htmls(docs: dict[str, str]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for name, expected_texts in EXPECTED_DOC_PAGE_TEXT.items():
        if name not in docs:
            continue
        html = docs.get(name, "")
        visible_text = _visible_text(html)
        missing = [text for text in expected_texts if text not in visible_text]
        page_label = name.replace("docs_", "docs ")
        checks.append(
            _check(
                name,
                not missing,
                f"{page_label} page exposes the required tester guidance"
                if not missing
                else f"{page_label} page is missing: {', '.join(missing)}",
                remediation="Restart the docs server or rebuild the Base docs if a tester page is stale.",
            )
        )
    return checks


def validate_live_docs_pages(targets: dict[str, str], *, request_timeout: float) -> list[dict[str, str]]:
    docs_html: dict[str, str] = {}
    checks: list[dict[str, str]] = []
    for name in EXPECTED_DOC_PAGE_TEXT:
        try:
            docs_html[name] = _http_text(targets[name], timeout=request_timeout)
        except Exception as exc:
            checks.append(
                _check(
                    name,
                    False,
                    f"{targets[name]} failed: {exc}",
                    remediation="Start the local docs server with ./scripts/sota_local_demo.py launch before tester review.",
                )
            )
    checks.extend(validate_docs_htmls(docs_html))
    return checks


def validate_handoff_page(targets: dict[str, str], *, request_timeout: float) -> list[dict[str, str]]:
    url = targets.get("handoff")
    if not url:
        return []
    try:
        html = _http_text(url, timeout=request_timeout)
    except Exception as exc:
        return [
            _check(
                "tester_handoff_page",
                False,
                f"{url} failed: {exc}",
                remediation="Regenerate and serve the tester handoff with ./scripts/sota_local_demo.py launch.",
            )
        ]
    visible_text = _visible_text(html)
    expected = (
        "SOTA Base Tester Handoff",
        "Local Demo",
        "Local-only private key",
        "Add SOTA Local Base network",
        "Open claims UI",
        "Mined emission claim",
        "Self-validation",
        "Peer validators",
        "State-changing claim proof",
    )
    missing = [text for text in expected if text not in visible_text]
    if "Local demo ready" not in visible_text and "Local demo blocked" not in visible_text:
        missing.append("Local demo ready or Local demo blocked")
    if "Base Sepolia" in visible_text and not any(text in visible_text for text in HANDOFF_BASE_SEPOLIA_STATUS_TEXTS):
        missing.append("Base Sepolia ready or blocked status")
    return [
        _check(
            "tester_handoff_page",
            not missing,
            "tester handoff page exposes local wallet steps and Base Sepolia gate status",
            remediation=f"Tester handoff page is missing: {', '.join(missing)}",
        )
    ]


def validate_tester_share(state: dict[str, Any]) -> list[dict[str, str]]:
    urls = dict(state.get("urls") or {})
    sharing = dict(state.get("sharing") or {})
    rpc_url = str(urls.get("anvil_rpc") or "")
    claims_url = str(urls.get("claims_ui") or "")
    if not rpc_url or not claims_url:
        return [
            _check(
                "tester_wallet_rpc",
                False,
                "tester URLs are missing the claims UI or local RPC URL",
                remediation="Relaunch the local demo so it regenerates state URLs for the handoff and claims UI.",
            )
        ]
    parsed = urlparse(rpc_url)
    host = (parsed.hostname or "").lower()
    browser_safe = (
        parsed.scheme == "https"
        and bool(sharing.get("wallet_rpc_browser_safe"))
    ) or (parsed.scheme == "http" and host in {"127.0.0.1", "localhost"})
    if browser_safe:
        return [
            _check(
                "tester_wallet_rpc",
                True,
                f"tester wallet RPC is browser-safe for share mode {sharing.get('mode') or 'local'}",
            )
        ]
    return [
        _yellow(
            "tester_wallet_rpc",
            f"tester wallet RPC may be rejected by MetaMask from another computer: {rpc_url}",
            remediation="Relaunch with ./scripts/sota_local_demo.py launch --share-mode tailscale-https for remote Tailscale MetaMask testing.",
        )
    ]


def _raw_credit(payload: dict[str, Any], key: str) -> str:
    return str(dict(dict(payload.get("credits") or {}).get(key) or {}).get("raw") or "")


def validate_api_payloads(
    state: dict[str, Any],
    *,
    genesis_lookup: dict[str, Any],
    emission_lookup: dict[str, Any],
    evidence: dict[str, Any],
    genesis_transaction: dict[str, Any],
    emission_transaction: dict[str, Any],
    claims_health: dict[str, Any],
    index_status: dict[str, Any],
    autoresearch_ready_status: int,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    alice = str(state["accounts"]["alice_reward"]).lower()
    genesis = dict(state["genesis"])
    contracts = {key: str(value).lower() for key, value in dict(state["contracts"]).items()}
    emission_root = dict(state["autoresearch"]["emission_root"])
    emission_amount = int(state["emission_onchain"]["amount"])
    chain_id = int(state["chain_id"])

    checks.append(
        _check(
            "local_readiness_claims_health",
            str(claims_health.get("status")).lower() == "ok" and int(claims_health.get("chain_id") or 0) == chain_id,
            "claims API health endpoint reports the local chain",
            remediation="Restart the local demo and check the indexer log.",
        )
    )
    contracts_configured = [str(item) for item in index_status.get("contracts_configured") or []]
    required_contracts = {"root_registry", "lane_registry", "genesis_distributor", "emission_distributor"}
    checks.append(
        _check(
            "local_readiness_index_status",
            int(index_status.get("chain_id") or 0) == chain_id
            and set(contracts_configured) >= required_contracts
            and int(index_status.get("lag_blocks") or 0) == 0
            and not index_status.get("last_sync_error"),
            "indexer status reports local RPC sync, all contract roles, and zero lag",
            remediation="Run local lookup or restart the local demo if index lag or contract config is wrong.",
        )
    )
    checks.append(
        _check(
            "local_readiness_autoresearch",
            200 <= autoresearch_ready_status < 300,
            "autoresearch readiness endpoint is reachable through the website proxy",
            remediation="Restart the local demo and check the autoresearch log.",
        )
    )
    checks.append(
        _check(
            "genesis_lookup",
            bool(genesis_lookup.get("eligible"))
            and str(genesis_lookup.get("account", "")).lower() == alice
            and _raw_credit(genesis_lookup, "tao") == str(genesis["tao_credit"])
            and _raw_credit(genesis_lookup, "alpha_synthetic") == str(genesis["alpha_synthetic_credit"])
            and _raw_credit(genesis_lookup, "total_sota") == str(genesis["amount"]),
            "genesis lookup returns TAO, synthetic alpha, and total SOTA for the seeded coldkey",
            remediation="Check indexer seeding and the local demo coldkey/address values.",
        )
    )
    checks.append(
        _check(
            "emission_lookup",
            bool(emission_lookup.get("eligible"))
            and str(emission_lookup.get("account", "")).lower() == alice
            and _raw_credit(emission_lookup, "total_sota") == str(emission_amount),
            "emission lookup returns the seeded miner reward allocation",
            remediation="Check autoresearch root publication and the indexer emission allocation.",
        )
    )

    bundle = dict(evidence.get("bundle") or {})
    claim_evidence = list(bundle.get("claim_evidence") or [])
    first_evidence = dict(dict(claim_evidence[0]).get("evidence") or {}) if claim_evidence else {}
    consensus = dict(first_evidence.get("self_validation_consensus") or {})
    accepted_count = int(consensus.get("accepted_count") or 0)
    committee_count = int(consensus.get("committee_count") or 0)
    committee_size = int(consensus.get("committee_size") or 0)
    checks.append(
        _check(
            "self_validation_evidence",
            dict(evidence.get("root") or {}).get("root") == emission_root.get("root")
            and consensus.get("status") == "accepted"
            and consensus.get("frontier_gate_passed") is True
            and consensus.get("quorum_gate_passed") is True
            and accepted_count >= MIN_LOCAL_SELF_VALIDATION_COMMITTEE
            and committee_count >= MIN_LOCAL_SELF_VALIDATION_COMMITTEE
            and committee_size >= MIN_LOCAL_SELF_VALIDATION_COMMITTEE,
            (
                "autoresearch evidence shows accepted multi-user self-validation "
                f"for the seeded miner ({accepted_count}/{committee_count} accepted)"
            ),
            remediation="Check the autoresearch backend, seeded submission, peer evaluation, and evidence endpoint.",
        )
    )

    genesis_tx = dict(genesis_transaction.get("transaction") or {})
    emission_tx = dict(emission_transaction.get("transaction") or {})
    checks.append(
        _check(
            "genesis_calldata",
            str(genesis_tx.get("to", "")).lower() == contracts.get("genesis_distributor")
            and str(genesis_tx.get("data", "")).startswith("0x")
            and int(genesis_tx.get("chainId") or 0) == chain_id,
            "claims API returns unsigned genesis calldata for the local distributor",
            remediation="Check the indexer contract config and GenesisClaimDistributor proof args.",
        )
    )
    checks.append(
        _check(
            "emission_calldata",
            str(emission_tx.get("to", "")).lower() == contracts.get("emission_distributor")
            and str(emission_tx.get("data", "")).startswith("0x")
            and int(emission_tx.get("chainId") or 0) == chain_id,
            "claims API returns unsigned emission calldata for the local distributor",
            remediation="Check the indexer contract config and EmissionClaimDistributor proof args.",
        )
    )
    return checks


def capture_firefox_screenshot(url: str, path: Path, *, timeout_seconds: float) -> dict[str, str]:
    firefox = shutil.which("firefox")
    if not firefox:
        return _yellow(
            "claims_page_screenshot",
            "Firefox is not installed, so the optional page screenshot was skipped.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sota-firefox-profile-") as profile_dir:
        try:
            result = subprocess.run(
                [
                    firefox,
                    "--headless",
                    "--no-remote",
                    "--profile",
                    profile_dir,
                    "--screenshot",
                    str(path),
                    "--window-size",
                    "1440,1000",
                    url,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return _yellow(
                "claims_page_screenshot",
                f"Firefox screenshot timed out after {timeout_seconds:.0f}s.",
                remediation=f"Open {url} manually or rerun with --skip-screenshot.",
            )
    ok = result.returncode == 0 and path.exists() and path.stat().st_size > 10_000
    if ok:
        return _check("claims_page_screenshot", True, f"saved screenshot to {path}")
    return _yellow(
        "claims_page_screenshot",
        f"Firefox screenshot did not complete cleanly: {result.stderr[-300:] or result.stdout[-300:]}",
        remediation="Open the claims page manually if visual verification is needed.",
    )


def run_smoke(
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    claims_url: str | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    screenshot_path: Path = DEFAULT_SCREENSHOT_PATH,
    skip_screenshot: bool = False,
    wait_seconds: float = 30.0,
    request_timeout: float = 15.0,
) -> dict[str, Any]:
    if not state_path.exists():
        raise RuntimeError(f"local demo state not found at {state_path}; start the demo first")
    state = _load_json(state_path)
    targets = build_targets(state, claims_url=claims_url)
    wait_for_claims_page(targets["claims_page"], timeout_seconds=wait_seconds, request_timeout=request_timeout)

    html = _http_text(targets["claims_page"], timeout=request_timeout)
    genesis_lookup = _http_json("GET", targets["genesis_lookup"], timeout=request_timeout)
    emission_lookup = _http_json("GET", targets["emission_lookup"], timeout=request_timeout)
    evidence = _http_json("GET", targets["self_validation_evidence"], timeout=request_timeout)
    alice = str(state["accounts"]["alice_reward"])
    lane_id = str(state.get("autoresearch", {}).get("subnet", {}).get("id") or LANE_ID)
    genesis_transaction = _http_json(
        "POST",
        targets["genesis_transaction"],
        payload={"program": "genesis", "rewardAddress": alice},
        timeout=request_timeout,
    )
    emission_transaction = _http_json(
        "POST",
        targets["emission_transaction"],
        payload={"program": "emission", "evmAddress": alice, "laneId": lane_id},
        timeout=request_timeout,
    )
    claims_health = _http_json("GET", targets["claims_health"], timeout=request_timeout)
    _http_json("POST", targets["index_sync"], timeout=request_timeout)
    index_status = _http_json("GET", targets["index_status"], timeout=request_timeout)
    autoresearch_ready_status = _http_status("GET", targets["autoresearch_ready"], timeout=request_timeout)

    checks = validate_page_html(html)
    checks.extend(validate_live_docs_pages(targets, request_timeout=request_timeout))
    checks.extend(validate_handoff_page(targets, request_timeout=request_timeout))
    checks.extend(validate_tester_share(state))
    checks.extend(
        validate_api_payloads(
            state,
            genesis_lookup=genesis_lookup,
            emission_lookup=emission_lookup,
            evidence=evidence,
            genesis_transaction=genesis_transaction,
            emission_transaction=emission_transaction,
            claims_health=claims_health,
            index_status=index_status,
            autoresearch_ready_status=autoresearch_ready_status,
        )
    )
    if skip_screenshot:
        checks.append(_yellow("claims_page_screenshot", "screenshot skipped by --skip-screenshot"))
    else:
        checks.append(capture_firefox_screenshot(targets["claims_page"], screenshot_path, timeout_seconds=45.0))

    summary = _summary(checks)
    report = {
        "schema": "sota-local-claims-ui-smoke/v1",
        "ok": summary["red"] == 0,
        "status": "green" if summary["red"] == 0 else "red",
        "summary": summary,
        "targets": targets,
        "tester_urls": dict(state.get("urls") or {}),
        "local_account": {
            "address": state["accounts"]["alice_reward"],
            "old_coldkey": state["genesis"]["old_coldkey"],
        },
        "checks": checks,
    }
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test the local SOTA Base claims UI and proxy APIs.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH, help="local demo state JSON")
    parser.add_argument("--claims-url", help="claims page URL; defaults to http://127.0.0.1:3000/claims")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH, help="JSON report path")
    parser.add_argument("--screenshot-out", type=Path, default=DEFAULT_SCREENSHOT_PATH, help="optional screenshot path")
    parser.add_argument("--skip-screenshot", action="store_true", help="skip optional Firefox screenshot")
    parser.add_argument("--wait-seconds", type=float, default=30.0, help="time to wait for the claims page")
    parser.add_argument("--request-timeout", type=float, default=15.0, help="per-request timeout")
    args = parser.parse_args(argv)

    report = run_smoke(
        state_path=args.state,
        claims_url=args.claims_url,
        report_path=args.report_out,
        screenshot_path=args.screenshot_out,
        skip_screenshot=args.skip_screenshot,
        wait_seconds=args.wait_seconds,
        request_timeout=args.request_timeout,
    )
    summary = report["summary"]
    _print(
        f"local claims UI smoke {report['status']}: "
        f"{summary['green']} green, {summary['yellow']} yellow, {summary['red']} red"
    )
    _print(f"report: {args.report_out}")
    screenshot_check = next((check for check in report["checks"] if check["name"] == "claims_page_screenshot"), None)
    if screenshot_check:
        _print(screenshot_check["detail"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
import json
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen


REPOS = Path("/home/mekaneeky/repos")
DEFAULT_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
BASE_SEPOLIA_CHAIN_ID = 84532
BASE_MAINNET_CHAIN_ID = 8453
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
LANE_ID = "base:sota-local"
EXPECTED_TESTNET_CLAIMS_TEXT = (
    "Base Sepolia claims",
    "Keep TAO and alpha. Claim SOTA separately.",
    "SOTA claims are non-custodial proof transactions",
    "Connect wallet",
    "Switch to Base Sepolia",
    "Testnet readiness",
    "Claim sources",
    "Binding payload",
    "Sign with extension",
    "TAO credit",
    "Alpha synthetic credit",
    "Total SOTA",
    "Unclaimed SOTA",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    remediation: str = ""

    def as_dict(self) -> dict[str, str]:
        payload = {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }
        if self.remediation:
            payload["remediation"] = self.remediation
        return payload


def _check(name: str, ok: bool, detail: str, *, remediation: str = "") -> Check:
    return Check(name, "green" if ok else "red", detail, "" if ok else remediation)


def _result_check(name: str, ok: bool, success: str, failure: str, *, remediation: str = "") -> Check:
    return Check(name, "green" if ok else "red", success if ok else failure, "" if ok else remediation)


def _yellow(name: str, detail: str, *, remediation: str = "") -> Check:
    return Check(name, "yellow", detail, remediation)


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(status, 2)


def _worst(checks: list[Check]) -> str:
    if not checks:
        return "green"
    return max((check.status for check in checks), key=_status_rank)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_env(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _join_url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def _visible_text(html: str) -> str:
    without_comments = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    without_tags = re.sub(r"<[^>]+>", " ", without_comments)
    return unescape(re.sub(r"\s+", " ", without_tags))


def _is_evm_address(value: str) -> bool:
    return bool(EVM_ADDRESS_RE.fullmatch(value)) and value.lower() != ZERO_ADDRESS


def _service_url(manifest: dict[str, Any], key: str) -> str:
    service = dict(dict(manifest.get("services") or {}).get(key) or {})
    for attr in ("public_url", "public_base_url", "service_url", "dashboard_url", "health_url"):
        value = str(service.get(attr) or "").strip()
        if value:
            return value
    return ""


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


def _http_json_response(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = int(response.status)
    except HTTPError as exc:
        body = exc.read()
        status = int(exc.code)
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc
    if not body:
        return status, {}
    try:
        return status, json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return status, body.decode("utf-8", errors="replace")


def _http_status(method: str, url: str, *, timeout: float) -> int:
    request = Request(url, headers={"Accept": "application/json"}, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except HTTPError as exc:
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        return None


def _raw_credit(payload: dict[str, Any], key: str) -> str:
    credits = dict(payload.get("credits") or {})
    direct = dict(credits.get(key) or {})
    if direct:
        return str(direct.get("raw") or "")
    camel = "".join([key.split("_")[0], *[part.capitalize() for part in key.split("_")[1:]]])
    return str(dict(credits.get(camel) or {}).get("raw") or "")


def _claim_is_fully_claimed(payload: dict[str, Any]) -> bool:
    state = dict(payload.get("claim_state") or {})
    if str(state.get("status") or "").lower() == "claimed":
        return True
    claimed = _as_int(_raw_credit(payload, "claimed_sota")) or 0
    total = _as_int(_raw_credit(payload, "total_sota")) or 0
    unclaimed = _as_int(_raw_credit(payload, "unclaimed_sota")) or 0
    return total > 0 and claimed >= total and unclaimed == 0


def _already_claimed_error(exc: Exception) -> bool:
    return "already_claimed" in str(exc)


def _readiness_payload(args: argparse.Namespace, readiness_url: str) -> dict[str, Any]:
    local = _load_json(args.readiness_file)
    if local:
        return local
    if not readiness_url:
        return {}
    return dict(_http_json("GET", readiness_url, timeout=args.timeout) or {})


def _config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    manifest = _load_json(args.manifest)
    env = _load_env(args.env_file)
    claims_ui = (
        args.claims_url
        or _service_url(manifest, "claims_ui")
        or env.get("SOTA_CLAIMS_UI_URL")
        or env.get("NEXT_PUBLIC_SOTA_CLAIMS_UI_URL")
        or ""
    )
    if claims_ui and not claims_ui.rstrip("/").endswith("/claims"):
        claims_ui = _join_url(claims_ui, "/claims")
    claims_api = args.claims_api_url or env.get("SOTA_CLAIMS_API_URL") or env.get("NEXT_PUBLIC_SOTA_CLAIMS_API_URL") or _service_url(manifest, "indexer_api")
    autoresearch = (
        args.autoresearch_url
        or env.get("SOTA_COORDINATOR_URL")
        or env.get("NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL")
        or _service_url(manifest, "attestation_builder")
    )
    readiness_url = (
        args.readiness_url
        or env.get("NEXT_PUBLIC_SOTA_READINESS_URL")
        or env.get("SOTA_READINESS_URL")
        or ""
    )
    values = {
        "claims_url": claims_ui,
        "claims_api_url": claims_api.rstrip("/") if claims_api else "",
        "autoresearch_url": autoresearch.rstrip("/") if autoresearch else "",
        "readiness_url": readiness_url,
        "test_wallet_address": args.test_wallet_address or env.get("SOTA_TEST_WALLET_ADDRESS") or env.get("TEST_WALLET_ADDRESS") or "",
        "test_old_coldkey": args.test_old_coldkey or env.get("SOTA_TEST_OLD_COLDKEY") or env.get("TEST_OLD_COLDKEY") or "",
        "snapshot_coldkey": getattr(args, "test_snapshot_coldkey", "") or env.get("SOTA_TEST_SNAPSHOT_COLDKEY") or env.get("SOTA_TEST_OLD_COLDKEY") or env.get("TEST_OLD_COLDKEY") or "",
        "lane_id": args.lane_id or env.get("SOTA_TEST_LANE_ID") or env.get("NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID") or LANE_ID,
        "epoch": str(args.epoch or env.get("SOTA_TEST_EPOCH") or "1"),
        "env_chain_id": env.get("NEXT_PUBLIC_SOTA_BASE_CHAIN_ID") or env.get("SOTA_BASE_CHAIN_ID") or "",
        "manifest_chain_id": str(dict(manifest.get("chain") or {}).get("chain_id") or ""),
    }
    return manifest, env, values


def _artifact_checks(args: argparse.Namespace, manifest: dict[str, Any], env: dict[str, str], values: dict[str, str]) -> list[Check]:
    checks: list[Check] = []
    manifest_ok = args.manifest.exists() and bool(manifest)
    checks.append(
        _result_check(
            "manifest_artifact",
            manifest_ok,
            f"{args.manifest} exists and is parseable.",
            f"Missing or invalid manifest artifact: {args.manifest}.",
            remediation="Run the guarded Base Sepolia rehearsal and keep the deployment manifest.",
        )
    )
    env_ok = args.env_file.exists() and bool(env)
    checks.append(
        _result_check(
            "env_artifact",
            env_ok,
            f"{args.env_file} exists and is parseable.",
            f"Missing or invalid env artifact: {args.env_file}.",
            remediation="Generate base-sota.env.testnet from the Base Sepolia manifest.",
        )
    )
    manifest_chain_id = _as_int(values["manifest_chain_id"])
    env_chain_id = _as_int(values["env_chain_id"])
    if manifest_chain_id == BASE_MAINNET_CHAIN_ID or env_chain_id == BASE_MAINNET_CHAIN_ID:
        checks.append(Check("chain_config", "red", "Base mainnet chain id 8453 is present.", "Use Base Sepolia chain id 84532 only."))
    else:
        chain_ok = manifest_chain_id == BASE_SEPOLIA_CHAIN_ID and env_chain_id == BASE_SEPOLIA_CHAIN_ID
        checks.append(
            _result_check(
                "chain_config",
                chain_ok,
                "Manifest and env are pinned to Base Sepolia chain id 84532.",
                f"Manifest chain id is {values['manifest_chain_id'] or 'missing'} and env chain id is {values['env_chain_id'] or 'missing'}, expected 84532.",
                remediation="Regenerate the manifest/env for Base Sepolia before public browser smoke.",
            )
        )
    return checks


def _config_checks(values: dict[str, str]) -> list[Check]:
    checks = [
        _result_check("claims_ui_url", bool(values["claims_url"]), "Claims UI URL is configured.", "Claims UI URL is missing.", remediation="Set the public Base Sepolia claims UI URL."),
        _result_check("claims_api_url", bool(values["claims_api_url"]), "Claims API URL is configured.", "Claims API URL is missing.", remediation="Set the public Base Sepolia claims API URL."),
        _result_check("autoresearch_url", bool(values["autoresearch_url"]), "Autoresearch/coordinator URL is configured.", "Autoresearch/coordinator URL is missing.", remediation="Set the public Base Sepolia autoresearch coordinator URL."),
        _result_check("readiness_url", bool(values["readiness_url"]), "Public readiness URL is configured.", "Public readiness URL is missing.", remediation="Publish and configure base-sota-testnet-readiness.json."),
        _result_check("test_old_coldkey", bool(values["test_old_coldkey"]), "Seeded test old coldkey is configured.", "Seeded test old coldkey is missing.", remediation="Set SOTA_TEST_OLD_COLDKEY for the public test claim."),
        _result_check("snapshot_coldkey", bool(values["snapshot_coldkey"]), "Snapshot binding smoke coldkey is configured.", "Snapshot binding smoke coldkey is missing.", remediation="Set SOTA_TEST_SNAPSHOT_COLDKEY to a real coldkey from the locked snapshot."),
        _result_check("test_wallet_address", _is_evm_address(values["test_wallet_address"]), "Seeded test wallet address is configured.", f"Seeded test wallet address is missing or invalid: {values['test_wallet_address'] or 'missing'}.", remediation="Set SOTA_TEST_WALLET_ADDRESS to a public funded Base Sepolia wallet."),
        _result_check("test_lane_id", bool(values["lane_id"]), "Seeded emission lane id is configured.", "Seeded emission lane id is missing.", remediation="Set NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID or SOTA_TEST_LANE_ID."),
    ]
    return checks


def _claims_page_check(claims_url: str, *, timeout: float) -> Check:
    if not claims_url:
        return Check("claims_page_text", "red", "Claims UI URL is missing.", "Set claims UI URL before browser smoke.")
    try:
        html = _http_text(claims_url, timeout=timeout)
    except Exception as exc:
        return Check("claims_page_text", "red", f"{claims_url} failed: {exc}", "Deploy the claims UI and make it public.")
    visible_text = _visible_text(html)
    missing = [text for text in EXPECTED_TESTNET_CLAIMS_TEXT if text not in visible_text]
    return _result_check(
        "claims_page_text",
        not missing,
        "Claims UI exposes Base Sepolia wallet, readiness, source, and credit text.",
        f"Claims UI is missing: {', '.join(missing)}",
        remediation=f"Claims UI is missing: {', '.join(missing)}",
    )


def _readiness_check(args: argparse.Namespace, readiness_url: str) -> Check:
    try:
        payload = _readiness_payload(args, readiness_url)
    except Exception as exc:
        return Check("readiness_report", "red", f"Readiness report failed: {exc}", "Publish the public readiness artifact before browser smoke.")
    ok = (
        payload.get("schema") == "sota-base-testnet-readiness/v1"
        and payload.get("ok") is True
        and payload.get("status") == "green"
    )
    return _result_check(
        "readiness_report",
        ok,
        "Readiness report is green and uses schema sota-base-testnet-readiness/v1.",
        f"Readiness report is not green: schema={payload.get('schema')!r}, status={payload.get('status')!r}, ok={payload.get('ok')!r}.",
        remediation="Clear preflight blockers, regenerate readiness, and publish the green readiness artifact.",
    )


def _claims_api_checks(values: dict[str, str], *, timeout: float) -> list[Check]:
    checks: list[Check] = []
    base = values["claims_api_url"]
    if not base:
        return [Check("claims_api_health", "red", "Claims API URL is missing.", "Set claims API URL before browser smoke.")]
    try:
        health = dict(_http_json("GET", _join_url(base, "/health"), timeout=timeout) or {})
    except Exception as exc:
        checks.append(Check("claims_api_health", "red", f"Claims API health failed: {exc}", "Deploy the public testnet claims API."))
    else:
        checks.append(
            _result_check(
                "claims_api_health",
                str(health.get("status")).lower() == "ok" and _as_int(health.get("chain_id")) == BASE_SEPOLIA_CHAIN_ID,
                "Claims API health reports Base Sepolia.",
                f"Claims API health returned status={health.get('status')!r}, chain_id={health.get('chain_id')!r}; expected ok/84532.",
                remediation="Fix claims API health or Base Sepolia chain config.",
            )
        )
    try:
        sync = dict(_http_json("POST", _join_url(base, "/api/v1/base/index/sync"), timeout=timeout) or {})
        status = dict(_http_json("GET", _join_url(base, "/api/v1/base/index/status"), timeout=timeout) or {})
    except Exception as exc:
        checks.append(Check("indexer_status", "red", f"Indexer status failed: {exc}", "Deploy/sync the public Base Sepolia indexer."))
    else:
        required = {"root_registry", "lane_registry", "genesis_distributor", "emission_distributor"}
        configured = {str(item) for item in status.get("contracts_configured") or []}
        sync_lag = _as_int(sync.get("lag_blocks"))
        status_lag = _as_int(status.get("lag_blocks"))
        lag_ok = sync_lag == 0 or status_lag == 0
        index_ok = (
            _as_int(status.get("chain_id")) == BASE_SEPOLIA_CHAIN_ID
            and configured >= required
            and lag_ok
            and not sync.get("last_sync_error")
            and not status.get("last_sync_error")
        )
        checks.append(
            _result_check(
                "indexer_status",
                index_ok,
                "Indexer reports Base Sepolia RPC sync, all contract roles, and a zero-lag sync result.",
                (
                    f"Indexer status chain_id={status.get('chain_id')!r}, "
                    f"contracts={sorted(configured)}, sync_lag={sync.get('lag_blocks')!r}, "
                    f"status_lag={status.get('lag_blocks')!r}, "
                    f"sync_error={sync.get('last_sync_error')!r}, "
                    f"last_sync_error={status.get('last_sync_error')!r}."
                ),
                remediation="Fix indexer contract config, RPC sync, or lag before browser smoke.",
            )
        )
    return checks


def _binding_route_checks(values: dict[str, str], *, timeout: float) -> list[Check]:
    checks: list[Check] = []
    base = values["claims_api_url"]
    wallet = values["test_wallet_address"]
    snapshot_coldkey = values["snapshot_coldkey"]
    if not base or not _is_evm_address(wallet) or not snapshot_coldkey:
        return [
            Check(
                "genesis_binding_inputs",
                "red",
                "Claims API URL, test wallet, and snapshot coldkey are required for binding-route smoke.",
                "Set public test claim inputs before inviting a nontechnical tester.",
            )
        ]
    try:
        binding = dict(
            _http_json(
                "POST",
                _join_url(base, "/api/v1/base/genesis/binding-message"),
                payload={"coldkey": snapshot_coldkey, "reward_address": wallet},
                timeout=timeout,
            )
            or {}
        )
    except Exception as exc:
        checks.append(
            Check(
                "genesis_binding_message",
                "red",
                f"Genesis binding-message route failed: {exc}",
                "Deploy the claims API with SOTA_BASE_SNAPSHOT_DIR and a real TAO/alpha snapshot.",
            )
        )
        return checks
    message = dict(binding.get("message") or {})
    claim = dict(binding.get("snapshot_claim") or {})
    amount = _as_int(message.get("allocation_amount"))
    binding_ok = (
        binding.get("schema") == "sota-snapshot-binding-message/v1"
        and str(binding.get("signing_payload") or "")
        and message.get("coldkey") == snapshot_coldkey
        and str(message.get("reward_address") or "").lower() == wallet.lower()
        and _as_int(message.get("base_chain_id")) == BASE_SEPOLIA_CHAIN_ID
        and amount is not None
        and amount > 0
        and "direct_tao_rao" in claim
        and "alpha_credit_rao" in claim
        and "alpha_credit_rao_by_netuid" in claim
    )
    checks.append(
        _result_check(
            "genesis_binding_message",
            binding_ok,
            "Claims API returns a Base Sepolia coldkey binding payload with TAO and alpha snapshot fields.",
            (
                f"Binding message is incomplete: schema={binding.get('schema')!r}, "
                f"chain={message.get('base_chain_id')!r}, amount={message.get('allocation_amount')!r}, "
                f"claim_fields={sorted(claim)}."
            ),
            remediation="Deploy the binding-message endpoint and configure the locked TAO/alpha snapshot.",
        )
    )
    if not binding_ok:
        return checks
    try:
        status, response = _http_json_response(
            "POST",
            _join_url(base, "/api/v1/base/genesis/bindings"),
            payload={"message": message, "signature": "0x" + "00" * 64},
            timeout=timeout,
        )
    except Exception as exc:
        checks.append(
            Check(
                "genesis_binding_submit_route",
                "red",
                f"Genesis binding submit route failed: {exc}",
                "Deploy the signed-binding submit route on the public claims API.",
            )
        )
        return checks
    detail = dict(response.get("detail") or {}) if isinstance(response, dict) else {}
    checks.append(
        _result_check(
            "genesis_binding_submit_route",
            status == 422 and detail.get("code") == "invalid_binding_signature",
            "Claims API signed-binding route is live and rejects an invalid coldkey signature.",
            f"Signed-binding route returned HTTP {status} with code {detail.get('code')!r}; expected 422 invalid_binding_signature.",
            remediation="Deploy the signed-binding submit route and keep invalid signatures rejected.",
        )
    )
    return checks


def _claim_lookup_checks(values: dict[str, str], *, timeout: float) -> list[Check]:
    checks: list[Check] = []
    base = values["claims_api_url"]
    wallet = values["test_wallet_address"]
    old_coldkey = values["test_old_coldkey"]
    lane_id = values["lane_id"]
    if not base or not _is_evm_address(wallet) or not old_coldkey:
        return [
            Check(
                "seeded_claim_inputs",
                "red",
                "Claims API URL, funded test wallet, and old coldkey are required for public browser smoke.",
                "Set public test claim inputs before inviting a nontechnical tester.",
            )
        ]
    genesis_query = urlencode({"old_coldkey": old_coldkey, "reward_address": wallet, "subnet_id": "genesis"})
    emission_query = urlencode({"evm_address": wallet, "subnet_id": lane_id})
    try:
        genesis = dict(_http_json("GET", f"{_join_url(base, f'/api/v1/base/eligibility/{quote(wallet)}')}?{genesis_query}", timeout=timeout) or {})
    except Exception as exc:
        checks.append(Check("genesis_lookup", "red", f"Genesis lookup failed: {exc}", "Publish/import the Base Sepolia genesis claim artifact."))
        genesis = {}
    checks.append(
        _result_check(
            "genesis_lookup",
            bool(genesis.get("eligible")) and _raw_credit(genesis, "total_sota") not in {"", "0"},
            "Seeded wallet has an eligible genesis SOTA claim.",
            f"Seeded wallet genesis lookup is not claimable: eligible={genesis.get('eligible')!r}, total_sota={_raw_credit(genesis, 'total_sota') or 'missing'}.",
            remediation="Seed the test coldkey/wallet into the public genesis claim artifact and indexer.",
        )
    )
    try:
        emission = dict(_http_json("GET", f"{_join_url(base, f'/api/v1/base/eligibility/{quote(wallet)}')}?{emission_query}", timeout=timeout) or {})
    except Exception as exc:
        checks.append(Check("emission_lookup", "red", f"Emission lookup failed: {exc}", "Publish/import the Base Sepolia emission claim artifact."))
        emission = {}
    checks.append(
        _result_check(
            "emission_lookup",
            bool(emission.get("eligible")) and _raw_credit(emission, "total_sota") not in {"", "0"},
            "Seeded wallet has an eligible mined-emission SOTA claim.",
            f"Seeded wallet emission lookup is not claimable: eligible={emission.get('eligible')!r}, total_sota={_raw_credit(emission, 'total_sota') or 'missing'}.",
            remediation="Run a self-validated test competition and import its emission claim artifact.",
        )
    )
    for label, payload, request_payload, remediation in (
        (
            "genesis",
            genesis,
            {"program": "genesis", "rewardAddress": wallet},
            "Fix genesis distributor address, proof args, or chain id in the claims API.",
        ),
        (
            "emission",
            emission,
            {"program": "emission", "evmAddress": wallet, "laneId": lane_id},
            "Fix emission distributor address, proof args, or chain id in the claims API.",
        ),
    ):
        try:
            tx_response = dict(
                _http_json(
                    "POST",
                    _join_url(base, "/api/v1/base/claims/transaction"),
                    payload=request_payload,
                    timeout=timeout,
                )
                or {}
            )
            tx = dict(tx_response.get("transaction") or {})
            checks.append(
                _result_check(
                    f"{label}_calldata",
                    str(tx.get("data") or "").startswith("0x") and _as_int(tx.get("chainId")) == BASE_SEPOLIA_CHAIN_ID,
                    f"Claims API returns unsigned {label} calldata for Base Sepolia.",
                    f"{label.capitalize()} transaction builder returned chainId={tx.get('chainId')!r}, data_prefix={str(tx.get('data') or '')[:10]!r}.",
                    remediation=remediation,
                )
            )
        except Exception as exc:
            if _already_claimed_error(exc) and _claim_is_fully_claimed(payload):
                checks.append(
                    Check(
                        f"{label}_calldata",
                        "green",
                        f"{label.capitalize()} claim is already complete for the seeded wallet; transaction builder correctly refuses a duplicate claim.",
                    )
                )
            else:
                checks.append(Check(f"{label}_calldata", "red", f"{label.capitalize()} calldata failed: {exc}", f"Fix transaction builder for {label} claims."))
    return checks


def _autoresearch_checks(values: dict[str, str], *, timeout: float) -> list[Check]:
    base = values["autoresearch_url"]
    lane_id = values["lane_id"]
    epoch = values["epoch"]
    if not base:
        return [Check("autoresearch_ready", "red", "Autoresearch URL is missing.", "Deploy/configure the public autoresearch coordinator.")]
    checks: list[Check] = []
    try:
        ready_status = _http_status("GET", _join_url(base, "/readyz"), timeout=timeout)
    except Exception as exc:
        checks.append(Check("autoresearch_ready", "red", f"Autoresearch readiness failed: {exc}", "Deploy the public autoresearch coordinator."))
    else:
        checks.append(
            _result_check(
                "autoresearch_ready",
                200 <= ready_status < 300,
                "Autoresearch readiness endpoint is reachable.",
                f"Autoresearch readiness returned HTTP {ready_status}.",
                remediation="Fix public autoresearch readiness before browser smoke.",
            )
        )
    try:
        evidence = dict(
            _http_json(
                "GET",
                _join_url(base, f"/api/v1/sota/subnets/{quote(lane_id, safe='')}/epochs/{epoch}/evidence"),
                timeout=timeout,
            )
            or {}
        )
        bundle = dict(evidence.get("bundle") or {})
        claim_evidence = list(bundle.get("claim_evidence") or [])
        consensus_values = [
            dict(dict(item).get("evidence") or {}).get("self_validation_consensus")
            for item in claim_evidence
            if isinstance(item, dict)
        ]
        accepted = any(dict(item or {}).get("status") == "accepted" for item in consensus_values)
        checks.append(
            _result_check(
                "self_validation_evidence",
                bool(dict(evidence.get("root") or {}).get("root")) and accepted,
                "Autoresearch evidence exposes an accepted self-validation claim root.",
                f"Self-validation evidence missing root or accepted consensus: root={dict(evidence.get('root') or {}).get('root')!r}, accepted={accepted}.",
                remediation="Run/publish the self-validated Base Sepolia test competition evidence.",
            )
        )
    except Exception as exc:
        checks.append(Check("self_validation_evidence", "red", f"Self-validation evidence failed: {exc}", "Publish self-validation evidence for the seeded test emission."))
    return checks


def run_browser_smoke(args: argparse.Namespace) -> dict[str, Any]:
    manifest, env, values = _config(args)
    checks: list[Check] = []
    checks.extend(_artifact_checks(args, manifest, env, values))
    checks.extend(_config_checks(values))
    checks.append(_readiness_check(args, values["readiness_url"]))
    checks.append(_claims_page_check(values["claims_url"], timeout=args.timeout))
    checks.extend(_claims_api_checks(values, timeout=args.timeout))
    checks.extend(_binding_route_checks(values, timeout=args.timeout))
    checks.extend(_claim_lookup_checks(values, timeout=args.timeout))
    checks.extend(_autoresearch_checks(values, timeout=args.timeout))
    checks.append(
        Check(
            "wallet_transaction_handoff",
            "green",
            "Automated smoke is read-only; after all checks are green, a human tester submits genesis and emission in MetaMask.",
        )
    )
    status = _worst(checks)
    return {
        "schema": "sota-base-testnet-browser-smoke/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": status == "green" or (status == "yellow" and args.allow_yellow),
        "status": status,
        "read_only": True,
        "does_not": ["deploy", "sign", "broadcast_transactions", "touch_production_bittensor"],
        "message": (
            "Base Sepolia public UI/API are ready for a human MetaMask smoke."
            if status == "green"
            else "Base Sepolia public UI/API are not ready for a nontechnical MetaMask tester."
        ),
        "manifest": str(args.manifest),
        "env_file": str(args.env_file),
        "targets": values,
        "checks": [check.as_dict() for check in checks],
        "summary": {
            "green": sum(1 for check in checks if check.status == "green"),
            "yellow": sum(1 for check in checks if check.status == "yellow"),
            "red": sum(1 for check in checks if check.status == "red"),
        },
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"Base SOTA testnet browser smoke: {report['status'].upper()}")
    print(report["message"])
    print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
    for check in report["checks"]:
        print(f"- [{check['status']}] {check['name']}: {check['detail']}")
        if check.get("remediation"):
            print(f"  next: {check['remediation']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only public Base Sepolia browser-wallet smoke precheck.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--readiness-file", type=Path)
    parser.add_argument("--claims-url", default="")
    parser.add_argument("--claims-api-url", default="")
    parser.add_argument("--autoresearch-url", default="")
    parser.add_argument("--readiness-url", default="")
    parser.add_argument("--test-wallet-address", default="")
    parser.add_argument("--test-old-coldkey", default="")
    parser.add_argument("--test-snapshot-coldkey", default="")
    parser.add_argument("--lane-id", default="")
    parser.add_argument("--epoch", default="")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--allow-yellow", action="store_true", help="Treat yellow-only reports as exit 0.")
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when red checks remain.")
    args = parser.parse_args(argv)

    args.manifest = args.manifest or args.artifacts_dir / "base-sepolia-deployment-manifest.json"
    args.env_file = args.env_file or args.artifacts_dir / "base-sota.env.testnet"
    args.readiness_file = args.readiness_file or args.artifacts_dir / "base-sota-testnet-readiness.json"

    report = run_browser_smoke(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text, encoding="utf-8")
    if args.json:
        print(text, end="")
    else:
        _print_text(report)
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

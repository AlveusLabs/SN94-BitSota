#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


REPOS = Path("/home/mekaneeky/repos")
RUN_DIR = REPOS / ".sota-base-local"
DEFAULT_STATE_PATH = RUN_DIR / "state.json"
DEFAULT_REPORT_PATH = RUN_DIR / "tailscale-preflight.json"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    remediation: str = ""

    def as_dict(self) -> dict[str, str]:
        payload = {"name": self.name, "status": self.status, "detail": self.detail}
        if self.remediation:
            payload["remediation"] = self.remediation
        return payload


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_json(command: list[str], *, timeout: float) -> tuple[dict[str, Any], str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        detail = exc.stderr or exc.stdout or f"{' '.join(command)} timed out after {timeout:.0f}s"
        return {}, str(detail).strip()
    if result.returncode != 0:
        return {}, (result.stderr.strip() or result.stdout.strip() or f"{' '.join(command)} exited {result.returncode}")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"{' '.join(command)} returned invalid JSON: {exc}"
    return payload if isinstance(payload, dict) else {}, ""


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}.get(status, 2)


def _worst(checks: list[Check]) -> str:
    if not checks:
        return "green"
    return max((check.status for check in checks), key=_status_rank)


def _summary(checks: list[Check]) -> dict[str, int]:
    return {
        "green": sum(1 for check in checks if check.status == "green"),
        "yellow": sum(1 for check in checks if check.status == "yellow"),
        "red": sum(1 for check in checks if check.status == "red"),
    }


def _tailscale_dns_name(status_payload: dict[str, Any]) -> str:
    self_info = dict(status_payload.get("Self") or {})
    dns_name = str(self_info.get("DNSName") or "").strip().rstrip(".")
    if dns_name:
        return dns_name
    host_name = str(self_info.get("HostName") or "").strip()
    suffix = str(status_payload.get("MagicDNSSuffix") or "").strip().strip(".")
    if host_name and suffix:
        return f"{host_name}.{suffix}"
    return ""


def _serve_enable_url(status_payload: dict[str, Any]) -> str:
    node_id = str(dict(status_payload.get("Self") or {}).get("ID") or "").strip()
    return f"https://login.tailscale.com/f/serve?node={node_id}" if node_id else ""


def _cert_domains(status_payload: dict[str, Any]) -> list[str]:
    value = status_payload.get("CertDomains")
    if not isinstance(value, list):
        return []
    return [str(item).strip().rstrip(".") for item in value if str(item).strip()]


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[Check] = []
    tailscale = shutil.which("tailscale")
    checks.append(
        Check(
            "tailscale_binary",
            "green" if tailscale else "red",
            f"tailscale CLI found at {tailscale}." if tailscale else "tailscale CLI is not installed or not on PATH.",
            "" if tailscale else "Install and authenticate Tailscale on the demo host.",
        )
    )
    status_payload: dict[str, Any] = {}
    serve_payload: dict[str, Any] = {}
    status_error = ""
    serve_error = ""
    if tailscale:
        status_payload, status_error = _run_json([tailscale, "status", "--json"], timeout=args.timeout)
        serve_payload, serve_error = _run_json([tailscale, "serve", "status", "--json"], timeout=args.timeout)
    backend = str(status_payload.get("BackendState") or "")
    checks.append(
        Check(
            "tailscale_running",
            "green" if backend == "Running" else "red",
            "Tailscale backend is running." if backend == "Running" else status_error or f"Tailscale backend state is {backend or 'unknown'}.",
            "" if backend == "Running" else "Start and authenticate Tailscale before remote browser-wallet testing.",
        )
    )
    dns_name = _tailscale_dns_name(status_payload)
    checks.append(
        Check(
            "magicdns_name",
            "green" if dns_name else "red",
            f"Tailscale MagicDNS name is {dns_name}." if dns_name else "Tailscale MagicDNS name is unavailable.",
            "" if dns_name else "Enable Tailscale MagicDNS for the tailnet and reconnect this node.",
        )
    )
    cert_domains = _cert_domains(status_payload)
    https_enabled = bool(dns_name and dns_name in cert_domains)
    enable_url = _serve_enable_url(status_payload)
    checks.append(
        Check(
            "tailscale_https_certificates",
            "green" if https_enabled else "red",
            (
                f"Tailscale HTTPS certificates include {dns_name}."
                if https_enabled
                else "Tailscale HTTPS certificates are not enabled for this node."
            ),
            (
                ""
                if https_enabled
                else (
                    f"Enable Tailscale Serve/HTTPS certificates for this node: {enable_url}"
                    if enable_url
                    else "Enable Tailscale Serve/HTTPS certificates in the Tailscale admin console."
                )
            ),
        )
    )
    state = _load_json(args.state)
    sharing = dict(state.get("sharing") or {})
    urls = dict(state.get("urls") or {})
    wallet_safe = bool(sharing.get("wallet_rpc_browser_safe"))
    mode = str(sharing.get("mode") or "")
    checks.append(
        Check(
            "local_wallet_rpc_share",
            "green" if wallet_safe and mode == "tailscale-https" else "yellow",
            (
                f"Local demo is sharing a browser-safe RPC at {urls.get('anvil_rpc')}."
                if wallet_safe and mode == "tailscale-https"
                else f"Local demo share mode is {mode or 'unknown'}; RPC {urls.get('anvil_rpc') or 'missing'} may not be accepted by MetaMask on another computer."
            ),
            (
                ""
                if wallet_safe and mode == "tailscale-https"
                else "After Tailscale HTTPS is enabled, rerun ./scripts/sota_local_demo.py launch --share-mode tailscale-https."
            ),
        )
    )
    serve_configured = bool(serve_payload)
    checks.append(
        Check(
            "tailscale_serve_config",
            "green" if serve_configured else "yellow",
            "Tailscale Serve has active configuration." if serve_configured else serve_error or "Tailscale Serve has no active configuration.",
            "" if serve_configured else "Rerun the local demo with --share-mode tailscale-https after enabling Tailscale HTTPS/operator access.",
        )
    )
    status = _worst(checks)
    report = {
        "schema": "sota-local-tailscale-preflight/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ok": status == "green",
        "status": status,
        "message": (
            "Tailscale HTTPS sharing is ready for remote MetaMask testing."
            if status == "green"
            else "Tailscale HTTPS sharing is not ready for remote MetaMask testing."
        ),
        "tailscale_dns_name": dns_name,
        "serve_enable_url": enable_url,
        "cert_domains": cert_domains,
        "local_urls": urls,
        "sharing": sharing,
        "checks": [check.as_dict() for check in checks],
        "summary": _summary(checks),
        "next_actions": [check.remediation for check in checks if check.status != "green" and check.remediation],
    }
    _write_json(args.report_out, report)
    return report


def _print_text(report: dict[str, Any]) -> None:
    print(f"SOTA local Tailscale preflight: {str(report['status']).upper()}")
    print(report["message"])
    print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
    for check in report["checks"]:
        print(f"- [{check['status']}] {check['name']}: {check['detail']}")
        if check.get("remediation"):
            print(f"  next: {check['remediation']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Tailscale HTTPS preflight for remote local MetaMask testing.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args(argv)
    report = run_preflight(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

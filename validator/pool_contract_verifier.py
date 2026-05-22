from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Mapping, Sequence

import requests


DEFAULT_POOL_URL = "https://pool.bitsota.com"
DEFAULT_STATUS_PATH = "/status"
DEFAULT_CLAIMS_PATH = "/claims"


@dataclass(frozen=True, slots=True)
class PoolContractCheck:
    ok: bool
    errors: list[str]
    warnings: list[str]
    status_url: str
    claims_epochs_url: str
    status: str | None
    window_number: int | None
    current_block: int | None
    active_veto_count: int | None
    is_veto_active: bool | None
    claim_ready: bool | None
    claim_epochs: list[int]
    default_claim_epoch: int | None
    publisher: str | None
    publisher_running: bool
    verifier_count: int | None


def _join_url(base_url: str, path: str) -> str:
    return f"{str(base_url).rstrip('/')}/{str(path).lstrip('/')}"


def _json_get(
    *,
    url: str,
    timeout_s: float,
    session: requests.Session | None = None,
) -> Mapping[str, Any]:
    client = session or requests.Session()
    response = client.get(url, timeout=max(float(timeout_s), 1.0))
    if response.status_code >= 400:
        raise RuntimeError(f"GET {url} failed: HTTP {response.status_code} ({response.text})")
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"GET {url} returned non-object JSON")
    return payload


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _publisher_running(processes: Any) -> bool:
    if not isinstance(processes, Sequence) or isinstance(processes, (str, bytes)):
        return False
    for process in processes:
        if not isinstance(process, Mapping):
            continue
        if str(process.get("name") or "") == "publisher" and bool(process.get("running")):
            return True
    return False


def check_pool_contract(
    *,
    pool_url: str = DEFAULT_POOL_URL,
    status_path: str = DEFAULT_STATUS_PATH,
    claims_path: str = DEFAULT_CLAIMS_PATH,
    timeout_s: float = 10.0,
    allow_active_veto: bool = False,
    require_claimable_epoch: bool = False,
    expected_publisher: str = "",
    session: requests.Session | None = None,
) -> PoolContractCheck:
    status_url = _join_url(pool_url, status_path)
    claims_epochs_url = _join_url(_join_url(pool_url, claims_path), "/epochs")

    errors: list[str] = []
    warnings: list[str] = []

    status_payload = _json_get(url=status_url, timeout_s=timeout_s, session=session)
    epochs_payload = _json_get(url=claims_epochs_url, timeout_s=timeout_s, session=session)

    onchain = status_payload.get("onchain_runtime")
    if not isinstance(onchain, Mapping):
        onchain = {}
        errors.append("status missing onchain_runtime")

    contract_status = onchain.get("contract_status")
    if not isinstance(contract_status, Mapping):
        contract_status = {}
        errors.append("onchain_runtime missing contract_status")

    proof_api = onchain.get("proof_api")
    if not isinstance(proof_api, Mapping):
        proof_api = {}
        errors.append("onchain_runtime missing proof_api")

    accounts = onchain.get("accounts")
    if not isinstance(accounts, Mapping):
        accounts = {}

    diagnostics = onchain.get("verifier_diagnostics")
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}

    if str(status_payload.get("status") or "") != "healthy":
        errors.append(f"pool status is not healthy: {status_payload.get('status')!r}")

    if onchain.get("enabled") is not True:
        errors.append("onchain runtime is not enabled")

    if proof_api.get("enabled") is not True:
        errors.append("proof API is not enabled")

    read_error = contract_status.get("read_error")
    if read_error:
        errors.append(f"contract read_error is set: {read_error}")

    active_veto_count = _int_or_none(contract_status.get("active_veto_count"))
    is_veto_active = contract_status.get("is_veto_active")
    veto_count_active = active_veto_count is not None and active_veto_count > 0
    if (bool(is_veto_active) or veto_count_active) and not allow_active_veto:
        errors.append(f"contract veto is active: active_veto_count={active_veto_count}")

    processes = onchain.get("processes")
    publisher_running = _publisher_running(processes)
    if not publisher_running:
        errors.append("Pool publisher process is not running")

    publisher = str(accounts.get("publisher") or "").strip() or None
    if expected_publisher:
        if publisher != expected_publisher:
            errors.append(
                f"publisher mismatch: expected {expected_publisher}, got {publisher or '<empty>'}"
            )

    raw_epochs = epochs_payload.get("epochs") or []
    claim_epochs: list[int] = []
    if isinstance(raw_epochs, Sequence) and not isinstance(raw_epochs, (str, bytes)):
        for raw_epoch in raw_epochs:
            epoch = _int_or_none(raw_epoch)
            if epoch is not None:
                claim_epochs.append(epoch)
    else:
        errors.append("/claims/epochs returned invalid epochs list")

    if require_claimable_epoch and not claim_epochs:
        errors.append("no claimable Pool/Merkle epoch is currently exposed")
    elif not claim_epochs:
        warnings.append("no claimable Pool/Merkle epoch is currently exposed")

    default_claim_epoch = _int_or_none(epochs_payload.get("default_epoch"))
    verifier_count = _int_or_none(diagnostics.get("verifier_count"))

    return PoolContractCheck(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        status_url=status_url,
        claims_epochs_url=claims_epochs_url,
        status=str(status_payload.get("status") or "") or None,
        window_number=_int_or_none(status_payload.get("window_number")),
        current_block=_int_or_none(status_payload.get("current_block")),
        active_veto_count=active_veto_count,
        is_veto_active=bool(is_veto_active) if is_veto_active is not None else None,
        claim_ready=(
            bool(contract_status.get("claim_ready"))
            if contract_status.get("claim_ready") is not None
            else None
        ),
        claim_epochs=claim_epochs,
        default_claim_epoch=default_claim_epoch,
        publisher=publisher,
        publisher_running=publisher_running,
        verifier_count=verifier_count,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check SN94 Pool/Merkle contract publication and claim health."
    )
    parser.add_argument("--pool-url", default=DEFAULT_POOL_URL)
    parser.add_argument("--status-path", default=DEFAULT_STATUS_PATH)
    parser.add_argument("--claims-path", default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a short text summary.")
    parser.add_argument("--allow-active-veto", action="store_true")
    parser.add_argument("--require-claimable-epoch", action="store_true")
    parser.add_argument("--expected-publisher", default="")
    return parser


def _print_text(check: PoolContractCheck) -> None:
    status = "ok" if check.ok else "failed"
    print(
        "[pool-contract] "
        f"status={status} pool_status={check.status} "
        f"window={check.window_number} block={check.current_block} "
        f"veto_active={check.is_veto_active} active_veto_count={check.active_veto_count} "
        f"claim_ready={check.claim_ready} claim_epochs={check.claim_epochs} "
        f"publisher={check.publisher} publisher_running={check.publisher_running} "
        f"verifier_count={check.verifier_count}",
        flush=True,
    )
    for warning in check.warnings:
        print(f"[pool-contract] warning: {warning}", flush=True)
    for error in check.errors:
        print(f"[pool-contract] error: {error}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    exit_code = 0
    while True:
        try:
            check = check_pool_contract(
                pool_url=args.pool_url,
                status_path=args.status_path,
                claims_path=args.claims_path,
                timeout_s=float(args.timeout_s),
                allow_active_veto=bool(args.allow_active_veto),
                require_claimable_epoch=bool(args.require_claimable_epoch),
                expected_publisher=str(args.expected_publisher or "").strip(),
            )
        except Exception as exc:
            print(f"[pool-contract] failed: {exc}", flush=True)
            check = None
            exit_code = 1
        else:
            exit_code = 0 if check.ok else 1
            if args.json:
                print(json.dumps(asdict(check), indent=2, sort_keys=True), flush=True)
            else:
                _print_text(check)

        if not args.loop:
            return exit_code
        time.sleep(max(float(args.interval_seconds), 1.0))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

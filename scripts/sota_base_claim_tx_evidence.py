#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPOS = Path("/home/mekaneeky/repos")
LOCAL_STATE = REPOS / ".sota-base-local" / "state.json"
TESTNET_ARTIFACTS_DIR = REPOS / ".sota-base-testnet"
BASE_SEPOLIA_CHAIN_ID = 84532
LOCAL_CHAIN_ID = 31337
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
TX_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
GENESIS_CLAIMED_TOPIC = "0x7a6cf42acc3ac4e9793b4308a0eb759568dcad1ceba245f7ac3e476df0d03de1"
EMISSION_CLAIMED_TOPIC = "0x838cb59433a7ac4c503675fd1f66f481f9aa8c69857a24d2689a0bc81cf6819b"
CLAIM_RECORDED_TOPIC = "0x90c6af521ca32363588ed3839025a4b2d3a7e25835d468ac889ccb945155aa65"
SOTA_RELEASED_TOPIC = "0xc0f614bb23dc13bfd21d5b49f156f3da48483508769ca92a92c692df8f6d5cf6"
GENESIS_CLAIM_SELECTOR = "0x6959669d"
EMISSION_CLAIM_SELECTOR = "0x090eb799"
BALANCE_OF_SELECTOR = "0x70a08231"


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


def _check(name: str, ok: bool, success: str, failure: str, *, remediation: str = "") -> Check:
    return Check(name, "green" if ok else "red", success if ok else failure, "" if ok else remediation)


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


def _json_rpc(rpc_url: str, method: str, params: list[Any] | None = None, *, timeout: float) -> Any:
    request = Request(
        rpc_url,
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "sota-base-claim-tx-evidence/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"{method} failed with HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} failed: {exc}") from exc
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload.get("result")


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value), 16 if str(value).startswith("0x") else 10)
    except ValueError:
        return None


def _normalize_address(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if not EVM_ADDRESS_RE.fullmatch(value):
        return value.lower()
    return "0x" + value[2:].lower()


def _is_address(value: str) -> bool:
    return bool(EVM_ADDRESS_RE.fullmatch(value)) and value.lower() != ZERO_ADDRESS


def _is_tx_hash(value: str) -> bool:
    return bool(TX_HASH_RE.fullmatch(value))


def _topic_address(topic: str) -> str:
    text = str(topic).lower()
    if not text.startswith("0x") or len(text) != 66:
        return ""
    return "0x" + text[-40:]


def _hex_data_to_int(data: Any) -> int:
    text = str(data or "0x0")
    return int(text, 16) if text.startswith("0x") else int(text)


def _log_topic(log: dict[str, Any], index: int) -> str:
    topics = list(log.get("topics") or [])
    if index >= len(topics):
        return ""
    return str(topics[index]).lower()


def _log_address(log: dict[str, Any]) -> str:
    return _normalize_address(log.get("address"))


def _logs(receipt: dict[str, Any], *, address: str | None = None, topic0: str | None = None) -> list[dict[str, Any]]:
    normalized_address = _normalize_address(address) if address else None
    normalized_topic = topic0.lower() if topic0 else None
    result: list[dict[str, Any]] = []
    for raw_log in receipt.get("logs") or []:
        if not isinstance(raw_log, dict):
            continue
        if normalized_address and _log_address(raw_log) != normalized_address:
            continue
        if normalized_topic and _log_topic(raw_log, 0) != normalized_topic:
            continue
        result.append(raw_log)
    return result


def _transfer_amount_to(receipt: dict[str, Any], *, token: str, account: str) -> int:
    account = _normalize_address(account)
    total = 0
    for log in _logs(receipt, address=token, topic0=TRANSFER_TOPIC):
        if _topic_address(_log_topic(log, 2)) == account:
            total += _hex_data_to_int(log.get("data"))
    return total


def _claim_amount(receipt: dict[str, Any], *, distributor: str, event_topic: str) -> int:
    for log in _logs(receipt, address=distributor, topic0=event_topic):
        data = str(log.get("data") or "0x")
        word_index = 1 if event_topic == EMISSION_CLAIMED_TOPIC else 0
        start = 2 + (word_index * 64)
        end = start + 64
        if len(data) >= end:
            return int(data[start:end], 16)
    return 0


def _has_claim_event(receipt: dict[str, Any], *, distributor: str, topic: str, account: str) -> bool:
    account = _normalize_address(account)
    for log in _logs(receipt, address=distributor, topic0=topic):
        if _topic_address(_log_topic(log, 2 if topic == GENESIS_CLAIMED_TOPIC else 3)) == account:
            return True
    return False


def _has_claim_recorded(receipt: dict[str, Any], *, distributor: str, account: str) -> bool:
    account = _normalize_address(account)
    for log in _logs(receipt, address=distributor, topic0=CLAIM_RECORDED_TOPIC):
        if _topic_address(_log_topic(log, 3)) == account:
            return True
    return False


def _has_vault_release(receipt: dict[str, Any], *, vault: str, account: str) -> bool:
    if not vault:
        return False
    account = _normalize_address(account)
    for log in _logs(receipt, address=vault, topic0=SOTA_RELEASED_TOPIC):
        if _topic_address(_log_topic(log, 2)) == account:
            return True
    return False


def _balance_call_data(account: str) -> str:
    normalized = _normalize_address(account)
    return BALANCE_OF_SELECTOR + normalized[2:].rjust(64, "0")


def _contract_address(manifest: dict[str, Any], env: dict[str, str], key: str, env_key: str) -> str:
    env_value = env.get(env_key)
    if env_value:
        return _normalize_address(env_value)
    contract = dict(dict(manifest.get("contracts") or {}).get(key) or {})
    return _normalize_address(contract.get("address"))


def _test_wallet_from_funding_report(artifacts_dir: Path) -> str:
    report = _load_json(artifacts_dir / "base-sota-testnet-funding.json")
    for target in report.get("funding_targets") or []:
        if not isinstance(target, dict):
            continue
        if str(target.get("label") or "") == "test_wallet":
            return str(target.get("address") or "")
    return ""


def _test_wallet_from_local_state(state: dict[str, Any]) -> str:
    return str(dict(state.get("accounts") or {}).get("alice_reward") or "")


def _local_config(state: dict[str, Any]) -> dict[str, str]:
    contracts = dict(state.get("contracts") or {})
    urls = dict(state.get("urls") or {})
    return {
        "rpc_url": urls.get("anvil_rpc") or "http://127.0.0.1:8545",
        "expected_chain_id": str(state.get("chain_id") or LOCAL_CHAIN_ID),
        "wallet_address": str(dict(state.get("accounts") or {}).get("alice_reward") or ""),
        "sota_token": str(contracts.get("sota_token") or ""),
        "vault": str(contracts.get("vault") or ""),
        "genesis_distributor": str(contracts.get("genesis_distributor") or ""),
        "emission_distributor": str(contracts.get("emission_distributor") or ""),
    }


def _testnet_config(
    manifest: dict[str, Any],
    env: dict[str, str],
    state: dict[str, Any],
    artifacts_dir: Path,
) -> dict[str, str]:
    chain = dict(manifest.get("chain") or {})
    wallet_address = (
        env.get("SOTA_TEST_WALLET_ADDRESS")
        or _test_wallet_from_funding_report(artifacts_dir)
        or _test_wallet_from_local_state(state)
    )
    return {
        "rpc_url": env.get("SOTA_BASE_RPC_URL") or env.get("NEXT_PUBLIC_SOTA_BASE_RPC_URL") or str(chain.get("public_browser_rpc_url") or "https://sepolia.base.org"),
        "expected_chain_id": str(env.get("SOTA_BASE_CHAIN_ID") or env.get("NEXT_PUBLIC_SOTA_BASE_CHAIN_ID") or chain.get("chain_id") or BASE_SEPOLIA_CHAIN_ID),
        "wallet_address": wallet_address,
        "sota_token": _contract_address(manifest, env, "sota_token", "SOTA_TOKEN_ADDRESS"),
        "vault": _contract_address(manifest, env, "vault", "SOTA_VAULT_ADDRESS"),
        "genesis_distributor": _contract_address(manifest, env, "genesis_distributor", "SOTA_GENESIS_DISTRIBUTOR_ADDRESS"),
        "emission_distributor": _contract_address(manifest, env, "emission_distributor", "SOTA_EMISSION_DISTRIBUTOR_ADDRESS"),
    }


def _effective_config(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    state = _load_json(args.state) if args.state else {}
    manifest = _load_json(args.manifest)
    env = _load_env(args.env_file)
    config = (
        _local_config(state)
        if args.environment == "local"
        else _testnet_config(manifest, env, state, args.artifacts_dir)
    )
    overrides = {
        "rpc_url": args.rpc_url,
        "wallet_address": args.wallet_address,
        "sota_token": args.sota_token_address,
        "vault": args.vault_address,
        "genesis_distributor": args.genesis_distributor_address,
        "emission_distributor": args.emission_distributor_address,
        "expected_chain_id": str(args.chain_id or ""),
    }
    for key, value in overrides.items():
        if value:
            config[key] = value
    for key in ("wallet_address", "sota_token", "vault", "genesis_distributor", "emission_distributor"):
        config[key] = _normalize_address(config.get(key))
    return manifest, env, config


def _config_checks(config: dict[str, str], args: argparse.Namespace) -> list[Check]:
    expected_chain_id = _as_int(config.get("expected_chain_id"))
    checks = [
        _check("rpc_url", bool(config.get("rpc_url")), "RPC URL is configured.", "RPC URL is missing.", remediation="Set --rpc-url or provide the local state/testnet env file."),
        _check("wallet_address", _is_address(config.get("wallet_address", "")), "Tester wallet address is configured.", f"Tester wallet address is missing or invalid: {config.get('wallet_address') or 'missing'}.", remediation="Set --wallet-address or SOTA_TEST_WALLET_ADDRESS."),
        _check("sota_token", _is_address(config.get("sota_token", "")), "SOTA token address is configured.", f"SOTA token address is missing or invalid: {config.get('sota_token') or 'missing'}.", remediation="Provide the SOTA token address from the deployment manifest."),
        _check("genesis_distributor", _is_address(config.get("genesis_distributor", "")), "Genesis distributor address is configured.", f"Genesis distributor address is missing or invalid: {config.get('genesis_distributor') or 'missing'}.", remediation="Provide the GenesisClaimDistributor address from the deployment manifest."),
        _check("emission_distributor", _is_address(config.get("emission_distributor", "")), "Emission distributor address is configured.", f"Emission distributor address is missing or invalid: {config.get('emission_distributor') or 'missing'}.", remediation="Provide the EmissionClaimDistributor address from the deployment manifest."),
        _check("genesis_tx_hash", _is_tx_hash(args.genesis_tx), "Genesis claim transaction hash is present.", f"Genesis claim transaction hash is missing or invalid: {args.genesis_tx or 'missing'}.", remediation="Paste the Base Sepolia genesis claim tx hash from MetaMask or Basescan."),
        _check("emission_tx_hash", _is_tx_hash(args.emission_tx), "Emission claim transaction hash is present.", f"Emission claim transaction hash is missing or invalid: {args.emission_tx or 'missing'}.", remediation="Paste the Base Sepolia emission claim tx hash from MetaMask or Basescan."),
    ]
    if args.environment == "testnet":
        checks.append(
            _check(
                "chain_config",
                expected_chain_id == BASE_SEPOLIA_CHAIN_ID,
                "Evidence verifier is pinned to Base Sepolia chain id 84532.",
                f"Evidence verifier chain id is {expected_chain_id}, expected Base Sepolia 84532.",
                remediation="Use Base Sepolia only for testnet evidence; never use Base mainnet chain id 8453.",
            )
        )
    return checks


def _fetch_tx_pair(rpc_url: str, tx_hash: str, *, timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _json_rpc(rpc_url, "eth_getTransactionReceipt", [tx_hash], timeout=timeout)
    tx = _json_rpc(rpc_url, "eth_getTransactionByHash", [tx_hash], timeout=timeout)
    return dict(receipt or {}), dict(tx or {})


def _verify_tx(
    *,
    label: str,
    tx_hash: str,
    distributor: str,
    event_topic: str,
    selector: str,
    token: str,
    vault: str,
    account: str,
    rpc_url: str,
    expected_chain_id: int,
    timeout: float,
) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    evidence: dict[str, Any] = {"tx_hash": tx_hash}
    if not _is_tx_hash(tx_hash) or not rpc_url:
        return checks, evidence
    try:
        receipt, tx = _fetch_tx_pair(rpc_url, tx_hash, timeout=timeout)
    except Exception as exc:
        return [
            Check(
                f"{label}_receipt",
                "red",
                f"Could not load {label} tx receipt/transaction: {exc}",
                "Check the RPC URL and tx hash.",
            )
        ], evidence
    evidence["block_number"] = _as_int(receipt.get("blockNumber"))
    evidence["transaction_index"] = _as_int(receipt.get("transactionIndex"))
    evidence["gas_used"] = _as_int(receipt.get("gasUsed"))
    evidence["from"] = _normalize_address(tx.get("from"))
    evidence["to"] = _normalize_address(tx.get("to"))
    evidence["status"] = _as_int(receipt.get("status"))
    evidence["transfer_to_account_raw"] = str(_transfer_amount_to(receipt, token=token, account=account))
    evidence["claim_amount_raw"] = str(_claim_amount(receipt, distributor=distributor, event_topic=event_topic))

    checks.append(
        _check(
            f"{label}_receipt_status",
            _as_int(receipt.get("status")) == 1,
            f"{label} transaction succeeded.",
            f"{label} transaction status is {receipt.get('status')!r}, expected 0x1.",
            remediation="Use the successful claim transaction hash from the tester wallet.",
        )
    )
    checks.append(
        _check(
            f"{label}_chain_id",
            _as_int(tx.get("chainId")) == expected_chain_id,
            f"{label} transaction is on chain id {expected_chain_id}.",
            f"{label} transaction chain id is {tx.get('chainId')!r}, expected {expected_chain_id}.",
            remediation="Use the expected local/Base Sepolia claim transaction hash.",
        )
    )
    checks.append(
        _check(
            f"{label}_from_wallet",
            _normalize_address(tx.get("from")) == _normalize_address(account),
            f"{label} transaction was sent by the tester wallet.",
            f"{label} sender is {_normalize_address(tx.get('from')) or 'missing'}, expected {account}.",
            remediation="Submit the claim from the configured tester wallet.",
        )
    )
    checks.append(
        _check(
            f"{label}_to_distributor",
            _normalize_address(tx.get("to")) == _normalize_address(distributor),
            f"{label} transaction targets the expected claim distributor.",
            f"{label} target is {_normalize_address(tx.get('to')) or 'missing'}, expected {distributor}.",
            remediation="Use the claim transaction built by the configured claims API/UI.",
        )
    )
    input_data = str(tx.get("input") or tx.get("data") or "").lower()
    checks.append(
        _check(
            f"{label}_function_selector",
            input_data.startswith(selector),
            f"{label} transaction uses the expected claim function selector.",
            f"{label} calldata selector is {input_data[:10] or 'missing'}, expected {selector}.",
            remediation="Use the matching genesis/emission claim transaction.",
        )
    )
    checks.append(
        _check(
            f"{label}_claim_event",
            _has_claim_event(receipt, distributor=distributor, topic=event_topic, account=account),
            f"{label} distributor emitted the expected claim event for the tester wallet.",
            f"{label} distributor did not emit the expected claim event for {account}.",
            remediation="Check the tx hash, distributor address, and tester wallet.",
        )
    )
    checks.append(
        _check(
            f"{label}_claim_recorded",
            _has_claim_recorded(receipt, distributor=distributor, account=account),
            f"{label} distributor emitted ClaimRecorded for the tester wallet.",
            f"{label} distributor did not emit ClaimRecorded for {account}.",
            remediation="Check the tx hash, distributor address, and claim proof.",
        )
    )
    transfer_amount = _transfer_amount_to(receipt, token=token, account=account)
    checks.append(
        _check(
            f"{label}_sota_transfer",
            transfer_amount > 0,
            f"{label} transferred {transfer_amount} raw SOTA units to the tester wallet.",
            f"{label} did not include a positive SOTA Transfer event to {account}.",
            remediation="Check token address, vault releaser config, and claim receipt logs.",
        )
    )
    if vault:
        checks.append(
            _check(
                f"{label}_vault_release",
                _has_vault_release(receipt, vault=vault, account=account),
                f"{label} vault release event names the tester wallet.",
                f"{label} did not include a SOTAReleased event to {account}.",
                remediation="Check the vault address and receipt logs.",
            )
        )
    return checks, evidence


def _token_balance(rpc_url: str, token: str, account: str, *, timeout: float) -> int | None:
    if not (rpc_url and _is_address(token) and _is_address(account)):
        return None
    raw = _json_rpc(
        rpc_url,
        "eth_call",
        [{"to": token, "data": _balance_call_data(account)}, "latest"],
        timeout=timeout,
    )
    return _as_int(raw)


def run_evidence(args: argparse.Namespace) -> dict[str, Any]:
    _manifest, _env, config = _effective_config(args)
    expected_chain_id = int(config.get("expected_chain_id") or (LOCAL_CHAIN_ID if args.environment == "local" else BASE_SEPOLIA_CHAIN_ID))
    checks: list[Check] = _config_checks(config, args)
    rpc_url = config.get("rpc_url", "")
    if rpc_url:
        try:
            raw_chain_id = _json_rpc(rpc_url, "eth_chainId", timeout=args.timeout)
            chain_id = _as_int(raw_chain_id)
        except Exception as exc:
            checks.append(Check("rpc_chain_id", "red", f"Could not read chain id: {exc}", "Fix the RPC URL."))
            chain_id = None
        checks.append(
            _check(
                "rpc_chain_id",
                chain_id == expected_chain_id,
                f"RPC returned expected chain id {expected_chain_id}.",
                f"RPC returned chain id {chain_id}, expected {expected_chain_id}.",
                remediation="Use the matching local Anvil or Base Sepolia RPC.",
            )
        )
    tx_evidence: dict[str, Any] = {}
    if not any(check.status == "red" for check in checks if check.name in {"rpc_url", "wallet_address", "sota_token", "genesis_distributor", "emission_distributor"}):
        genesis_checks, genesis_evidence = _verify_tx(
            label="genesis",
            tx_hash=args.genesis_tx,
            distributor=config["genesis_distributor"],
            event_topic=GENESIS_CLAIMED_TOPIC,
            selector=GENESIS_CLAIM_SELECTOR,
            token=config["sota_token"],
            vault=config.get("vault", ""),
            account=config["wallet_address"],
            rpc_url=rpc_url,
            expected_chain_id=expected_chain_id,
            timeout=args.timeout,
        )
        emission_checks, emission_evidence = _verify_tx(
            label="emission",
            tx_hash=args.emission_tx,
            distributor=config["emission_distributor"],
            event_topic=EMISSION_CLAIMED_TOPIC,
            selector=EMISSION_CLAIM_SELECTOR,
            token=config["sota_token"],
            vault=config.get("vault", ""),
            account=config["wallet_address"],
            rpc_url=rpc_url,
            expected_chain_id=expected_chain_id,
            timeout=args.timeout,
        )
        checks.extend(genesis_checks)
        checks.extend(emission_checks)
        tx_evidence = {"genesis": genesis_evidence, "emission": emission_evidence}
        try:
            balance = _token_balance(rpc_url, config["sota_token"], config["wallet_address"], timeout=args.timeout)
        except Exception as exc:
            checks.append(Check("sota_balance", "red", f"Could not read SOTA balance: {exc}", "Check RPC/token address."))
            balance = None
        checks.append(
            _check(
                "sota_balance",
                balance is not None and balance > 0,
                f"Tester wallet has {balance} raw SOTA units after claims.",
                f"Tester wallet SOTA balance is {balance}, expected a positive balance.",
                remediation="Submit the claims from the tester wallet and verify token address.",
            )
        )
    status = _worst(checks)
    return {
        "schema": "sota-base-claim-tx-evidence/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": args.environment,
        "ok": status == "green",
        "status": status,
        "read_only": True,
        "does_not": ["deploy", "sign", "broadcast_transactions", "touch_production_bittensor"],
        "message": (
            "Claim transaction evidence verifies both genesis and emission SOTA claims."
            if status == "green"
            else "Claim transaction evidence is incomplete or failed verification."
        ),
        "config": {
            key: value
            for key, value in config.items()
            if key in {"rpc_url", "expected_chain_id", "wallet_address", "sota_token", "vault", "genesis_distributor", "emission_distributor"}
        },
        "transactions": tx_evidence,
        "checks": [check.as_dict() for check in checks],
        "summary": {
            "green": sum(1 for check in checks if check.status == "green"),
            "yellow": sum(1 for check in checks if check.status == "yellow"),
            "red": sum(1 for check in checks if check.status == "red"),
        },
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"SOTA claim tx evidence: {report['status'].upper()}")
    print(report["message"])
    print(f"Summary: {report['summary']['green']} green, {report['summary']['yellow']} yellow, {report['summary']['red']} red")
    for check in report["checks"]:
        print(f"- [{check['status']}] {check['name']}: {check['detail']}")
        if check.get("remediation"):
            print(f"  next: {check['remediation']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only evidence verifier for Base SOTA genesis/emission claim transactions.")
    parser.add_argument("--environment", choices=("local", "testnet"), default="testnet")
    parser.add_argument("--state", type=Path, default=LOCAL_STATE)
    parser.add_argument("--artifacts-dir", type=Path, default=TESTNET_ARTIFACTS_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--rpc-url", default="")
    parser.add_argument("--chain-id", type=int, default=0)
    parser.add_argument("--wallet-address", default="")
    parser.add_argument("--sota-token-address", default="")
    parser.add_argument("--vault-address", default="")
    parser.add_argument("--genesis-distributor-address", default="")
    parser.add_argument("--emission-distributor-address", default="")
    parser.add_argument("--genesis-tx", default="")
    parser.add_argument("--emission-tx", default="")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report-out", type=Path, default=Path(""))
    parser.add_argument("--allow-blocked", action="store_true", help="Exit 0 even when evidence checks are red.")
    args = parser.parse_args(argv)
    args.manifest = args.manifest or args.artifacts_dir / "base-sepolia-deployment-manifest.json"
    args.env_file = args.env_file or args.artifacts_dir / "base-sota.env.testnet"
    report = run_evidence(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text, encoding="utf-8")
    if args.json:
        print(text, end="")
    else:
        _print_text(report)
    return 0 if report["ok"] or args.allow_blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())

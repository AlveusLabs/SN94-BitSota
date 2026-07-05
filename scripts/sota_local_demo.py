#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from eth_account import Account
from eth_abi import encode
from eth_utils import keccak
from substrateinterface import Keypair
from web3 import Web3

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sota_emission_policy import frontier_capacitor_reward_policy, sota_epoch_budget_units


REPOS = Path("/home/mekaneeky/repos")
DOCS_REPO = Path(__file__).resolve().parents[1]
POOL_REPO = REPOS / "Pool"
CONTRACTS_REPO = POOL_REPO / "contracts" / "sota-base"
COMMUNITY_REPO = REPOS / "94-agent-community"
AUTORESEARCH_REPO = REPOS / "autoresearch-bittensor"
WEBSITE_REPO = REPOS / "bitsota_website"
RUN_DIR = REPOS / ".sota-base-local"
TESTNET_RUN_DIR = REPOS / ".sota-base-testnet"
LOG_DIR = RUN_DIR / "logs"
HANDOFF_DIR = RUN_DIR / "handoff"
STATE_PATH = RUN_DIR / "state.json"
PIDS_PATH = RUN_DIR / "pids.json"

ANVIL_HOST = "0.0.0.0"
ANVIL_RPC_LOCAL = "http://127.0.0.1:8545"
CHAIN_ID = 31337
ONE_SOTA = 10**18
SERVICE_PORTS = {
    "anvil": 8545,
    "autoresearch": 8000,
    "indexer": 8010,
    "website": 3000,
    "docs": 9002,
    "handoff": 9003,
}
PUBLIC_SERVICE_PORTS = {
    "claims_ui": 3000,
    "autoresearch_dashboard": 8000,
    "indexer_health": 8010,
    "docs": 9002,
    "anvil_rpc": 8545,
    "handoff": 9003,
}
ADMIN_TOKEN = "local-admin"
LANE_ID = "base:sota-local"
DEMO_MNEMONIC = "test test test test test test test test test test test junk"
ANVIL_PRIVATE_KEYS = {
    "owner": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    "publisher": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    "alice_reward": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
    "miner": "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",
}
DEMO_VALIDATOR_URIS = ("//Bob", "//Charlie", "//Dave")
DEMO_SELF_VALIDATION_COMMITTEE_SIZE = len(DEMO_VALIDATOR_URIS)
DEMO_SWARM_MINER_COUNT = 5


def _print(message: str) -> None:
    print(message, flush=True)


def _python(repo: Path) -> str:
    venv_python = repo / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else sys.executable)


def _tailscale_ip() -> str | None:
    binary = shutil.which("tailscale")
    if not binary:
        return None
    try:
        output = subprocess.check_output([binary, "ip", "-4"], text=True, stderr=subprocess.DEVNULL, timeout=2)
    except Exception:
        return None
    for line in output.splitlines():
        text = line.strip()
        if text:
            return text
    return None


def _tailscale_status() -> dict[str, Any]:
    binary = shutil.which("tailscale")
    if not binary:
        return {}
    try:
        output = subprocess.check_output([binary, "status", "--json"], text=True, stderr=subprocess.DEVNULL, timeout=3)
    except Exception:
        return {}
    try:
        payload = json.loads(output or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tailscale_dns_name(status: dict[str, Any] | None = None) -> str | None:
    status = status or _tailscale_status()
    self_info = dict(status.get("Self") or {})
    dns_name = str(self_info.get("DNSName") or "").strip().rstrip(".")
    if dns_name:
        return dns_name
    host_name = str(self_info.get("HostName") or "").strip()
    suffix = str(status.get("MagicDNSSuffix") or "").strip().strip(".")
    if host_name and suffix:
        return f"{host_name}.{suffix}"
    return None


def _primary_host() -> str:
    return _tailscale_ip() or "127.0.0.1"


def _localhost_url_set() -> dict[str, str]:
    return _public_url_set(scheme="http", host="127.0.0.1")


def _public_url_set(*, scheme: str, host: str) -> dict[str, str]:
    def base_url(service: str) -> str:
        return f"{scheme}://{host}:{PUBLIC_SERVICE_PORTS[service]}"

    return {
        "claims_ui": f"{base_url('claims_ui')}/claims",
        "autoresearch_dashboard": f"{base_url('autoresearch_dashboard')}/dashboard",
        "indexer_health": f"{base_url('indexer_health')}/health",
        "docs": f"{base_url('docs')}/base/",
        "anvil_rpc": base_url("anvil_rpc"),
        "handoff": f"{base_url('handoff')}/",
    }


def _run_tailscale_serve_https(port: int) -> tuple[bool, str]:
    binary = shutil.which("tailscale")
    if not binary:
        return False, "tailscale binary is not installed"
    try:
        result = subprocess.run(
            [
                binary,
                "serve",
                "--bg",
                "--yes",
                "--https",
                str(port),
                f"http://127.0.0.1:{port}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        detail = (stderr.strip() or stdout.strip() or f"tailscale serve timed out while configuring HTTPS port {port}")
        return False, detail
    if result.returncode == 0:
        return True, ""
    return False, (result.stderr.strip() or result.stdout.strip() or f"tailscale serve exited {result.returncode}")


def _clear_tailscale_serve_https_ports(ports: list[int]) -> None:
    binary = shutil.which("tailscale")
    if not binary:
        return
    for port in sorted(set(int(port) for port in ports)):
        subprocess.run(
            [binary, "serve", "--yes", "--https", str(port), "off"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )


def _plan_public_share(
    share_mode: str,
    *,
    require_remote_wallet: bool,
    warning_override: str = "",
) -> tuple[dict[str, str], dict[str, Any]]:
    if share_mode == "localhost" or not require_remote_wallet:
        return _localhost_url_set(), {
            "mode": "localhost",
            "status": "green",
            "host": "127.0.0.1",
            "wallet_rpc_browser_safe": True,
            "warning": warning_override
            or (
                "This wallet-safe local URL only works on the computer running the demo. "
                "Use --share-mode tailscale-https after enabling Tailscale Serve for remote MetaMask testing."
            ),
        }

    host = _primary_host()
    http_urls = _public_url_set(scheme="http", host=host)
    if share_mode == "http":
        return http_urls, {
            "mode": "http",
            "status": "green" if host == "127.0.0.1" else "yellow",
            "host": host,
            "wallet_rpc_browser_safe": host == "127.0.0.1",
            "warning": (
                ""
                if host == "127.0.0.1"
                else warning_override
                or "HTTP Tailscale-IP RPC can be rejected by wallet extensions on another computer; use --share-mode tailscale-https for remote MetaMask testing."
            ),
        }

    status = _tailscale_status()
    dns_name = _tailscale_dns_name(status)
    if not dns_name:
        if share_mode == "tailscale-https":
            raise RuntimeError("Tailscale MagicDNS name is unavailable; cannot publish the local demo over Tailscale Serve HTTPS.")
        return _localhost_url_set(), {
            "mode": "localhost",
            "status": "green",
            "host": "127.0.0.1",
            "wallet_rpc_browser_safe": True,
            "warning": (
                "Tailscale MagicDNS is unavailable, so the launcher fell back to wallet-safe localhost URLs. "
                "These URLs only work on the computer running the demo."
            ),
        }

    return _public_url_set(scheme="https", host=dns_name), {
        "mode": "tailscale-https",
        "status": "pending",
        "host": dns_name,
        "tailscale_dns_name": dns_name,
        "configured_https_ports": [],
        "wallet_rpc_browser_safe": True,
        "warning": "",
    }


def _activate_public_share(sharing: dict[str, Any]) -> dict[str, Any]:
    if sharing.get("mode") != "tailscale-https":
        return sharing
    configured_ports: list[int] = []
    errors: list[str] = []
    for port in sorted(set(PUBLIC_SERVICE_PORTS.values())):
        ok, detail = _run_tailscale_serve_https(port)
        if ok:
            configured_ports.append(port)
        else:
            errors.append(f"{port}: {detail}")
            break
    if errors:
        _clear_tailscale_serve_https_ports(configured_ports)
        raise RuntimeError("Tailscale Serve HTTPS setup failed: " + "; ".join(errors))
    return {
        **sharing,
        "status": "green",
        "configured_https_ports": configured_ports,
        "warning": "",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return "0x" + bytes(value).hex()
    hex_method = getattr(value, "hex", None)
    if callable(hex_method):
        text = hex_method()
        return text if str(text).startswith("0x") else f"0x{text}"
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> Any:
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request_headers.setdefault("Accept", "application/json")
    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {body}") from exc
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def _wait_http(url: str, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except (URLError, TimeoutError, ConnectionError):
            time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for {url}")


def _wait_rpc(timeout_seconds: float = 60.0) -> Web3:
    w3 = Web3(Web3.HTTPProvider(ANVIL_RPC_LOCAL))
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if w3.is_connected():
                _ = w3.eth.block_number
                return w3
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for {ANVIL_RPC_LOCAL}")


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def _port_owner(port: int) -> str | None:
    try:
        result = subprocess.run(
            ["ss", "-ltnp", "sport", "=", f":{port}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[-1] if len(lines) > 1 else None


def _port_in_use_error(port: int) -> RuntimeError:
    owner = _port_owner(port)
    suffix = f"; current listener: {owner}" if owner else ""
    return RuntimeError(f"port {port} is already in use{suffix}")


def _wait_for_port_closed(port: int, *, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _is_port_open("127.0.0.1", port):
            return
        time.sleep(0.2)
    owner = _port_owner(port)
    suffix = f"; current listener: {owner}" if owner else ""
    raise RuntimeError(f"port {port} is still in use after shutdown{suffix}")


def stop_stack() -> None:
    state = _load_json(STATE_PATH)
    sharing = dict(state.get("sharing") or {})
    if sharing.get("mode") == "tailscale-https":
        _clear_tailscale_serve_https_ports([int(port) for port in sharing.get("configured_https_ports") or []])
    pids = _load_json(PIDS_PATH)
    for name, pid in list(pids.items()):
        try:
            os.killpg(int(pid), signal.SIGTERM)
            _print(f"stopped {name} pid={pid}")
        except ProcessLookupError:
            pass
        except Exception as exc:
            _print(f"could not stop {name} pid={pid}: {exc}")
    for name in pids:
        port = SERVICE_PORTS.get(name)
        if port is not None:
            _wait_for_port_closed(port)
    PIDS_PATH.unlink(missing_ok=True)


def _reset_runtime_state() -> None:
    for path in (
        RUN_DIR / "indexer.sqlite3",
        RUN_DIR / "indexer.sqlite3-shm",
        RUN_DIR / "indexer.sqlite3-wal",
        RUN_DIR / "autoresearch.sqlite3",
        RUN_DIR / "autoresearch.sqlite3-shm",
        RUN_DIR / "autoresearch.sqlite3-wal",
        STATE_PATH,
    ):
        path.unlink(missing_ok=True)
    if HANDOFF_DIR.exists():
        shutil.rmtree(HANDOFF_DIR)
    task_repo = RUN_DIR / "task-repo"
    if task_repo.exists():
        shutil.rmtree(task_repo)


def _start_process(name: str, args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    log = log_path.open("ab")
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        env=process_env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pids = _load_json(PIDS_PATH)
    pids[name] = process.pid
    _write_json(PIDS_PATH, pids)
    _print(f"started {name} pid={process.pid} log={log_path}")
    return process


def _artifact(name: str) -> dict[str, Any]:
    path = CONTRACTS_REPO / "out" / f"{name}.sol" / f"{name}.json"
    if not path.exists():
        subprocess.check_call([str(Path.home() / ".foundry" / "bin" / "forge"), "build"], cwd=CONTRACTS_REPO)
    return json.loads(path.read_text(encoding="utf-8"))


def _bytecode(artifact: dict[str, Any]) -> str:
    value = artifact["bytecode"]
    return value["object"] if isinstance(value, dict) else value


def _deploy(w3: Web3, name: str, *args: Any, sender: str) -> Any:
    artifact = _artifact(name)
    contract = w3.eth.contract(abi=artifact["abi"], bytecode=_bytecode(artifact))
    tx_hash = contract.constructor(*args).transact({"from": sender})
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if receipt.contractAddress is None:
        raise RuntimeError(f"{name} deployment did not return a contract address")
    return w3.eth.contract(address=receipt.contractAddress, abi=artifact["abi"])


def _bytes32_text(label: str) -> bytes:
    return keccak(text=label)


def _hex32(value: bytes | str) -> str:
    if isinstance(value, str):
        return value if value.startswith("0x") else f"0x{value}"
    return "0x" + value.hex()


def _demo_validator_keypairs() -> list[tuple[str, Keypair]]:
    return [(uri.removeprefix("//"), Keypair.create_from_uri(uri)) for uri in DEMO_VALIDATOR_URIS]


def _deterministic_private_key(label: str) -> str:
    value = keccak(text=label)
    if int.from_bytes(value, "big") == 0:
        value = b"\x01".rjust(32, b"\x00")
    return "0x" + value.hex()


def _demo_swarm_miner(index: int) -> dict[str, Any]:
    if index < 1:
        raise ValueError("miner index must start at 1")
    hotkey_uri = f"//SotaLocalMiner{index}"
    hotkey = Keypair.create_from_uri(hotkey_uri)
    miner_private_key = _deterministic_private_key(f"sota-local-swarm:miner:{index}")
    reward_private_key = _deterministic_private_key(f"sota-local-swarm:reward:{index}")
    return {
        "index": index,
        "name": f"miner-{index}",
        "hotkey_uri": hotkey_uri,
        "hotkey": hotkey.ss58_address,
        "miner_private_key": miner_private_key,
        "miner_address": Account.from_key(miner_private_key).address,
        "reward_private_key": reward_private_key,
        "reward_address": Account.from_key(reward_private_key).address,
    }


def _root_pair(left: str, right: str) -> str:
    left_b = bytes.fromhex(left.removeprefix("0x"))
    right_b = bytes.fromhex(right.removeprefix("0x"))
    first, second = sorted((left_b, right_b))
    return "0x" + keccak(first + second).hex()


def _publish_root(
    w3: Web3,
    registry: Any,
    *,
    publisher: str,
    kind: int,
    merkle_root: str,
    budget_cap: int,
    policy_hash: str,
    attestation_hash: str,
    nonce: str,
) -> str:
    args = (
        int(kind),
        bytes.fromhex(merkle_root.removeprefix("0x")),
        int(budget_cap),
        bytes.fromhex(policy_hash.removeprefix("0x")),
        bytes.fromhex(attestation_hash.removeprefix("0x")),
        bytes.fromhex(nonce.removeprefix("0x")),
    )
    root_id = registry.functions.publishRoot(*args).call({"from": publisher})
    tx_hash = registry.functions.publishRoot(*args).transact({"from": publisher})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    return _hex32(root_id)


def deploy_contracts() -> dict[str, Any]:
    w3 = _wait_rpc()
    accounts = w3.eth.accounts
    owner, publisher, alice_reward = accounts[0], accounts[1], accounts[2]
    token = _deploy(w3, "SOTAToken", owner, owner, owner, sender=owner)
    vault = _deploy(w3, "SOTAVault", token.address, owner, sender=owner)
    root_registry = _deploy(w3, "SOTARootRegistry", owner, sender=owner)
    lane_registry = _deploy(w3, "SOTALaneRegistry", owner, sender=owner)
    genesis = _deploy(w3, "GenesisClaimDistributor", root_registry.address, vault.address, owner, sender=owner)
    emission = _deploy(
        w3,
        "EmissionClaimDistributor",
        root_registry.address,
        lane_registry.address,
        vault.address,
        owner,
        sender=owner,
    )
    root_registry.functions.setRootPublisher(publisher, True).transact({"from": owner})
    vault.functions.setReleaser(genesis.address, True).transact({"from": owner})
    vault.functions.setReleaser(emission.address, True).transact({"from": owner})
    token.functions.mintSupply(vault.address, 1_000_000 * ONE_SOTA).transact({"from": owner})
    return {
        "chain_id": w3.eth.chain_id,
        "accounts": {
            "owner": owner,
            "publisher": publisher,
            "alice_reward": alice_reward,
            "miner": Account.from_key(ANVIL_PRIVATE_KEYS["miner"]).address,
        },
        "contracts": {
            "sota_token": token.address,
            "vault": vault.address,
            "root_registry": root_registry.address,
            "lane_registry": lane_registry.address,
            "genesis_distributor": genesis.address,
            "emission_distributor": emission.address,
        },
    }


def _contract(w3: Web3, name: str, address: str) -> Any:
    return w3.eth.contract(address=Web3.to_checksum_address(address), abi=_artifact(name)["abi"])


def seed_genesis_onchain_and_indexer(state: dict[str, Any]) -> dict[str, Any]:
    w3 = _wait_rpc()
    contracts = state["contracts"]
    accounts = state["accounts"]
    root_registry = _contract(w3, "SOTARootRegistry", contracts["root_registry"])
    genesis = _contract(w3, "GenesisClaimDistributor", contracts["genesis_distributor"])
    old_coldkey = Keypair.create_from_uri("//Alice").ss58_address
    reward_address = Web3.to_checksum_address(accounts["alice_reward"])
    tao_credit = ONE_SOTA
    alpha_credit = ONE_SOTA // 2
    amount = tao_credit + alpha_credit
    allocation_hash = _hex32(
        keccak(
            encode(
                ["string", "string", "address", "uint256", "uint256"],
                ["SOTA_LOCAL_GENESIS", old_coldkey, reward_address, tao_credit, alpha_credit],
            )
        )
    )
    leaf = _hex32(genesis.functions.leafFor(reward_address, amount, bytes.fromhex(allocation_hash[2:])).call())
    policy_hash = _hex32(_bytes32_text("local-genesis-policy-v1"))
    attestation_hash = _hex32(_bytes32_text("local-genesis-attestation-v1"))
    root_id = _publish_root(
        w3,
        root_registry,
        publisher=accounts["publisher"],
        kind=1,
        merkle_root=leaf,
        budget_cap=amount,
        policy_hash=policy_hash,
        attestation_hash=attestation_hash,
        nonce=_hex32(_bytes32_text("local-genesis-root-v1")),
    )
    _request_json(
        "POST",
        "http://127.0.0.1:8010/api/v1/base/index/artifact",
        {
            "subnet": {
                "id": "genesis",
                "title": "SOTA genesis fork claim",
                "owner": accounts["owner"],
                "budget": amount,
                "metadata_uri": "local://sota/genesis",
                "token": "SOTA",
            },
            "root": {
                "root_id": root_id,
                "subnet_id": "genesis",
                "epoch": 0,
                "root": leaf,
                "total_amount": amount,
                "budget": amount,
                "status": "finalized",
                "validation_status": "accepted",
            },
            "allocations": [
                {
                    "kind": "genesis",
                    "index": 0,
                    "account": reward_address,
                    "amount": amount,
                    "allocation_hash": allocation_hash,
                    "old_coldkey": old_coldkey,
                    "reward_address": reward_address,
                    "tao_credit": tao_credit,
                    "alpha_synthetic_credit": alpha_credit,
                    "leaf": leaf,
                    "proof": [],
                }
            ],
        },
    )
    state["genesis"] = {
        "old_coldkey": old_coldkey,
        "reward_address": reward_address,
        "tao_credit": tao_credit,
        "alpha_synthetic_credit": alpha_credit,
        "amount": amount,
        "allocation_hash": allocation_hash,
        "leaf": leaf,
        "root_id": root_id,
    }
    _write_json(STATE_PATH, state)
    return state


def _signed_headers(keypair: Keypair, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, str]:
    sys.path.insert(0, str(AUTORESEARCH_REPO / "src"))
    from autoresearch_bittensor.auth.hotkey import sign_request

    headers = sign_request(keypair=keypair, method=method, path=path, body=body).as_headers()
    headers["Content-Type"] = "application/json"
    return headers


def _attach_evm_authorization(
    body: dict[str, Any],
    *,
    claim_id: str,
    task_id: str,
    miner_private_key: str,
    reward_private_key: str,
) -> dict[str, Any]:
    sys.path.insert(0, str(AUTORESEARCH_REPO / "src"))
    from autoresearch_bittensor.auth.evm import (
        build_reward_delegation_payload,
        build_submission_authorization_payload,
        build_submission_content_hash,
        sign_payload,
    )

    miner_address = Account.from_key(miner_private_key).address
    reward_address = Account.from_key(reward_private_key).address
    content_hash = build_submission_content_hash(
        claim_id=claim_id,
        base_ref=str(body["base_ref"]),
        patch=str(body.get("patch") or ""),
        summary=str(body.get("summary") or ""),
        proposed_idea=body.get("proposed_idea"),
        implemented_submission_id=body.get("implemented_submission_id"),
        artifact_uri=body.get("artifact_uri"),
        artifact_sha256=body.get("artifact_sha256"),
        artifact_size_bytes=body.get("artifact_size_bytes"),
        claimed_metrics=dict(body.get("claimed_metrics") or {}),
    )
    nonce = f"{LANE_ID}:{claim_id}"
    body.update(
        {
            "evm_miner_address": miner_address,
            "reward_address": reward_address,
            "nonce": nonce,
            "competition_id": task_id,
            "subnet_id": LANE_ID,
            "artifact_hash": content_hash,
        }
    )
    authorization_payload = build_submission_authorization_payload(
        miner_address=miner_address,
        reward_address=reward_address,
        nonce=nonce,
        competition_id=task_id,
        subnet_id=LANE_ID,
        claim_id=claim_id,
        artifact_hash=content_hash,
        content_hash=content_hash,
    )
    body["signature"] = sign_payload(private_key=miner_private_key, payload=authorization_payload)
    delegation_payload = build_reward_delegation_payload(
        miner_address=miner_address,
        reward_address=reward_address,
        nonce=nonce,
        competition_id=task_id,
        subnet_id=LANE_ID,
    )
    body["reward_signature"] = sign_payload(private_key=reward_private_key, payload=delegation_payload)
    return body


def _create_local_task_repo() -> Path:
    repo_dir = RUN_DIR / "task-repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True)
    (repo_dir / "train.py").write_text(
        "score = 0.90\nprint({'heldout_ppl': score})\n",
        encoding="utf-8",
    )
    (repo_dir / "README.md").write_text(
        "# Local SOTA binary frontier\n\nOffline demo task shaped like the binary frontier competition.\n",
        encoding="utf-8",
    )
    subprocess.check_call(["git", "init"], cwd=repo_dir, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "sota-local@example.invalid"], cwd=repo_dir)
    subprocess.check_call(["git", "config", "user.name", "SOTA Local Demo"], cwd=repo_dir)
    subprocess.check_call(["git", "add", "."], cwd=repo_dir, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "commit", "-m", "seed local task"], cwd=repo_dir, stdout=subprocess.DEVNULL)
    return repo_dir


def seed_autoresearch_and_emission(state: dict[str, Any]) -> dict[str, Any]:
    alice = Keypair.create_from_uri("//Alice")
    validators = _demo_validator_keypairs()
    validator_hotkeys = [keypair.ss58_address for _, keypair in validators]
    repo_dir = _create_local_task_repo()
    task_body = {
        "slug": "sota-local-binary-frontier",
        "title": "SOTA Local Binary Frontier",
        "brief": "Local Base fork demo task: improve a tiny binary-style frontier score and pass self-validation.",
        "repository": str(repo_dir),
        "base_ref": "HEAD",
        "setup_command": None,
        "benchmark_command": "python3 train.py",
        "allowed_patch_paths": ["train.py"],
        "metric_name": "heldout_ppl",
        "metric_direction": "minimize",
        "competition_mode": "self_validation",
        "min_peer_evaluations": DEMO_SELF_VALIDATION_COMMITTEE_SIZE,
        "self_validation_policy": {
            "committee_size": DEMO_SELF_VALIDATION_COMMITTEE_SIZE,
            "committee_hotkeys": validator_hotkeys,
            "approval_threshold": 0.5,
            "min_effective_committee_size": float(DEMO_SELF_VALIDATION_COMMITTEE_SIZE),
            "max_approval_concentration": 1.0,
            "new_identity_weight": 1.0,
            "reputation_gain": 0.0,
            "max_reputation_weight": 1.0,
            "slash_tolerance": 0.05,
            "min_improvement": 0.0,
            "sortition_seed": "sota-local-demo",
        },
        "time_budget_seconds": 900,
    }
    task = _request_json(
        "POST",
        "http://127.0.0.1:8000/api/v1/tasks",
        task_body,
        headers={"X-Admin-Token": ADMIN_TOKEN},
    )
    subnet_body = {
        "id": LANE_ID,
        "title": "SOTA local frontier lane",
        "task_slugs": [task["slug"]],
        "budget_units_per_epoch": sota_epoch_budget_units(),
        "reward_policy": frontier_capacitor_reward_policy(),
        "active": True,
        "base_registry_chain_id": CHAIN_ID,
        "base_registry_address": state["contracts"]["lane_registry"],
        "base_registry_subnet_key": "sota-foundation/local-binary-frontier",
        "metadata": {"demo": True, "fork": "sota-base"},
    }
    subnet = _request_json(
        "POST",
        "http://127.0.0.1:8000/api/v1/sota/subnets",
        subnet_body,
        headers={"X-Admin-Token": ADMIN_TOKEN},
    )
    claim_body = {"claim_description": "local SOTA binary frontier mining claim"}
    claim_path = f"/api/v1/tasks/{task['id']}/claim"
    claim = _request_json(
        "POST",
        f"http://127.0.0.1:8000{claim_path}",
        claim_body,
        headers=_signed_headers(alice, "POST", claim_path, claim_body),
    )
    submission_body = {
        "claim_id": claim["id"],
        "base_ref": "HEAD",
        "patch": "diff --git a/train.py b/train.py\n--- a/train.py\n+++ b/train.py\n@@\n-score = 0.90\n+score = 0.82\n",
        "summary": "Lower local heldout PPL with a deterministic binary-frontier patch.",
        "claimed_metrics": {"heldout_ppl": 0.82},
    }
    _attach_evm_authorization(
        submission_body,
        claim_id=claim["id"],
        task_id=task["id"],
        miner_private_key=ANVIL_PRIVATE_KEYS["miner"],
        reward_private_key=ANVIL_PRIVATE_KEYS["alice_reward"],
    )
    submission = _request_json(
        "POST",
        "http://127.0.0.1:8000/api/v1/submissions",
        submission_body,
        headers=_signed_headers(alice, "POST", "/api/v1/submissions", submission_body),
    )
    evaluation_body = {
        "status": "accepted",
        "observed_metrics": {"heldout_ppl": 0.82},
        "notes": "local committee accepts deterministic frontier improvement",
    }
    evaluation_path = f"/api/v1/submissions/{submission['id']}/peer-evaluate"
    evaluations = []
    for name, validator in validators:
        body = {
            **evaluation_body,
            "notes": f"{name} accepts deterministic frontier improvement in the local multi-user committee",
        }
        evaluations.append(
            _request_json(
                "POST",
                f"http://127.0.0.1:8000{evaluation_path}",
                body,
                headers=_signed_headers(validator, "POST", evaluation_path, body),
            )
        )
    consensus = _request_json(
        "GET",
        f"http://127.0.0.1:8000/api/v1/submissions/{submission['id']}/peer-consensus",
    )
    if int(consensus.get("committee_count") or 0) < DEMO_SELF_VALIDATION_COMMITTEE_SIZE:
        raise RuntimeError(f"local self-validation committee too small: {consensus}")
    if int(consensus.get("accepted_count") or 0) < DEMO_SELF_VALIDATION_COMMITTEE_SIZE:
        raise RuntimeError(f"local self-validation did not collect all accepted peer evaluations: {consensus}")
    root = _request_json(
        "POST",
        f"http://127.0.0.1:8000/api/v1/sota/subnets/{LANE_ID}/epochs/1/root",
        {"include_proofs": True},
        headers={"X-Admin-Token": ADMIN_TOKEN},
    )
    evidence = _request_json("GET", f"http://127.0.0.1:8000/api/v1/sota/subnets/{LANE_ID}/epochs/1/evidence")
    state["autoresearch"] = {
        "task": task,
        "subnet": subnet,
        "claim": claim,
        "submission": submission,
        "evaluations": evaluations,
        "consensus": consensus,
        "emission_root": root,
        "evidence": evidence,
        "participants": {
            "miner": {"name": "Alice", "hotkey": alice.ss58_address},
            "validators": [
                {"name": name, "hotkey": keypair.ss58_address}
                for name, keypair in validators
            ],
        },
    }
    _write_json(STATE_PATH, state)
    return state


def _local_seed_task(state: dict[str, Any]) -> dict[str, Any]:
    task = dict(dict(state.get("autoresearch") or {}).get("task") or {})
    if task.get("id"):
        return task
    tasks = _request_json("GET", "http://127.0.0.1:8000/api/v1/tasks")
    for row in tasks if isinstance(tasks, list) else []:
        if str(dict(row).get("slug") or "") == "sota-local-binary-frontier":
            return dict(row)
    raise RuntimeError("local autoresearch seed task is missing; start the local SOTA Base stack first")


def _next_sota_emission_epoch(lane_id: str) -> int:
    roots = _request_json("GET", f"http://127.0.0.1:8000/api/v1/sota/emission-roots?subnet_id={lane_id}")
    max_epoch = 0
    for row in roots if isinstance(roots, list) else []:
        try:
            max_epoch = max(max_epoch, int(dict(row).get("epoch") or 0))
        except Exception:
            continue
    return max_epoch + 1


def _extract_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for index, char in enumerate(str(text or "")):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not text[index + end :].strip():
            found = value
    if found is None:
        raise RuntimeError("miner process did not print a final JSON result")
    return found


def _claimable_amount_units(eligibility: dict[str, Any], *, root_id: str) -> int:
    total = 0
    for allocation in list(eligibility.get("allocations") or []):
        item = dict(allocation or {})
        if str(item.get("root_id") or "").lower() == root_id.lower() and not bool(item.get("claimed")):
            total += int(item.get("amount") or 0)
    return total


def _send_local_private_key_tx(w3: Web3, private_key: str, tx: dict[str, Any]) -> str:
    account = Account.from_key(private_key)
    value_text = str(tx.get("value") or "0x0")
    value = int(value_text, 16 if value_text.startswith("0x") else 10)
    transaction = {
        "to": Web3.to_checksum_address(str(tx["to"])),
        "from": account.address,
        "data": str(tx["data"]),
        "value": value,
        "nonce": w3.eth.get_transaction_count(account.address),
        "chainId": int(tx.get("chainId") or w3.eth.chain_id),
        "gasPrice": int(w3.eth.gas_price),
    }
    transaction["gas"] = int(w3.eth.estimate_gas(transaction))
    signed = Account.sign_transaction(transaction, private_key)
    raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
    tx_hash = w3.eth.send_raw_transaction(raw)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    if int(receipt.status) != 1:
        raise RuntimeError(f"local private-key transaction reverted: {tx_hash.hex()}")
    text = tx_hash.hex()
    return text if text.startswith("0x") else f"0x{text}"


def _fund_local_reward_accounts(w3: Web3, state: dict[str, Any], miners: list[dict[str, Any]]) -> None:
    sender = state["accounts"]["owner"]
    minimum_balance = 2 * 10**16
    top_up_amount = 10**17
    for miner in miners:
        reward_address = Web3.to_checksum_address(str(miner["reward_address"]))
        if int(w3.eth.get_balance(reward_address)) >= minimum_balance:
            continue
        tx_hash = w3.eth.send_transaction(
            {
                "from": sender,
                "to": reward_address,
                "value": top_up_amount,
            }
        )
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        if int(receipt.status) != 1:
            raise RuntimeError(f"failed to fund local reward address {reward_address}: {tx_hash.hex()}")


def _run_local_miner_processes(
    *,
    miners: list[dict[str, Any]],
    task_id: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    miner_root = RUN_DIR / "miners"
    log_root = LOG_DIR / "miners"
    miner_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    pythonpath_parts = [str(DOCS_REPO), str(AUTORESEARCH_REPO / "src")]
    if os.environ.get("PYTHONPATH"):
        pythonpath_parts.append(str(os.environ["PYTHONPATH"]))
    processes: list[dict[str, Any]] = []
    for miner in miners:
        workspace_root = miner_root / str(miner["name"])
        if workspace_root.exists():
            shutil.rmtree(workspace_root)
        workspace_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{miner['name']}.log"
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": os.pathsep.join(pythonpath_parts),
                "BITSOTA_LOCAL_MINER_INDEX": str(miner["index"]),
                "BITSOTA_RESEARCH_CLAIM_DESCRIPTION": (
                    f"local SOTA multi-miner swarm claim {miner['index']} for reward {miner['reward_address']}"
                ),
                "BITSOTA_EVM_MINER_PRIVATE_KEY": str(miner["miner_private_key"]),
                "BITSOTA_EVM_REWARD_PRIVATE_KEY": str(miner["reward_private_key"]),
                "BITSOTA_EVM_COMPETITION_ID": str(task_id),
                "BITSOTA_EVM_LANE_ID": LANE_ID,
            }
        )
        command = [
            sys.executable,
            "-m",
            "neurons.research_agent_miner",
            "mine-once",
            "--coordinator-url",
            "http://127.0.0.1:8000",
            "--task-id",
            str(task_id),
            "--hotkey-uri",
            str(miner["hotkey_uri"]),
            "--workspace-root",
            str(workspace_root),
            "--agent-command",
            f"{sys.executable} {DOCS_REPO / 'scripts' / 'sota_local_miner_agent.py'}",
            "--agent-mode",
            "gui_managed",
        ]
        handle = log_path.open("wb")
        process = subprocess.Popen(
            command,
            cwd=DOCS_REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(
            {
                "miner": miner,
                "workspace_root": workspace_root,
                "log_path": log_path,
                "handle": handle,
                "process": process,
                "pid": process.pid,
            }
        )

    results: list[dict[str, Any]] = []
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    for item in processes:
        process: subprocess.Popen = item["process"]
        remaining = max(1.0, deadline - time.monotonic())
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                pass
            raise RuntimeError(f"miner process timed out pid={process.pid} log={item['log_path']}") from exc
        finally:
            item["handle"].close()
        log_text = Path(item["log_path"]).read_text(encoding="utf-8", errors="replace")
        if int(returncode) != 0:
            raise RuntimeError(f"miner process exited {returncode} pid={process.pid} log={item['log_path']}")
        payload = _extract_last_json_object(log_text)
        miner = dict(item["miner"])
        submission = dict(payload.get("submission") or {})
        claim = dict(payload.get("claim") or {})
        results.append(
            {
                "name": miner["name"],
                "index": miner["index"],
                "pid": item["pid"],
                "returncode": int(returncode),
                "workspace_root": str(item["workspace_root"]),
                "log_path": str(item["log_path"]),
                "hotkey": miner["hotkey"],
                "miner_address": miner["miner_address"],
                "reward_address": miner["reward_address"],
                "claim_id": str(claim.get("id") or ""),
                "submission_id": str(submission.get("id") or ""),
                "claimed_metrics": dict(submission.get("claimed_metrics") or {}),
                "submission": submission,
                "claim": claim,
            }
        )
    return results


def _self_validate_swarm_submissions(miner_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validators = _demo_validator_keypairs()
    validated: list[dict[str, Any]] = []
    for result in miner_results:
        submission = dict(result.get("submission") or {})
        submission_id = str(submission.get("id") or "")
        if not submission_id:
            raise RuntimeError(f"miner result is missing submission id: {result}")
        claimed_metrics = dict(submission.get("claimed_metrics") or result.get("claimed_metrics") or {})
        observed_metric = float(claimed_metrics.get("heldout_ppl") or 0.0)
        evaluation_path = f"/api/v1/submissions/{submission_id}/peer-evaluate"
        evaluations = []
        for name, validator in validators:
            body = {
                "status": "accepted",
                "observed_metrics": {"heldout_ppl": observed_metric},
                "notes": f"{name} accepts local miner swarm submission {submission_id}",
            }
            evaluations.append(
                _request_json(
                    "POST",
                    f"http://127.0.0.1:8000{evaluation_path}",
                    body,
                    headers=_signed_headers(validator, "POST", evaluation_path, body),
                )
            )
        consensus = _request_json(
            "GET",
            f"http://127.0.0.1:8000/api/v1/submissions/{submission_id}/peer-consensus",
        )
        if int(consensus.get("committee_count") or 0) < DEMO_SELF_VALIDATION_COMMITTEE_SIZE:
            raise RuntimeError(f"swarm committee too small for {submission_id}: {consensus}")
        if int(consensus.get("accepted_count") or 0) < DEMO_SELF_VALIDATION_COMMITTEE_SIZE:
            raise RuntimeError(f"swarm submission was not fully accepted for {submission_id}: {consensus}")
        validated.append({**result, "evaluations": evaluations, "consensus": consensus})
    return validated


def _claim_swarm_emissions(
    *,
    state: dict[str, Any],
    published: dict[str, Any],
    miners: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    w3 = _wait_rpc()
    _fund_local_reward_accounts(w3, state, miners)
    claim_results: list[dict[str, Any]] = []
    root_id = str(published["root_id"])
    token = _contract(w3, "SOTAToken", state["contracts"]["sota_token"])
    for miner in miners:
        reward_address = Web3.to_checksum_address(str(miner["reward_address"]))
        eligibility = _request_json(
            "GET",
            f"http://127.0.0.1:8010/api/v1/base/eligibility/{reward_address}?root_id={root_id}",
        )
        claimable = _claimable_amount_units(dict(eligibility), root_id=root_id)
        if claimable <= 0:
            raise RuntimeError(f"expected positive SOTA emission claim for {reward_address}")
        tx_payload = _request_json(
            "POST",
            "http://127.0.0.1:8010/api/v1/base/claims/transaction",
            {
                "program": "emission",
                "evmAddress": reward_address,
                "laneId": LANE_ID,
                "rootId": root_id,
            },
        )
        tx_hash = _send_local_private_key_tx(w3, str(miner["reward_private_key"]), dict(tx_payload["transaction"]))
        _request_json("POST", "http://127.0.0.1:8010/api/v1/base/index/sync")
        claimed = _request_json(
            "GET",
            f"http://127.0.0.1:8010/api/v1/base/eligibility/{reward_address}?root_id={root_id}",
        )
        if dict(claimed.get("claim_state") or {}).get("status") != "claimed":
            raise RuntimeError(f"expected claimed indexer state for {reward_address}: {claimed.get('claim_state')}")
        balance = int(token.functions.balanceOf(reward_address).call())
        if balance < claimable:
            raise RuntimeError(f"expected {reward_address} SOTA balance >= {claimable}, got {balance}")
        claim_results.append(
            {
                "reward_address": reward_address,
                "amount_units": claimable,
                "tx_hash": tx_hash,
                "sota_balance_units": balance,
                "claim_state": dict(claimed.get("claim_state") or {}),
            }
        )
    return claim_results


def run_local_miner_swarm(
    *,
    count: int = DEMO_SWARM_MINER_COUNT,
    epoch: int | None = None,
    report_out: Path = RUN_DIR / "miner-swarm" / "latest.json",
    claim: bool = True,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    _wait_http("http://127.0.0.1:8000/readyz", timeout_seconds=15)
    _wait_http("http://127.0.0.1:8010/health", timeout_seconds=15)
    state = _load_json(STATE_PATH)
    if not state:
        raise RuntimeError("local SOTA Base state is missing; start the local stack first")
    if count < 1:
        raise ValueError("miner count must be positive")
    task = _local_seed_task(state)
    task_id = str(task["id"])
    miners = [_demo_swarm_miner(index) for index in range(1, count + 1)]
    for field in ("hotkey", "miner_address", "reward_address"):
        values = [str(miner[field]).lower() for miner in miners]
        if len(set(values)) != len(values):
            raise RuntimeError(f"duplicate local swarm {field} values")
    miner_results = _run_local_miner_processes(
        miners=miners,
        task_id=task_id,
        timeout_seconds=timeout_seconds,
    )
    validated = _self_validate_swarm_submissions(miner_results)
    root_epoch = int(epoch or _next_sota_emission_epoch(LANE_ID))
    root = _request_json(
        "POST",
        f"http://127.0.0.1:8000/api/v1/sota/subnets/{LANE_ID}/epochs/{root_epoch}/root",
        {"include_proofs": True},
        headers={"X-Admin-Token": ADMIN_TOKEN},
    )
    evidence = _request_json("GET", f"http://127.0.0.1:8000/api/v1/sota/subnets/{LANE_ID}/epochs/{root_epoch}/evidence")
    bundle = dict(evidence.get("bundle") or {})
    claims = [dict(item) for item in list(bundle.get("claim_list") or [])]
    reward_addresses = {str(miner["reward_address"]).lower() for miner in miners}
    matching_claims = [
        claim_item
        for claim_item in claims
        if str(claim_item.get("reward_address") or "").lower() in reward_addresses
    ]
    if len(matching_claims) < count:
        raise RuntimeError(
            f"swarm root contains {len(matching_claims)} matching reward leaves for {count} local miners"
        )
    published = _publish_emission_artifact(
        state,
        root=root,
        evidence=evidence,
        nonce_label=f"local-miner-swarm-root-v{root_epoch}",
    )
    claim_results = _claim_swarm_emissions(state=state, published=published, miners=miners) if claim else []
    public_miners = [
        {key: value for key, value in miner.items() if not key.endswith("_private_key")}
        for miner in miners
    ]
    report = {
        "schema": "sota-local-multi-miner/v1",
        "ok": True,
        "miner_count": count,
        "accepted_count": len(validated),
        "committee_size": DEMO_SELF_VALIDATION_COMMITTEE_SIZE,
        "epoch": root_epoch,
        "lane_id": LANE_ID,
        "task_id": task_id,
        "task_slug": task.get("slug"),
        "processes": [
            {
                key: value
                for key, value in item.items()
                if key not in {"submission", "claim", "evaluations"}
            }
            for item in validated
        ],
        "miners": public_miners,
        "emission_root": root,
        "published": {key: value for key, value in published.items() if key != "artifact"},
        "claim_count": len(claims),
        "matching_claim_count": len(matching_claims),
        "matching_claims": matching_claims,
        "claim_transactions": claim_results,
        "checks": {
            "distinct_hotkeys": len({miner["hotkey"].lower() for miner in public_miners}) == count,
            "distinct_miner_addresses": len({miner["miner_address"].lower() for miner in public_miners}) == count,
            "distinct_reward_addresses": len({miner["reward_address"].lower() for miner in public_miners}) == count,
            "all_processes_exited_zero": all(int(item["returncode"]) == 0 for item in validated),
            "all_self_validation_accepted": all(
                str(dict(item.get("consensus") or {}).get("status") or "") == "accepted"
                for item in validated
            ),
            "all_claims_submitted": (not claim) or len(claim_results) == count,
        },
        "does_not": [
            "touch production Bittensor",
            "touch Base mainnet",
            "use TAO or alpha token transfers",
        ],
    }
    if not all(bool(value) for value in dict(report["checks"]).values()):
        report["ok"] = False
        _write_json(report_out, report)
        raise RuntimeError(f"local miner swarm checks failed; see {report_out}")
    state["local_miner_swarm"] = {
        "report_path": str(report_out),
        "epoch": root_epoch,
        "root_id": published["root_id"],
        "miner_count": count,
        "claimed": bool(claim),
    }
    _write_json(STATE_PATH, state)
    _write_json(report_out, report)
    return report


def swarm_smoke(
    *,
    count: int = DEMO_SWARM_MINER_COUNT,
    report_out: Path = RUN_DIR / "miner-swarm" / "latest.json",
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    try:
        start_stack(website=False, docs=False, hold=False)
        report = run_local_miner_swarm(
            count=count,
            report_out=report_out,
            claim=True,
            timeout_seconds=timeout_seconds,
        )
        _print("local miner swarm smoke passed")
        return report
    finally:
        stop_stack()


def _publish_emission_artifact(
    state: dict[str, Any],
    *,
    root: dict[str, Any],
    evidence: dict[str, Any],
    nonce_label: str,
) -> dict[str, Any]:
    w3 = _wait_rpc()
    contracts = state["contracts"]
    accounts = state["accounts"]
    root_registry = _contract(w3, "SOTARootRegistry", contracts["root_registry"])
    lane_registry = _contract(w3, "SOTALaneRegistry", contracts["lane_registry"])
    bundle = dict(evidence.get("bundle") or evidence)
    offchain_lane_id = bundle["subnet"]["offchain_lane_id"]
    policy_hash = root["policy_hash"]
    attestation_hash = "0x" + str(root["evidence_hash"]).removeprefix("0x")[:64]
    if int(attestation_hash, 16) == 0:
        attestation_hash = _hex32(_bytes32_text("local-emission-attestation"))
    lane_registry.functions.setLane(
        bytes.fromhex(offchain_lane_id[2:]),
        int(root["total_amount_units"]),
        True,
        bytes.fromhex(policy_hash[2:]),
    ).transact({"from": accounts["owner"]})
    root_id = _publish_root(
        w3,
        root_registry,
        publisher=accounts["publisher"],
        kind=2,
        merkle_root=root["root"],
        budget_cap=int(root["total_amount_units"]),
        policy_hash=policy_hash,
        attestation_hash=attestation_hash,
        nonce=_hex32(_bytes32_text(nonce_label)),
    )
    artifact = {
        "subnet": {
            **dict(bundle.get("subnet") or {}),
            "id": LANE_ID,
            "title": "SOTA local binary frontier",
            "owner": accounts["owner"],
            "metadata_uri": "local://sota/lane/binary-frontier",
            "token": "SOTA",
        },
        "root": {
            "root_id": root_id,
            "subnet_id": LANE_ID,
            "epoch": int(root["epoch"]),
            "root": root["root"],
            "total_amount_units": int(root["total_amount_units"]),
            "budget": int(root["total_amount_units"]),
            "status": "finalized",
            "validation_status": "accepted",
        },
        "claim_list": bundle["claim_list"],
        "leaves": bundle["leaves"],
    }
    _request_json("POST", "http://127.0.0.1:8010/api/v1/base/index/artifact", artifact)
    return {
        "root_id": root_id,
        "offchain_lane_id": offchain_lane_id,
        "amount": int(root["total_amount_units"]),
        "artifact": artifact,
    }


def publish_emission_onchain_and_indexer(state: dict[str, Any]) -> dict[str, Any]:
    published = _publish_emission_artifact(
        state,
        root=state["autoresearch"]["emission_root"],
        evidence=state["autoresearch"]["evidence"],
        nonce_label="local-emission-root-v1",
    )
    state["emission_onchain"] = {
        "root_id": published["root_id"],
        "offchain_lane_id": published["offchain_lane_id"],
        "amount": int(published["amount"]),
    }
    _write_json(STATE_PATH, state)
    return state


def _start_anvil() -> None:
    if _is_port_open("127.0.0.1", 8545):
        raise _port_in_use_error(8545)
    anvil = str(Path.home() / ".foundry" / "bin" / "anvil")
    _start_process(
        "anvil",
        [
            anvil,
            "--host",
            ANVIL_HOST,
            "--port",
            "8545",
            "--chain-id",
            str(CHAIN_ID),
            "--mnemonic",
            DEMO_MNEMONIC,
        ],
        cwd=RUN_DIR,
    )
    _wait_rpc()


def _start_indexer(state: dict[str, Any]) -> None:
    if _is_port_open("127.0.0.1", 8010):
        raise _port_in_use_error(8010)
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(COMMUNITY_REPO),
        "SOTA_BASE_INDEXER_DB": str(RUN_DIR / "indexer.sqlite3"),
        "SOTA_BASE_CHAIN_ID": str(CHAIN_ID),
        "SOTA_BASE_RPC_URL": ANVIL_RPC_LOCAL,
        "SOTA_BASE_SYNC_FROM_BLOCK": "0",
        "SOTA_BASE_CONTRACTS_ABI_DIR": str(CONTRACTS_REPO / "abi"),
        "SOTA_ROOT_REGISTRY_ADDRESS": state["contracts"]["root_registry"],
        "SOTA_LANE_REGISTRY_ADDRESS": state["contracts"]["lane_registry"],
        "SOTA_GENESIS_DISTRIBUTOR_ADDRESS": state["contracts"]["genesis_distributor"],
        "SOTA_EMISSION_DISTRIBUTOR_ADDRESS": state["contracts"]["emission_distributor"],
        "SOTA_BASE_CORS_ORIGINS": "http://127.0.0.1:3000,http://localhost:3000",
    }
    _start_process(
        "indexer",
        [
            _python(COMMUNITY_REPO),
            "-m",
            "uvicorn",
            "experiments.base_protocol_design.sota_base_indexer.api:create_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            "8010",
        ],
        cwd=COMMUNITY_REPO,
        env=env,
    )
    _wait_http("http://127.0.0.1:8010/health")


def _start_autoresearch() -> None:
    if _is_port_open("127.0.0.1", 8000):
        raise _port_in_use_error(8000)
    validator_hotkeys = ",".join(keypair.ss58_address for _, keypair in _demo_validator_keypairs())
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(AUTORESEARCH_REPO / "src"),
        "DATABASE_URL": f"sqlite:///{RUN_DIR / 'autoresearch.sqlite3'}",
        "ADMIN_TOKEN": ADMIN_TOKEN,
        "AUTH_MAX_AGE_SECONDS": "300",
        "MINER_AUTH_STAKE_GATE_ENABLED": "0",
        "SOTA_DEFAULT_LANE_ID": LANE_ID,
        "VALIDATOR_HOTKEYS": validator_hotkeys,
        "VALIDATOR_LEGACY_VERIFY_ENABLED": "0",
        "BOOTSTRAP_BUILTIN_COMPETITIONS_ON_STARTUP": "0",
        "CLAIM_RATE_LIMIT_COUNT": "100",
        "SUBMISSION_RATE_LIMIT_COUNT": "100",
        "VALIDATOR_ALLOW_LOCAL_ARTIFACT_URIS": "1",
        "VALIDATOR_ALLOW_UNSAFE_HOST_MODE": "1",
        "VALIDATOR_SANDBOX_MODE": "host",
    }
    _start_process(
        "autoresearch",
        [
            _python(AUTORESEARCH_REPO),
            "-m",
            "uvicorn",
            "autoresearch_bittensor.api.app:create_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        cwd=AUTORESEARCH_REPO,
        env=env,
    )
    _wait_http("http://127.0.0.1:8000/readyz")


def _start_website(state: dict[str, Any], *, browser_rpc_url: str) -> None:
    if _is_port_open("127.0.0.1", 3000):
        raise _port_in_use_error(3000)
    env = {
        "NEXT_PUBLIC_SOTA_CLAIMS_API_URL": "http://127.0.0.1:8010",
        "NEXT_PUBLIC_SOTA_BASE_CHAIN_ID": str(CHAIN_ID),
        "NEXT_PUBLIC_SOTA_BASE_CHAIN_NAME": "SOTA Local Base",
        "NEXT_PUBLIC_SOTA_BASE_RPC_URL": browser_rpc_url,
        "NEXT_PUBLIC_SOTA_BASE_EXPLORER_URL": browser_rpc_url,
        "NEXT_PUBLIC_SOTA_ENVIRONMENT": "local",
        "NEXT_PUBLIC_SOTA_DEMO_ENABLED": "true",
        "NEXT_PUBLIC_SOTA_DEMO_EVM_ADDRESS": state["accounts"]["alice_reward"],
        "NEXT_PUBLIC_SOTA_DEMO_OLD_COLDKEY": state["genesis"]["old_coldkey"],
        "NEXT_PUBLIC_SOTA_DEFAULT_LANE_ID": LANE_ID,
        "NEXT_PUBLIC_SOTA_AUTORESEARCH_API_URL": "http://127.0.0.1:8000",
        "NEXT_PUBLIC_SOTA_TOKEN_ADDRESS": state["contracts"]["sota_token"],
        "NEXT_PUBLIC_SOTA_GENESIS_DISTRIBUTOR_ADDRESS": state["contracts"]["genesis_distributor"],
        "NEXT_PUBLIC_SOTA_EMISSION_DISTRIBUTOR_ADDRESS": state["contracts"]["emission_distributor"],
    }
    _start_process("website", ["corepack", "pnpm", "dev", "-p", "3000", "-H", "0.0.0.0"], cwd=WEBSITE_REPO, env=env)
    _wait_http("http://127.0.0.1:3000/claims", timeout_seconds=120)


def _start_docs() -> None:
    if _is_port_open("127.0.0.1", 9002):
        _print("docs port 9002 is already open; using the existing docs server")
        return
    python = str(DOCS_REPO / ".venv-docs" / "bin" / "python")
    if not Path(python).exists():
        python = sys.executable
    _start_process(
        "docs",
        [python, "-m", "mkdocs", "serve", "-a", "0.0.0.0:9002"],
        cwd=DOCS_REPO,
    )
    _wait_http("http://127.0.0.1:9002/base/", timeout_seconds=120)


def _run_local_ui_smoke_report() -> None:
    subprocess.run(
        [
            sys.executable,
            str(DOCS_REPO / "scripts" / "sota_local_claims_ui_smoke.py"),
            "--report-out",
            str(RUN_DIR / "ui-smoke" / "report.json"),
            "--skip-screenshot",
        ],
        cwd=DOCS_REPO,
        check=True,
        text=True,
        capture_output=True,
    )


def _run_tailscale_preflight_report() -> None:
    subprocess.run(
        [
            sys.executable,
            str(DOCS_REPO / "scripts" / "sota_local_tailscale_preflight.py"),
            "--report-out",
            str(RUN_DIR / "tailscale-preflight.json"),
            "--allow-blocked",
        ],
        cwd=DOCS_REPO,
        check=True,
        text=True,
        capture_output=True,
    )


def _run_local_claim_proof_reset() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(DOCS_REPO / "scripts" / "sota_local_claim_proof.py"),
            "--reset-after",
            "--report-out",
            str(RUN_DIR / "claim-proof" / "latest.json"),
            "--evidence-out",
            str(RUN_DIR / "claim-proof" / "local-claim-tx-evidence.json"),
        ],
        cwd=DOCS_REPO,
        check=False,
        text=True,
        capture_output=True,
        timeout=420,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"local claim proof exited {result.returncode}")


def _generate_release_status_report() -> None:
    subprocess.run(
        [
            sys.executable,
            str(DOCS_REPO / "scripts" / "sota_base_release_status.py"),
            "--report-out",
            str(TESTNET_RUN_DIR / "base-sota-release-status.json"),
            "--allow-blocked",
        ],
        cwd=DOCS_REPO,
        check=True,
        text=True,
        capture_output=True,
    )


def _generate_handoff() -> None:
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(DOCS_REPO / "scripts" / "sota_base_tester_handoff.py"),
            "--environment",
            "local",
            "--json-out",
            str(HANDOFF_DIR / "handoff.json"),
            "--markdown-out",
            str(HANDOFF_DIR / "handoff.md"),
            "--html-out",
            str(HANDOFF_DIR / "index.html"),
        ],
        cwd=DOCS_REPO,
        check=True,
        text=True,
        capture_output=True,
    )


def _start_handoff() -> None:
    if _is_port_open("127.0.0.1", 9003):
        raise _port_in_use_error(9003)
    _start_process(
        "handoff",
        [
            sys.executable,
            "-m",
            "http.server",
            "9003",
            "--bind",
            "0.0.0.0",
            "--directory",
            str(HANDOFF_DIR),
        ],
        cwd=RUN_DIR,
    )
    _wait_http("http://127.0.0.1:9003/", timeout_seconds=30)


def _refresh_tester_artifacts() -> None:
    _run_local_ui_smoke_report()
    _run_tailscale_preflight_report()
    _generate_release_status_report()
    _generate_handoff()


def start_stack(
    *,
    website: bool = True,
    docs: bool = True,
    hold: bool = True,
    claim_proof: bool = False,
    share_mode: str = "auto",
    share_warning_override: str = "",
) -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stop_stack()
    try:
        _reset_runtime_state()
        _start_anvil()
        state = deploy_contracts()
        _write_json(STATE_PATH, state)
        _start_indexer(state)
        state = seed_genesis_onchain_and_indexer(state)
        _start_autoresearch()
        state = seed_autoresearch_and_emission(state)
        state = publish_emission_onchain_and_indexer(state)
        urls, sharing = _plan_public_share(
            share_mode,
            require_remote_wallet=website and docs,
            warning_override=share_warning_override,
        )
        state["urls"] = urls
        state["sharing"] = sharing
        if website:
            _start_website(state, browser_rpc_url=urls["anvil_rpc"])
        if docs:
            _start_docs()
        if website and docs:
            _write_json(STATE_PATH, state)
            _generate_handoff()
            _start_handoff()
            try:
                sharing = _activate_public_share(sharing)
            except RuntimeError as exc:
                if share_mode == "auto":
                    fallback_warning = (
                        "Tailscale Serve HTTPS is unavailable, so this run is using wallet-safe localhost URLs. "
                        f"{exc}. Enable Tailscale Serve for this node, run `sudo tailscale set --operator=$USER` once if the CLI reports operator access denied, then relaunch with `./scripts/sota_local_demo.py launch --share-mode tailscale-https` for remote MetaMask testing."
                    )
                    _print(f"Tailscale Serve HTTPS unavailable, falling back to wallet-safe localhost URLs: {exc}")
                    stop_stack()
                    return start_stack(
                        website=website,
                        docs=docs,
                        hold=hold,
                        claim_proof=claim_proof,
                        share_mode="localhost",
                        share_warning_override=fallback_warning,
                    )
                raise
            state["sharing"] = sharing
            _write_json(STATE_PATH, state)
        _write_json(STATE_PATH, state)
        if website and docs:
            _refresh_tester_artifacts()
            if claim_proof:
                _print("running state-changing local claim proof and resetting to a fresh claimable stack...")
                _run_local_claim_proof_reset()
                state = _load_json(STATE_PATH)
                _refresh_tester_artifacts()
    except Exception:
        stop_stack()
        raise
    print_summary(state)
    if hold:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            stop_stack()
    return state


def print_summary(state: dict[str, Any]) -> None:
    urls = state.get("urls", {})
    sharing = dict(state.get("sharing") or {})
    _print("\nSOTA Base local demo is ready.")
    _print(f"Claims UI: {urls.get('claims_ui', 'http://127.0.0.1:3000/claims')}")
    _print(f"Autoresearch dashboard: {urls.get('autoresearch_dashboard', 'http://127.0.0.1:8000/dashboard')}")
    _print(f"Docs: {urls.get('docs', 'http://127.0.0.1:9002/base/')}")
    if urls.get("handoff"):
        _print(f"Tester handoff: {urls['handoff']}")
    _print(f"Anvil RPC for MetaMask: {urls.get('anvil_rpc', ANVIL_RPC_LOCAL)}")
    if sharing:
        _print(f"Share mode: {sharing.get('mode', 'unknown')} ({sharing.get('status', 'unknown')})")
        if sharing.get("warning"):
            _print(f"Share warning: {sharing['warning']}")
    _print("\nImport this local-only account in MetaMask:")
    _print(f"Private key: {ANVIL_PRIVATE_KEYS['alice_reward']}")
    _print(f"Address: {state['accounts']['alice_reward']}")
    _print(f"Old coldkey for genesis lookup: {state['genesis']['old_coldkey']}")
    _print("\nThe seeded miner submission has self-validation consensus:")
    _print(json.dumps(state["autoresearch"]["consensus"], indent=2, sort_keys=True))
    validators = [item["name"] for item in state["autoresearch"].get("participants", {}).get("validators", [])]
    if validators:
        _print(f"Peer validators: {', '.join(validators)}")


def smoke() -> None:
    try:
        state = start_stack(website=False, docs=False, hold=False)
        alice = state["accounts"]["alice_reward"]
        eligibility = _request_json("GET", f"http://127.0.0.1:8010/api/v1/base/eligibility/{alice}")
        if not eligibility.get("eligible"):
            raise RuntimeError("expected Alice to be eligible")
        genesis_tx = _request_json(
            "POST",
            "http://127.0.0.1:8010/api/v1/base/claims/transaction",
            {"program": "genesis", "rewardAddress": alice},
        )
        emission_tx = _request_json(
            "POST",
            "http://127.0.0.1:8010/api/v1/base/claims/transaction",
            {"program": "emission", "evmAddress": alice},
        )
        if not genesis_tx["transaction"]["data"].startswith("0x"):
            raise RuntimeError("missing genesis calldata")
        if not emission_tx["transaction"]["data"].startswith("0x"):
            raise RuntimeError("missing emission calldata")
        w3 = _wait_rpc()
        token = _contract(w3, "SOTAToken", state["contracts"]["sota_token"])
        alice_checksum = Web3.to_checksum_address(alice)
        expected_balance = int(state["genesis"]["amount"]) + int(state["emission_onchain"]["amount"])
        consensus = dict(state["autoresearch"]["consensus"])
        if int(consensus.get("committee_count") or 0) < DEMO_SELF_VALIDATION_COMMITTEE_SIZE:
            raise RuntimeError(f"expected multi-user self-validation committee: {consensus}")
        if int(consensus.get("accepted_count") or 0) < DEMO_SELF_VALIDATION_COMMITTEE_SIZE:
            raise RuntimeError(f"expected all local peer validators to accept: {consensus}")
        for tx in (genesis_tx["transaction"], emission_tx["transaction"]):
            tx_hash = w3.eth.send_transaction(
                {
                    "from": alice_checksum,
                    "to": Web3.to_checksum_address(tx["to"]),
                    "data": tx["data"],
                    "value": int(str(tx.get("value") or "0x0"), 16),
                }
            )
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status != 1:
                raise RuntimeError(f"claim transaction reverted: {tx_hash.hex()}")
        _request_json("POST", "http://127.0.0.1:8010/api/v1/base/index/sync")
        claimed = _request_json("GET", f"http://127.0.0.1:8010/api/v1/base/eligibility/{alice}")
        if claimed.get("claim_state", {}).get("status") != "claimed":
            raise RuntimeError(f"expected claimed indexer status, got {claimed.get('claim_state')}")
        balance = int(token.functions.balanceOf(alice_checksum).call())
        if balance != expected_balance:
            raise RuntimeError(f"expected SOTA balance {expected_balance}, got {balance}")
        consensus = state["autoresearch"]["consensus"]
        if consensus.get("status") != "accepted":
            raise RuntimeError(f"expected accepted self-validation consensus, got {consensus}")
        _print("local smoke passed")
    finally:
        stop_stack()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local SOTA Base fork demo stack.")
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch", help="start the full local tester stack, prove claims, reset fresh, and return")
    launch.add_argument(
        "--skip-claim-proof",
        action="store_true",
        help="skip the state-changing local claim proof; used by the proof reset path",
    )
    launch.add_argument(
        "--share-mode",
        choices=("auto", "localhost", "http", "tailscale-https"),
        default="auto",
        help="how to publish browser-facing local URLs; auto uses Tailscale Serve HTTPS when available and otherwise wallet-safe localhost",
    )
    start = sub.add_parser("start", help="start the full local stack and keep it running")
    start.add_argument("--no-website", action="store_true", help="skip Next.js claims UI")
    start.add_argument("--no-docs", action="store_true", help="skip MkDocs")
    start.add_argument("--detach", action="store_true", help="start the full stack and return after readiness checks")
    start.add_argument(
        "--share-mode",
        choices=("auto", "localhost", "http", "tailscale-https"),
        default="auto",
        help="how to publish browser-facing local URLs; auto uses Tailscale Serve HTTPS when available and otherwise wallet-safe localhost",
    )
    sub.add_parser("stop", help="stop processes started by this launcher")
    sub.add_parser("status", help="print the last demo state")
    sub.add_parser("smoke", help="run a noninteractive local E2E smoke and stop")
    miner_swarm = sub.add_parser(
        "miner-swarm",
        help="run multiple local miner processes against an already-running stack and publish/claim their emission root",
    )
    miner_swarm.add_argument("--count", type=int, default=DEMO_SWARM_MINER_COUNT)
    miner_swarm.add_argument("--epoch", type=int, default=0, help="emission epoch; default picks the next local epoch")
    miner_swarm.add_argument("--report-out", type=Path, default=RUN_DIR / "miner-swarm" / "latest.json")
    miner_swarm.add_argument("--timeout-seconds", type=float, default=180.0)
    miner_swarm.add_argument("--skip-claims", action="store_true", help="publish/index the root but do not submit claim txs")
    miner_swarm.add_argument("--json", action="store_true", help="print the non-secret JSON report")
    swarm_smoke_parser = sub.add_parser(
        "swarm-smoke",
        help="start a fresh local stack, run multiple real miner processes, claim their emissions, and stop",
    )
    swarm_smoke_parser.add_argument("--count", type=int, default=DEMO_SWARM_MINER_COUNT)
    swarm_smoke_parser.add_argument("--report-out", type=Path, default=RUN_DIR / "miner-swarm" / "latest.json")
    swarm_smoke_parser.add_argument("--timeout-seconds", type=float, default=180.0)
    swarm_smoke_parser.add_argument("--json", action="store_true", help="print the non-secret JSON report")
    ui_smoke = sub.add_parser("ui-smoke", help="verify the running claims UI, proxy APIs, and self-validation evidence")
    ui_smoke.add_argument("--skip-screenshot", action="store_true", help="skip optional Firefox screenshot")
    ui_smoke.add_argument("--report-out", type=Path, default=RUN_DIR / "ui-smoke" / "report.json")
    args = parser.parse_args()
    if args.command == "stop":
        stop_stack()
        return 0
    if args.command == "status":
        state = _load_json(STATE_PATH)
        if not state:
            raise SystemExit("no local SOTA Base state found")
        print_summary(state)
        return 0
    if args.command == "smoke":
        smoke()
        return 0
    if args.command == "miner-swarm":
        report = run_local_miner_swarm(
            count=args.count,
            epoch=args.epoch or None,
            report_out=args.report_out,
            claim=not args.skip_claims,
            timeout_seconds=args.timeout_seconds,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
        else:
            _print("local miner swarm passed")
            _print(f"Report: {args.report_out}")
            _print(f"Miners: {report['miner_count']}")
            _print(f"Root id: {report['published']['root_id']}")
        return 0
    if args.command == "swarm-smoke":
        report = swarm_smoke(
            count=args.count,
            report_out=args.report_out,
            timeout_seconds=args.timeout_seconds,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
        else:
            _print(f"Report: {args.report_out}")
            _print(f"Miners: {report['miner_count']}")
            _print(f"Root id: {report['published']['root_id']}")
        return 0
    if args.command == "ui-smoke":
        smoke_script = DOCS_REPO / "scripts" / "sota_local_claims_ui_smoke.py"
        command = [sys.executable, str(smoke_script), "--report-out", str(args.report_out)]
        if args.skip_screenshot:
            command.append("--skip-screenshot")
        return subprocess.call(command, cwd=DOCS_REPO)
    if args.command == "launch":
        start_stack(
            website=True,
            docs=True,
            hold=False,
            claim_proof=not args.skip_claim_proof,
            share_mode=args.share_mode,
        )
        return 0
    if args.command == "start":
        start_stack(
            website=not args.no_website,
            docs=not args.no_docs,
            hold=not args.detach,
            share_mode=args.share_mode,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

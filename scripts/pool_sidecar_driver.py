#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import requests


def _default_sidecar_url() -> str:
    host = os.getenv("BITSOTA_SIDECAR_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.getenv("BITSOTA_SIDECAR_PORT", "8123").strip() or "8123"
    return f"http://{host}:{port}"


def _import_substrate() -> tuple[Any, Any]:
    try:
        from substrateinterface import Keypair, KeypairType

        return Keypair, KeypairType
    except Exception as e:
        raise RuntimeError("Missing dependency substrate-interface. Install requirements.txt first.") from e


def _make_keypair(seed: Optional[int]) -> Any:
    Keypair, KeypairType = _import_substrate()
    if seed is None:
        mnemonic = Keypair.generate_mnemonic()
    else:
        # Deterministic-ish mnemonic generation isn't supported by substrate-interface; fall back to random.
        mnemonic = Keypair.generate_mnemonic()
    return Keypair.create_from_mnemonic(mnemonic, crypto_type=KeypairType.SR25519)


@dataclass
class _DummyHotkey:
    keypair: Any

    @property
    def ss58_address(self) -> str:
        return str(self.keypair.ss58_address)

    def sign(self, message: Any) -> bytes:
        if isinstance(message, str):
            message = message.encode("utf-8")
        return self.keypair.sign(message)


@dataclass
class _DummyWallet:
    hotkey: _DummyHotkey


def _start_run(sidecar_url: str, run_id: Optional[str]) -> str:
    if not run_id:
        run_id = str(uuid4())
    r = requests.post(
        f"{sidecar_url.rstrip('/')}/runs/start",
        json={"run_id": str(run_id), "replace": True},
        timeout=2.0,
    )
    r.raise_for_status()
    payload = r.json() or {}
    return str(payload.get("run_id") or run_id)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless Pool driver using the sidecar job queue.")
    parser.add_argument("--pool-url", default="http://127.0.0.1:8434")
    parser.add_argument("--sidecar-url", default=os.getenv("BITSOTA_SIDECAR_URL", _default_sidecar_url()))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    sidecar_url = str(args.sidecar_url).rstrip("/")
    pool_url = str(args.pool_url).rstrip("/")

    # Create a local identity to sign pool requests.
    keypair = _make_keypair(args.seed)
    wallet = _DummyWallet(hotkey=_DummyHotkey(keypair))
    print(f"Hotkey: {wallet.hotkey.ss58_address}")

    # Start a sidecar run.
    run_id = _start_run(sidecar_url, args.run_id)
    print(f"Sidecar run_id: {run_id}")

    from gui.pool_task_driver import PoolApiClient, PoolTaskCoordinator, SidecarJobClient

    pool_client = PoolApiClient(pool_url, wallet, timeout_s=2.0)
    sidecar_jobs = SidecarJobClient(sidecar_url, run_id, timeout_s=0.5)
    coord = PoolTaskCoordinator(
        pool_client=pool_client,
        sidecar_jobs=sidecar_jobs,
        log=lambda msg: print(msg),
        request_interval_s=1.0,
    )

    deadline = time.time() + float(args.duration_s)
    while time.time() < deadline:
        coord.tick()
        time.sleep(0.2)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

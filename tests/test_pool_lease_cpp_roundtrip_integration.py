from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from substrateinterface import Keypair, KeypairType

from gui.pool_task_driver import PoolApiClient, PoolLeaseCoordinator, SidecarJobClient


class _Wallet:
    def __init__(self, keypair: Keypair) -> None:
        self.hotkey = keypair


def _wait_http_ok(url: str, *, timeout_s: float = 15.0) -> None:
    deadline = time.time() + float(timeout_s)
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=0.5)
            r.raise_for_status()
            return
        except Exception as e:  # pragma: no cover - exercised in real integration runs
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {url}: {last_err}")


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=8.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


@pytest.mark.integration
def test_pool_to_cpp_lease_worker_roundtrip_real() -> None:
    if os.getenv("BITSOTA_RUN_REAL_CPP_POOL_E2E", "0").strip() != "1":
        pytest.skip("Set BITSOTA_RUN_REAL_CPP_POOL_E2E=1 to run real Pool<->C++ lease integration test")

    repo_root = Path(__file__).resolve().parents[1]
    pool_url = str(os.getenv("BITSOTA_POOL_URL", "http://127.0.0.1:8434")).rstrip("/")
    sidecar_port = int(os.getenv("BITSOTA_E2E_SIDECAR_PORT", "18123"))
    sidecar_url = f"http://127.0.0.1:{sidecar_port}"
    run_id = f"cpp_e2e_{int(time.time())}"
    timeout_s = float(os.getenv("BITSOTA_E2E_TIMEOUT_S", "240"))
    automl_root = str(
        Path(
            os.getenv(
                "AUTOML_ZERO_CPP_ROOT",
                "/home/mekaneeky/repos/automl_zero_original/automl_zero",
            )
        ).expanduser()
    )

    # Validate required external services and binaries first.
    _wait_http_ok(f"{pool_url}/health", timeout_s=20.0)
    if not Path(automl_root).exists():
        pytest.skip(f"AUTOML_ZERO_CPP_ROOT missing: {automl_root}")

    sidecar_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sidecar",
            "--host",
            "127.0.0.1",
            "--port",
            str(sidecar_port),
        ],
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    miner_proc = None
    try:
        _wait_http_ok(f"{sidecar_url}/health", timeout_s=10.0)
        requests.post(
            f"{sidecar_url}/runs/start",
            json={"run_id": run_id, "replace": True},
            timeout=1.0,
        ).raise_for_status()

        env = os.environ.copy()
        env["BITSOTA_MINER_BACKEND"] = "cpp"
        env["BITSOTA_POOL_MINER_MODE"] = "real"
        env["BITSOTA_CPP_TASK_TYPE"] = "cifar10_binary"
        env["BITSOTA_CPP_RANDOM_SEED"] = env.get("BITSOTA_CPP_RANDOM_SEED", "42")
        miner_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.miner_cpp_sidecar",
                "--cpp-mode",
                "lease",
                "--sidecar-url",
                sidecar_url,
                "--run-id",
                run_id,
                "--workers",
                "1",
                "--mode",
                "real",
                "--lease-evolve-generations",
                "12",
                "--automl-root",
                automl_root,
                "--bitsota-root",
                str(repo_root),
            ],
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        keypair = Keypair.create_from_mnemonic(
            Keypair.generate_mnemonic(),
            crypto_type=KeypairType.SR25519,
        )
        wallet = _Wallet(keypair)
        pool_client = PoolApiClient(pool_url, wallet, timeout_s=3.0)
        sidecar_jobs = SidecarJobClient(sidecar_url, run_id, timeout_s=1.0)
        logs: list[str] = []
        coordinator = PoolLeaseCoordinator(
            pool_client=pool_client,
            sidecar_jobs=sidecar_jobs,
            log=logs.append,
            request_interval_s=0.2,
        )

        submit_line = None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            coordinator.tick()
            for line in logs:
                if "[pool] submit_lease ok=True" in str(line):
                    submit_line = str(line)
                    break
            if submit_line:
                break
            time.sleep(0.2)

        if submit_line is None:
            tail = "\n".join(logs[-30:])
            raise AssertionError(f"Did not observe successful lease submit within {timeout_s}s.\nRecent logs:\n{tail}")

        m = re.search(r"eval_n=(\d+)\s+evo_n=(\d+)\s+iter_n=(\d+)", submit_line)
        assert m, f"Could not parse submit line: {submit_line}"
        eval_n = int(m.group(1))
        evo_n = int(m.group(2))
        iter_n = int(m.group(3))

        assert evo_n > 0, f"Expected at least one evolution in lease submission, got line: {submit_line}"
        assert iter_n > 0, f"Expected positive iteration count, got line: {submit_line}"
        if eval_n == 0:
            assert any("bootstrapping evolution from empty seed set" in str(line) for line in logs), (
                "eval_n=0 is only expected when no compliant eval algorithms are available; "
                f"submit line={submit_line}"
            )
    finally:
        if miner_proc is not None:
            _terminate_process(miner_proc)
        _terminate_process(sidecar_proc)

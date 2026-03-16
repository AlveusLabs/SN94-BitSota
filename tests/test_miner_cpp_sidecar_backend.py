from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from typing import Any, Dict, List

import scripts.miner_cpp_sidecar as miner_cpp_sidecar


def test_default_bridge_and_baseline_cmd_env_override(monkeypatch) -> None:
    monkeypatch.setenv("AUTOML_ZERO_CPP_BRIDGE", "~/custom_bridge.py")
    monkeypatch.setenv("AUTOML_ZERO_CPP_BASELINE_CMD", "~/custom_run_baseline.sh")
    bridge = miner_cpp_sidecar._default_bridge_path("/ignored")
    baseline = miner_cpp_sidecar._default_baseline_cmd("/ignored")
    assert str(Path(bridge)).endswith("custom_bridge.py")
    assert str(Path(baseline)).endswith("custom_run_baseline.sh")


def test_load_seed_from_config_prefers_args_seed(tmp_path) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"args": {"seed": 77}, "seed": 13}), encoding="utf-8")
    assert miner_cpp_sidecar._load_seed_from_config(str(cfg)) == 77


def test_parse_extra_baseline_args_handles_invalid_quoting(monkeypatch) -> None:
    monkeypatch.setenv("AUTOML_ZERO_CPP_EXTRA_ARGS", "--foo 'unterminated")
    assert miner_cpp_sidecar._parse_extra_baseline_args() == []


def test_build_direct_worker_cmd_includes_expected_flags(monkeypatch) -> None:
    monkeypatch.setenv("AUTOML_ZERO_CPP_EXTRA_ARGS", "--foo bar --x=1")
    cmd = miner_cpp_sidecar._build_direct_worker_cmd(
        python="/usr/bin/python3",
        bridge_path="/opt/bridge.py",
        sidecar_url="http://127.0.0.1:8123",
        run_id="run-1",
        worker_id=2,
        task_type="cifar10_binary",
        input_dim=16,
        baseline_cmd="/opt/run_baseline.sh",
        iterations=0,
        random_seed=1234,
        echo_stdout=True,
    )
    assert cmd[:4] == ["/usr/bin/python3", "/opt/bridge.py", "direct", "--sidecar-url"]
    assert "--echo-stdout" in cmd
    assert "--max-experiments=0" in cmd
    assert "--iters=-1" in cmd
    assert "--random_seed=1234" in cmd
    assert cmd[-3:] == ["--foo", "bar", "--x=1"]


def test_build_lease_worker_cmd_includes_expected_flags(monkeypatch) -> None:
    monkeypatch.setenv("AUTOML_ZERO_CPP_EXTRA_ARGS", "")
    cmd = miner_cpp_sidecar._build_lease_worker_cmd(
        python="/usr/bin/python3",
        bridge_path="/opt/bridge.py",
        sidecar_url="http://127.0.0.1:8123",
        run_id="run-2",
        worker_id=0,
        baseline_cmd="/opt/run_baseline.sh",
        lease_iters=77,
        eval_mode="real",
        bitsota_root="/repo/current-sn-2",
        echo_stdout=False,
    )
    assert "lease" in cmd
    assert "--lease-iters" in cmd
    assert "77" in cmd
    assert "--eval-mode" in cmd
    assert "real" in cmd
    assert "--bitsota-root" in cmd


def test_main_direct_mode_normalizes_pool_task_type(monkeypatch) -> None:
    popen_calls: List[Dict[str, Any]] = []

    class _DoneProc:
        def __init__(self, cmd: List[str], *args, **kwargs) -> None:
            self.cmd = list(cmd)
            self.returncode = 0
            self.pid = 12345
            popen_calls.append({"cmd": self.cmd, "kwargs": dict(kwargs)})

        def poll(self):
            return self.returncode

        def wait(self, timeout: float | None = None):
            return self.returncode

    args = argparse.Namespace(
        cpp_mode="direct",
        sidecar_url="http://127.0.0.1:8123",
        run_id="cpp-direct-run",
        workers=2,
        task_type="pool_lease",
        iterations=0,
        feature_dim=16,
        config=None,
        mode="real",
        lease_evolve_generations=160,
        echo_stdout=False,
        sota_threshold=0.0,
        population_snapshot_every=0,
        initial_population_path=None,
        automl_root="/tmp/automl",
        bitsota_root="/tmp/bitsota",
    )
    monkeypatch.setattr(miner_cpp_sidecar, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(miner_cpp_sidecar, "_default_bridge_path", lambda root: "/tmp/bridge.py")
    monkeypatch.setattr(miner_cpp_sidecar, "_default_baseline_cmd", lambda root: "/tmp/run_baseline.sh")
    monkeypatch.setattr(miner_cpp_sidecar, "_load_seed_from_config", lambda cfg: 100)
    monkeypatch.setattr(miner_cpp_sidecar.subprocess, "Popen", _DoneProc)

    rc = miner_cpp_sidecar.main([])
    assert rc == 0
    assert len(popen_calls) == 2

    for call in popen_calls:
        cmd = call["cmd"]
        assert "direct" in cmd
        assert "--task-type" in cmd
        task_value = cmd[cmd.index("--task-type") + 1]
        assert task_value == "cifar10_binary"
        assert any(token.startswith("--random_seed=") for token in cmd)
        assert call["kwargs"].get("start_new_session") is True


def test_main_lease_mode_builds_real_eval_worker(monkeypatch) -> None:
    popen_calls: List[List[str]] = []

    class _DoneProc:
        def __init__(self, cmd: List[str], *args, **kwargs) -> None:
            self.cmd = list(cmd)
            self.returncode = 0
            self.pid = 999
            popen_calls.append(self.cmd)

        def poll(self):
            return self.returncode

        def wait(self, timeout: float | None = None):
            return self.returncode

    args = argparse.Namespace(
        cpp_mode="lease",
        sidecar_url="http://127.0.0.1:8123",
        run_id="cpp-lease-run",
        workers=1,
        task_type="cifar10_binary",
        iterations=0,
        feature_dim=16,
        config=None,
        mode="real",
        lease_evolve_generations=9,
        echo_stdout=True,
        sota_threshold=0.0,
        population_snapshot_every=0,
        initial_population_path=None,
        automl_root="/tmp/automl",
        bitsota_root="/tmp/bitsota",
    )
    monkeypatch.setattr(miner_cpp_sidecar, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(miner_cpp_sidecar, "_default_bridge_path", lambda root: "/tmp/bridge.py")
    monkeypatch.setattr(miner_cpp_sidecar, "_default_baseline_cmd", lambda root: "/tmp/run_baseline.sh")
    monkeypatch.setattr(miner_cpp_sidecar.subprocess, "Popen", _DoneProc)

    rc = miner_cpp_sidecar.main([])
    assert rc == 0
    assert len(popen_calls) == 1
    cmd = popen_calls[0]
    assert "lease" in cmd
    assert "--eval-mode" in cmd
    assert cmd[cmd.index("--eval-mode") + 1] == "real"
    assert cmd[cmd.index("--lease-iters") + 1] == "9"


def test_main_returns_nonzero_if_child_exits_nonzero(monkeypatch) -> None:
    class _DoneProc:
        def __init__(self, cmd: List[str], *args, **kwargs) -> None:
            self.cmd = list(cmd)
            self.returncode = 1
            self.pid = 222

        def poll(self):
            return self.returncode

        def wait(self, timeout: float | None = None):
            return self.returncode

    args = argparse.Namespace(
        cpp_mode="direct",
        sidecar_url="http://127.0.0.1:8123",
        run_id="cpp-direct-fail",
        workers=1,
        task_type="cifar10_binary",
        iterations=1,
        feature_dim=16,
        config=None,
        mode="real",
        lease_evolve_generations=10,
        echo_stdout=False,
        sota_threshold=0.0,
        population_snapshot_every=0,
        initial_population_path=None,
        automl_root="/tmp/automl",
        bitsota_root="/tmp/bitsota",
    )
    monkeypatch.setattr(miner_cpp_sidecar, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(miner_cpp_sidecar, "_default_bridge_path", lambda root: "/tmp/bridge.py")
    monkeypatch.setattr(miner_cpp_sidecar, "_default_baseline_cmd", lambda root: "/tmp/run_baseline.sh")
    monkeypatch.setattr(miner_cpp_sidecar.subprocess, "Popen", _DoneProc)
    rc = miner_cpp_sidecar.main([])
    assert rc == 1


def test_main_handles_keyboard_interrupt_and_stops_children(monkeypatch) -> None:
    terminate_called = {"value": False}

    class _RunningProc:
        def __init__(self, cmd: List[str], *args, **kwargs) -> None:
            self.cmd = list(cmd)
            self.returncode = None
            self.pid = 888

        def poll(self):
            return self.returncode

        def wait(self, timeout: float | None = None):
            return 0

    args = argparse.Namespace(
        cpp_mode="direct",
        sidecar_url="http://127.0.0.1:8123",
        run_id="cpp-direct-int",
        workers=1,
        task_type="cifar10_binary",
        iterations=1,
        feature_dim=16,
        config=None,
        mode="real",
        lease_evolve_generations=10,
        echo_stdout=False,
        sota_threshold=0.0,
        population_snapshot_every=0,
        initial_population_path=None,
        automl_root="/tmp/automl",
        bitsota_root="/tmp/bitsota",
    )
    monkeypatch.setattr(miner_cpp_sidecar, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(miner_cpp_sidecar, "_default_bridge_path", lambda root: "/tmp/bridge.py")
    monkeypatch.setattr(miner_cpp_sidecar, "_default_baseline_cmd", lambda root: "/tmp/run_baseline.sh")
    monkeypatch.setattr(miner_cpp_sidecar.subprocess, "Popen", _RunningProc)
    monkeypatch.setattr(
        miner_cpp_sidecar,
        "_terminate_children",
        lambda children: terminate_called.__setitem__("value", True),
    )
    monkeypatch.setattr(
        miner_cpp_sidecar.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    rc = miner_cpp_sidecar.main([])
    assert rc == 0
    assert terminate_called["value"] is True


def test_terminate_children_escalates_to_kill(monkeypatch) -> None:
    kill_calls: List[tuple[int, int]] = []

    class _HangingProc:
        def __init__(self, pid: int) -> None:
            self.pid = int(pid)

        def poll(self):
            return None

        def wait(self, timeout: float | None = None):
            raise TimeoutError("still running")

    monkeypatch.setattr(miner_cpp_sidecar.os, "name", "posix", raising=False)
    monkeypatch.setattr(
        miner_cpp_sidecar.os,
        "killpg",
        lambda pid, sig: kill_calls.append((int(pid), int(sig))),
    )

    miner_cpp_sidecar._terminate_children([_HangingProc(333)])

    assert (333, int(signal.SIGTERM)) in kill_calls
    assert (333, int(signal.SIGKILL)) in kill_calls

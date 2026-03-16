from __future__ import annotations

import sys
import threading
import types
from datetime import datetime, timezone
from typing import Any, Dict, List

import scripts.pool_miner_sidecar as pool_miner_sidecar


def _install_real_mode_stubs(monkeypatch) -> None:
    fake_eval_mod = types.ModuleType("core.evaluations")

    def _score_algorithm_on_eval_suite(dsl: str, *, input_dim: int) -> float:
        return 0.5 + (0.01 * float(max(1, int(input_dim))))

    fake_eval_mod.score_algorithm_on_eval_suite = _score_algorithm_on_eval_suite  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "core.evaluations", fake_eval_mod)

    fake_dsl_mod = types.ModuleType("core.dsl_parser")

    class _FakeDSLParser:
        @staticmethod
        def from_dsl(dsl: str, input_dim: int):
            return {"dsl": str(dsl), "input_dim": int(input_dim)}

        @staticmethod
        def to_dsl(algo: Any) -> str:
            if isinstance(algo, dict) and algo.get("dsl_out"):
                return str(algo["dsl_out"])
            if isinstance(algo, dict) and algo.get("dsl"):
                return str(algo["dsl"])
            return "def Predict():\n  s0 = dot(v0, v1)\n"

    fake_dsl_mod.DSLParser = _FakeDSLParser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "core.dsl_parser", fake_dsl_mod)

    fake_task_mod = types.ModuleType("core.tasks.cifar10")

    class _FakeTask:
        def load_data(self, task_id: int = 0) -> None:
            return None

    fake_task_mod.CIFAR10BinaryTask = _FakeTask  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "core.tasks.cifar10", fake_task_mod)

    fake_engine_mod = types.ModuleType("miner.engines.archive_engine")

    class _FakeEngine:
        def __init__(self, task: Any, pop_size: int = 5, verbose: bool = False) -> None:
            self.task = task
            self.pop_size = int(pop_size)
            self.verbose = bool(verbose)
            self.population: List[Any] = []
            self.best_algo: Dict[str, Any] | None = {"dsl_out": "def Predict():\n  s0 = dot(v0, v1)\n"}
            self._gen = 0

        def _random_mutate(self, algo: Any):
            self._gen += 1
            return {"dsl_out": f"def Predict():\n  s0 = dot(v0, v1)\n  #mut{self._gen}\n"}

        def evolve(self, generations: int = 1):
            self._gen += max(1, int(generations))
            self.best_algo = {"dsl_out": f"def Predict():\n  s0 = dot(v0, v1)\n  #evolve{self._gen}\n"}
            return self.best_algo, 0.9

        def evolve_generation(self) -> None:
            self._gen += 1
            self.best_algo = {"dsl_out": f"def Predict():\n  s0 = dot(v0, v1)\n  #gen{self._gen}\n"}

    fake_engine_mod.ArchiveAwareBaselineEvolution = _FakeEngine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "miner.engines.archive_engine", fake_engine_mod)


class _FakeSidecarClient:
    def __init__(self, jobs: List[Dict[str, Any]], stop: threading.Event) -> None:
        self._jobs = list(jobs)
        self._stop = stop
        self.logs: List[str] = []
        self.results: List[Dict[str, Any]] = []
        self.progress_events: List[Dict[str, Any]] = []

    def log(self, message: str) -> None:
        self.logs.append(str(message))

    def progress(self, iteration: int, *, rate: float | None = None) -> None:
        self.progress_events.append({"iteration": int(iteration), "rate": None if rate is None else float(rate)})

    def lease_job(self, *, lease_seconds: float) -> Dict[str, Any] | None:
        if self._jobs:
            return self._jobs.pop(0)
        return None

    def submit_result(self, job_id: str, *, ok: bool, result: Dict[str, Any], error: str | None) -> None:
        self.results.append(
            {
                "job_id": str(job_id),
                "ok": bool(ok),
                "result": dict(result or {}),
                "error": error,
            }
        )
        self._stop.set()


def test_default_sidecar_url_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("BITSOTA_SIDECAR_HOST", "10.0.0.2")
    monkeypatch.setenv("BITSOTA_SIDECAR_PORT", "8999")
    assert pool_miner_sidecar._default_sidecar_url() == "http://10.0.0.2:8999"


def test_numeric_helpers_are_stable() -> None:
    assert pool_miner_sidecar._clamp01(-1.0) == 0.0
    assert pool_miner_sidecar._clamp01(2.0) == 1.0
    assert pool_miner_sidecar._as_float("12.5") == 12.5
    assert pool_miner_sidecar._as_float("not-a-number") is None
    assert 0.0 <= pool_miner_sidecar._hash_to_unit_interval("seed-text") <= 1.0


def test_effective_submit_buffer_clamps_for_short_leases() -> None:
    clamped = pool_miner_sidecar._effective_submit_buffer_s(
        45.0,
        lease_timeout_at_s=110.0,
        now_s=100.0,
    )
    assert 0.0 <= clamped <= 10.0


def test_as_epoch_s_accepts_numeric_and_iso() -> None:
    assert pool_miner_sidecar._as_epoch_s(123.0) == 123.0
    assert pool_miner_sidecar._as_epoch_s("123.5") == 123.5

    iso = "2026-01-01T00:00:00Z"
    parsed = pool_miner_sidecar._as_epoch_s(iso)
    expected = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
    assert parsed is not None
    assert abs(parsed - expected) < 1e-6


def test_sidecar_client_roundtrip_methods(monkeypatch) -> None:
    calls: Dict[str, List[Dict[str, Any]]] = {"post": [], "get": []}

    class _Resp:
        def __init__(self, status_code: int, payload: Dict[str, Any] | None = None) -> None:
            self.status_code = int(status_code)
            self._payload = dict(payload or {})

        def json(self) -> Dict[str, Any]:
            return dict(self._payload)

    class _Session:
        def __init__(self) -> None:
            self._next_get = [
                _Resp(200, {"job": {"job_id": "job-1", "kind": "lease", "payload": {}}}),
                _Resp(204, {}),
            ]

        def post(self, url: str, json: Dict[str, Any], timeout: float):
            calls["post"].append({"url": str(url), "json": dict(json), "timeout": float(timeout)})
            return _Resp(200, {})

        def get(self, url: str, params: Dict[str, Any], timeout: float):
            calls["get"].append({"url": str(url), "params": dict(params), "timeout": float(timeout)})
            return self._next_get.pop(0)

    monkeypatch.setattr(pool_miner_sidecar.requests, "Session", _Session)
    client = pool_miner_sidecar._SidecarClient(
        "http://127.0.0.1:8123",
        run_id="run-a",
        worker_id="worker-a",
        timeout_s=1.0,
    )
    client.log("hello")
    client.progress(5, rate=0.2)
    first_job = client.lease_job(lease_seconds=60.0)
    second_job = client.lease_job(lease_seconds=60.0)
    client.submit_result("job-1", ok=True, result={"ok": True}, error=None)

    assert first_job is not None
    assert first_job.get("job_id") == "job-1"
    assert second_job is None
    assert len(calls["post"]) == 3
    assert len(calls["get"]) == 2


def test_run_worker_lease_real_evaluates_and_evolves(monkeypatch) -> None:
    _install_real_mode_stubs(monkeypatch)
    monkeypatch.setattr(pool_miner_sidecar.time, "sleep", lambda _: None)

    stop = threading.Event()
    job = {
        "job_id": "lease-job-1",
        "kind": "lease",
        "payload": {
            "evaluate_algorithms": [
                {"id": 101, "algorithm_dsl": "def Predict():\n  s0 = dot(v0, v1)\n", "input_dim": 16},
                {"id": 102, "algorithm_dsl": "def Predict():\n  s0 = dot(v0, v1)\n", "input_dim": 16},
            ],
            "seed_algorithms": [
                {"id": 201, "algorithm_dsl": "def Predict():\n  s0 = dot(v0, v1)\n", "input_dim": 16}
            ],
            "evolve_budget": 2,
            "input_dim": 16,
        },
    }
    fake_client = _FakeSidecarClient([job], stop=stop)
    monkeypatch.setattr(
        pool_miner_sidecar,
        "_SidecarClient",
        lambda *args, **kwargs: fake_client,
    )

    pool_miner_sidecar._run_worker(
        sidecar_url="http://127.0.0.1:8123",
        run_id="run-cpp-lease",
        worker_id="0",
        poll_interval_s=0.01,
        lease_seconds=120.0,
        lease_submit_buffer_s=45.0,
        lease_evolve_reserve_s=90.0,
        mode="real",
        evolve_generations=2,
        lease_evolve_generations=4,
        seed=123,
        stop=stop,
    )

    assert fake_client.results, "expected worker to submit a lease result"
    result = fake_client.results[0]
    assert result["ok"] is True
    payload = result["result"]
    assert len(payload.get("evaluations") or []) == 2
    assert len(payload.get("evolutions") or []) >= 1
    assert int(payload.get("iterations") or 0) >= 3


def test_run_worker_lease_real_evolves_from_seed_when_no_evals(monkeypatch) -> None:
    _install_real_mode_stubs(monkeypatch)
    monkeypatch.setattr(pool_miner_sidecar.time, "sleep", lambda _: None)

    stop = threading.Event()
    job = {
        "job_id": "lease-job-2",
        "kind": "lease",
        "payload": {
            "evaluate_algorithms": [],
            "seed_algorithms": [
                {"id": 301, "algorithm_dsl": "def Predict():\n  s0 = dot(v0, v1)\n", "input_dim": 16}
            ],
            "evolve_budget": 1,
            "input_dim": 16,
        },
    }
    fake_client = _FakeSidecarClient([job], stop=stop)
    monkeypatch.setattr(
        pool_miner_sidecar,
        "_SidecarClient",
        lambda *args, **kwargs: fake_client,
    )

    pool_miner_sidecar._run_worker(
        sidecar_url="http://127.0.0.1:8123",
        run_id="run-cpp-lease-seed",
        worker_id="0",
        poll_interval_s=0.01,
        lease_seconds=120.0,
        lease_submit_buffer_s=45.0,
        lease_evolve_reserve_s=90.0,
        mode="real",
        evolve_generations=2,
        lease_evolve_generations=3,
        seed=456,
        stop=stop,
    )

    assert fake_client.results, "expected worker to submit a lease result"
    result = fake_client.results[0]
    assert result["ok"] is True
    payload = result["result"]
    assert payload.get("evaluations") == []
    evolutions = payload.get("evolutions") or []
    assert len(evolutions) >= 1
    assert evolutions[0].get("parent_algorithm_ids") == [301]


def test_run_worker_evaluate_and_evolve_jobs_real_mode(monkeypatch) -> None:
    _install_real_mode_stubs(monkeypatch)
    monkeypatch.setattr(pool_miner_sidecar.time, "sleep", lambda _: None)

    # Evaluate job
    stop_eval = threading.Event()
    evaluate_job = {
        "job_id": "eval-job-1",
        "kind": "evaluate",
        "payload": {
            "algorithms": [
                {"id": 1, "algorithm_dsl": "def Predict():\n  s0 = dot(v0, v1)\n", "input_dim": 16},
                {"id": 2, "algorithm_dsl": "def Predict():\n  s0 = dot(v0, v1)\n", "input_dim": 16},
            ]
        },
    }
    eval_client = _FakeSidecarClient([evaluate_job], stop=stop_eval)
    monkeypatch.setattr(pool_miner_sidecar, "_SidecarClient", lambda *args, **kwargs: eval_client)

    pool_miner_sidecar._run_worker(
        sidecar_url="http://127.0.0.1:8123",
        run_id="run-eval",
        worker_id="0",
        poll_interval_s=0.01,
        lease_seconds=120.0,
        lease_submit_buffer_s=45.0,
        lease_evolve_reserve_s=90.0,
        mode="real",
        evolve_generations=2,
        lease_evolve_generations=3,
        seed=1,
        stop=stop_eval,
    )
    assert eval_client.results and eval_client.results[0]["ok"] is True
    assert len(eval_client.results[0]["result"].get("evaluations") or []) == 2

    # Evolve job
    stop_evo = threading.Event()
    evolve_job = {
        "job_id": "evo-job-1",
        "kind": "evolve",
        "payload": {
            "input_dim": 16,
            "algorithms": [
                {"id": 7, "algorithm_dsl": "def Predict():\n  s0 = dot(v0, v1)\n", "input_dim": 16}
            ],
        },
    }
    evo_client = _FakeSidecarClient([evolve_job], stop=stop_evo)
    monkeypatch.setattr(pool_miner_sidecar, "_SidecarClient", lambda *args, **kwargs: evo_client)

    pool_miner_sidecar._run_worker(
        sidecar_url="http://127.0.0.1:8123",
        run_id="run-evo",
        worker_id="0",
        poll_interval_s=0.01,
        lease_seconds=120.0,
        lease_submit_buffer_s=45.0,
        lease_evolve_reserve_s=90.0,
        mode="real",
        evolve_generations=2,
        lease_evolve_generations=3,
        seed=2,
        stop=stop_evo,
    )
    assert evo_client.results and evo_client.results[0]["ok"] is True
    evolved_dsl = str(evo_client.results[0]["result"].get("evolved_function") or "")
    assert "def Predict()" in evolved_dsl


def test_run_worker_unknown_kind_returns_error(monkeypatch) -> None:
    monkeypatch.setattr(pool_miner_sidecar.time, "sleep", lambda _: None)

    stop = threading.Event()
    weird_job = {
        "job_id": "job-weird",
        "kind": "mystery",
        "payload": {},
    }
    fake_client = _FakeSidecarClient([weird_job], stop=stop)
    monkeypatch.setattr(pool_miner_sidecar, "_SidecarClient", lambda *args, **kwargs: fake_client)

    pool_miner_sidecar._run_worker(
        sidecar_url="http://127.0.0.1:8123",
        run_id="run-weird",
        worker_id="0",
        poll_interval_s=0.01,
        lease_seconds=120.0,
        lease_submit_buffer_s=45.0,
        lease_evolve_reserve_s=90.0,
        mode="real",
        evolve_generations=1,
        lease_evolve_generations=1,
        seed=3,
        stop=stop,
    )

    assert fake_client.results and fake_client.results[0]["ok"] is False
    assert "unknown kind" in str(fake_client.results[0]["error"])


def test_parse_args_and_main_keyboard_interrupt(monkeypatch) -> None:
    monkeypatch.setenv("BITSOTA_POOL_MINER_MODE", "invalid")
    parsed = pool_miner_sidecar.parse_args([])
    assert parsed.mode == "real"

    started: Dict[str, int] = {"count": 0}
    joined: Dict[str, int] = {"count": 0}

    class _FakeThread:
        def __init__(self, target: Any, kwargs: Dict[str, Any], daemon: bool) -> None:
            self.target = target
            self.kwargs = dict(kwargs)
            self.daemon = bool(daemon)

        def start(self) -> None:
            started["count"] += 1

        def join(self, timeout: float | None = None) -> None:
            joined["count"] += 1

    monkeypatch.setattr(pool_miner_sidecar.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        pool_miner_sidecar,
        "parse_args",
        lambda argv=None: types.SimpleNamespace(
            sidecar_url="http://127.0.0.1:8123",
            run_id="run-main",
            workers=2,
            poll_interval_s=0.1,
            lease_seconds=120.0,
            lease_submit_buffer_s=45.0,
            lease_evolve_reserve_s=90.0,
            mode="real",
            evolve_generations=2,
            lease_evolve_generations=3,
            seed=11,
        ),
    )
    monkeypatch.setattr(
        pool_miner_sidecar.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    rc = pool_miner_sidecar.main([])
    assert rc == 0
    assert started["count"] == 2
    assert joined["count"] == 2

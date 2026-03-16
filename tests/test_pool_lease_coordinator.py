from __future__ import annotations

from typing import Any, Dict, List, Optional

import gui.pool_task_driver as pool_task_driver
from gui.pool_task_driver import CppDslComplianceError, PoolLeaseAssignment, PoolLeaseCoordinator


class _PoolClientStub:
    def __init__(self, submit_outcomes: List[Any]) -> None:
        self._submit_outcomes = list(submit_outcomes)
        self.submit_calls: List[Dict[str, Any]] = []
        self.request_lease_calls = 0
        self.register_calls = 0

    def register(self) -> bool:
        self.register_calls += 1
        return True

    def submit_lease(self, **kwargs):
        self.submit_calls.append(dict(kwargs))
        outcome = self._submit_outcomes.pop(0) if self._submit_outcomes else True
        if isinstance(outcome, Exception):
            raise outcome
        return bool(outcome)

    def request_lease(self, **kwargs):
        self.request_lease_calls += 1
        return None


class _SidecarStub:
    def __init__(self, polls: List[List[Dict[str, Any]]]) -> None:
        self._polls = list(polls)
        self.noted: List[bool] = []

    def poll_results(self, *, limit: int = 50):
        if self._polls:
            return self._polls.pop(0)
        return []

    def note_submission(self, *, ok: bool, score: float = 0.0) -> None:
        self.noted.append(bool(ok))

    def enqueue(self, *, kind: str, payload: Dict[str, Any]) -> Optional[str]:
        return None


def _completed_lease_event(*, job_id: str, iterations: int) -> Dict[str, Any]:
    return {
        "job_id": str(job_id),
        "status": "completed",
        "kind": "lease",
        "payload": {},
        "result": {
            "evaluations": [{"algorithm_id": 1, "score": 0.5}],
            "evolutions": [],
            "iterations": int(iterations),
        },
    }


def test_coordinator_forwards_iteration_count_to_pool_submit():
    pool = _PoolClientStub(submit_outcomes=[True])
    sidecar = _SidecarStub(
        polls=[
            [_completed_lease_event(job_id="job-1", iterations=123)],
        ]
    )
    logs: List[str] = []
    coordinator = PoolLeaseCoordinator(
        pool_client=pool,
        sidecar_jobs=sidecar,
        log=logs.append,
        request_interval_s=9999.0,
    )
    coordinator._registered = True
    coordinator._active = ("job-1", "lease-1")

    coordinator.tick()

    assert pool.submit_calls, "expected submit_lease to be called"
    assert pool.submit_calls[0].get("iterations") == 123


def test_coordinator_does_not_request_new_lease_after_transient_submit_failure():
    pool = _PoolClientStub(submit_outcomes=[RuntimeError("network timeout")])
    sidecar = _SidecarStub(
        polls=[
            [_completed_lease_event(job_id="job-2", iterations=42)],
        ]
    )
    logs: List[str] = []
    coordinator = PoolLeaseCoordinator(
        pool_client=pool,
        sidecar_jobs=sidecar,
        log=logs.append,
        request_interval_s=0.1,
    )
    coordinator._registered = True
    coordinator._active = ("job-2", "lease-2")

    coordinator.tick()

    assert pool.submit_calls, "expected submit_lease attempt"
    assert pool.request_lease_calls == 0


def test_cpp_lease_bootstraps_when_all_algorithms_noncompliant(monkeypatch):
    class _BootstrapPoolClient:
        def __init__(self) -> None:
            self.register_calls = 0
            self.request_lease_calls = 0

        def register(self) -> bool:
            self.register_calls += 1
            return True

        def request_lease(self, **kwargs):
            self.request_lease_calls += 1
            return PoolLeaseAssignment(
                lease_id="lease-bootstrap",
                window_number=1,
                timeout_at_s=None,
                evolve_budget=3,
                evaluate_algorithms=[{"id": 1, "algorithm_dsl": "bad-eval"}],
                seed_algorithms=[{"id": 2, "algorithm_dsl": "bad-seed"}],
            )

    class _BootstrapSidecar:
        def __init__(self) -> None:
            self.enqueued: List[Dict[str, Any]] = []

        def poll_results(self, *, limit: int = 50):
            return []

        def note_submission(self, *, ok: bool, score: float = 0.0) -> None:
            return None

        def enqueue(self, *, kind: str, payload: Dict[str, Any]) -> Optional[str]:
            self.enqueued.append({"kind": str(kind), "payload": dict(payload)})
            return "job-bootstrap"

    def _always_raise(*args, **kwargs):
        raise CppDslComplianceError("bad-dsl")

    monkeypatch.setattr("gui.pool_task_driver.normalize_algorithm_record_for_cpp", _always_raise)

    pool = _BootstrapPoolClient()
    sidecar = _BootstrapSidecar()
    logs: List[str] = []
    coordinator = PoolLeaseCoordinator(
        pool_client=pool,
        sidecar_jobs=sidecar,
        log=logs.append,
        request_interval_s=0.0,
    )
    coordinator._registered = True
    coordinator._cpp_backend = True
    coordinator._cpp_task_type = "cifar10_binary"

    coordinator.tick()

    assert sidecar.enqueued, "expected lease job enqueue"
    payload = sidecar.enqueued[0]["payload"]
    assert sidecar.enqueued[0]["kind"] == "lease"
    assert payload.get("evaluate_algorithms") == []
    assert payload.get("seed_algorithms") == []
    assert int(payload.get("evolve_budget") or 0) == 3
    assert int(payload.get("input_dim") or 0) == 16
    assert int(payload.get("bootstrap_population_size") or 0) == 8
    assert coordinator._active == ("job-bootstrap", "lease-bootstrap")
    assert any("bootstrapping evolution from empty seed set" in line for line in logs)


def test_lease_window_conflict_applies_backoff_and_suppresses_spam(monkeypatch):
    class _WindowBusyPoolClient:
        def __init__(self) -> None:
            self.register_calls = 0
            self.request_lease_calls = 0

        def register(self) -> bool:
            self.register_calls += 1
            return True

        def request_lease(self, **kwargs):
            self.request_lease_calls += 1
            raise RuntimeError(
                "Pool /tasks/lease failed: HTTP 400 (Lease already issued for this window. Wait for next window.)"
            )

    pool = _WindowBusyPoolClient()
    sidecar = _SidecarStub(polls=[])
    logs: List[str] = []
    coordinator = PoolLeaseCoordinator(
        pool_client=pool,
        sidecar_jobs=sidecar,
        log=logs.append,
        request_interval_s=0.0,
    )
    coordinator._registered = True
    coordinator._lease_window_retry_backoff_s = 30.0

    now_s = {"value": 100.0}
    monkeypatch.setattr(pool_task_driver, "_now_s", lambda: float(now_s["value"]))

    coordinator.tick()
    assert pool.request_lease_calls == 1
    assert coordinator._request_blocked_until_s >= 130.0
    assert any("lease already issued for current window" in line for line in logs)

    # Still inside backoff window, so no second request_lease call.
    now_s["value"] = 110.0
    coordinator.tick()
    assert pool.request_lease_calls == 1

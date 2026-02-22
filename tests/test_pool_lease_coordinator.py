from __future__ import annotations

from typing import Any, Dict, List, Optional

from gui.pool_task_driver import PoolLeaseCoordinator


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

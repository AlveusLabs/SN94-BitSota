from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from core.cpp_dsl_compat import CppDslComplianceError, normalize_algorithm_dsl_for_cpp
from core.dsl_parser import DSLParser
from gui.pool_task_driver import PoolLeaseAssignment, PoolLeaseCoordinator


def test_normalize_algorithm_dsl_for_cpp_rejects_non_cpp_ops():
    dsl = """
# meta: scalar_count=20 vector_count=10 matrix_count=5 vector_dim=16
def Setup():
  s10 = 1.0
def Predict():
  s3 = s1 / s2
def Learn():
  s5 = tan(s1)
"""
    with pytest.raises(CppDslComplianceError):
        normalize_algorithm_dsl_for_cpp(
            dsl,
            input_dim=16,
            task_type="cifar10_binary",
        )


def test_normalize_algorithm_dsl_for_cpp_accepts_valid_cpp_subset():
    dsl = """
# meta: scalar_count=5 vector_count=9 matrix_count=2 vector_dim=16
def Setup():
  s1 = gaussian(0.0, 1.0)
def Predict():
  v2 = dot(m0, v0)
  v3 = maximum(v2, v1)
  s0 = dot(v3, v0)
def Learn():
  m1 = outer(v2, v3)
"""
    normalized = normalize_algorithm_dsl_for_cpp(dsl, input_dim=16, task_type="cifar10_binary")

    parsed = DSLParser.from_dsl(normalized, 16)
    assert parsed.scalar_count == 5
    assert parsed.vector_count == 9
    assert parsed.matrix_count == 2

    errors = parsed.validate_addresses() + parsed.validate_semantics()
    assert not errors


def test_dsl_parser_to_dsl_uses_cpp_phase_style_and_noop():
    dsl = """
# meta: scalar_count=5 vector_count=9 matrix_count=2 vector_dim=16
def Setup():
  NoOp()
def Predict():
  s0 = s1 + s2
def Learn():
  NoOp()
"""
    algo = DSLParser.from_dsl(dsl, 16)
    out = DSLParser.to_dsl(algo)

    assert "def Setup():" in out
    assert "def Predict():" in out
    assert "def Learn():" in out
    assert "NoOp()" in out


class _PoolClientLeaseStub:
    def __init__(self, assignment: Optional[PoolLeaseAssignment]) -> None:
        self._assignment = assignment
        self.request_calls = 0
        self.submit_calls: List[Dict[str, Any]] = []

    def register(self) -> bool:
        return True

    def request_lease(self, **kwargs):
        self.request_calls += 1
        assignment = self._assignment
        self._assignment = None
        return assignment

    def submit_lease(self, **kwargs):
        self.submit_calls.append(dict(kwargs))
        return True


class _SidecarCaptureStub:
    def __init__(self) -> None:
        self.enqueued: List[Dict[str, Any]] = []

    def poll_results(self, *, limit: int = 50):
        return []

    def enqueue(self, *, kind: str, payload: Dict[str, Any]) -> Optional[str]:
        self.enqueued.append({"kind": str(kind), "payload": dict(payload or {})})
        return "job-1"

    def note_submission(self, *, ok: bool, score: float = 0.0) -> None:
        return None


def test_pool_lease_coordinator_normalizes_payload_when_cpp_backend(monkeypatch):
    monkeypatch.setenv("BITSOTA_MINER_BACKEND", "cpp")
    assignment = PoolLeaseAssignment(
        lease_id="lease-1",
        window_number=1,
        timeout_at_s=None,
        evolve_budget=1,
        evaluate_algorithms=[
            {
                "id": 10,
                "input_dim": 16,
                "algorithm_dsl": "def Predict():\n  v2 = dot(m0, v0)\n",
            },
            {
                "id": 12,
                "input_dim": 16,
                "algorithm_dsl": "def Predict():\n  s9 = s8 / s7\n",
            }
        ],
        seed_algorithms=[
            {
                "id": 11,
                "input_dim": 16,
                "algorithm_dsl": "def Predict():\n  s8 = s7\n",
            }
        ],
    )
    pool = _PoolClientLeaseStub(assignment=assignment)
    sidecar = _SidecarCaptureStub()
    logs: List[str] = []
    coordinator = PoolLeaseCoordinator(
        pool_client=pool,
        sidecar_jobs=sidecar,
        log=logs.append,
        request_interval_s=0.0,
    )
    coordinator._registered = True

    coordinator.tick()

    assert sidecar.enqueued
    payload = sidecar.enqueued[0]["payload"]
    eval_algos = payload.get("evaluate_algorithms") or []
    seed_algos = payload.get("seed_algorithms") or []
    assert eval_algos
    assert len(eval_algos) == 1
    assert int(eval_algos[0].get("id")) == 10
    # COPY-only seed is non-compliant and should be dropped.
    assert seed_algos == []

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


def _now_s() -> float:
    return float(time.time())


def _wallet_sign_hex(wallet: Any, message: str) -> str:
    signer = getattr(getattr(wallet, "hotkey", None), "sign", None)
    if not callable(signer):
        raise RuntimeError("Wallet hotkey signer unavailable")
    try:
        return signer(message).hex()
    except Exception:
        return signer(message.encode("utf-8")).hex()


def _pool_auth_headers(wallet: Any) -> Dict[str, str]:
    ts = str(int(time.time()))
    msg = f"auth:{ts}"
    sig = _wallet_sign_hex(wallet, msg)
    hotkey = getattr(getattr(wallet, "hotkey", None), "ss58_address", None)
    if not hotkey:
        raise RuntimeError("Wallet hotkey ss58 address unavailable")
    return {"X-Key": str(hotkey), "X-Timestamp": ts, "X-Signature": sig}


@dataclass
class PoolTaskAssignment:
    batch_id: str
    task_type: str  # evolve|evaluate
    algorithms: List[Dict[str, Any]]

@dataclass
class PoolLeaseAssignment:
    lease_id: str
    window_number: int
    evolve_budget: int
    evaluate_algorithms: List[Dict[str, Any]]
    seed_algorithms: List[Dict[str, Any]]


class PoolApiClient:
    def __init__(self, base_url: str, wallet: Any, *, timeout_s: float = 5.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.wallet = wallet
        self.timeout_s = max(0.1, float(timeout_s))
        self._session = requests.Session()

    @staticmethod
    def _response_detail(r: Any) -> Any:
        detail = None
        try:
            payload = r.json() if getattr(r, "content", None) else {}
        except Exception:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message") or payload
        elif payload is not None:
            detail = payload
        else:
            detail = (getattr(r, "text", "") or "").strip() or None
        return detail

    def register(self) -> bool:
        r = self._session.post(
            f"{self.base_url}/api/v1/miners/register",
            headers=_pool_auth_headers(self.wallet),
            json={},
            timeout=self.timeout_s,
        )
        return r.status_code == 200

    def request_task(self, *, task_type: Optional[str] = None) -> Optional[PoolTaskAssignment]:
        body: Dict[str, Any] = {}
        if task_type is not None:
            body["task_type"] = str(task_type)

        r = self._session.post(
            f"{self.base_url}/api/v1/tasks/request",
            headers=_pool_auth_headers(self.wallet),
            json=body,
            timeout=self.timeout_s,
        )
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            detail = self._response_detail(r)
            raise RuntimeError(f"Pool /tasks/request failed: HTTP {r.status_code} ({detail})")
        payload = r.json() if r.content else {}
        if not isinstance(payload, dict):
            return None

        batch_id = payload.get("batch_id")
        kind = payload.get("task_type")
        algos = payload.get("algorithms") or []
        if not batch_id or not kind or not isinstance(algos, list) or not algos:
            return None

        return PoolTaskAssignment(batch_id=str(batch_id), task_type=str(kind), algorithms=list(algos))

    def request_lease(
        self,
        *,
        task_type: Optional[str] = None,
        eval_batch_size: Optional[int] = None,
        seed_batch_size: Optional[int] = None,
        gossip_limit: Optional[int] = None,
    ) -> Optional[PoolLeaseAssignment]:
        body: Dict[str, Any] = {}
        if task_type is not None:
            body["task_type"] = str(task_type)
        if eval_batch_size is not None:
            body["eval_batch_size"] = int(eval_batch_size)
        if seed_batch_size is not None:
            body["seed_batch_size"] = int(seed_batch_size)
        if gossip_limit is not None:
            body["gossip_limit"] = int(gossip_limit)

        r = self._session.post(
            f"{self.base_url}/api/v1/tasks/lease",
            headers=_pool_auth_headers(self.wallet),
            json=body,
            timeout=self.timeout_s,
        )
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            detail = self._response_detail(r)
            raise RuntimeError(f"Pool /tasks/lease failed: HTTP {r.status_code} ({detail})")

        payload = r.json() if r.content else {}
        if not isinstance(payload, dict):
            return None

        lease_id = payload.get("lease_id")
        window_number = payload.get("window_number")
        evolve_budget = payload.get("evolve_budget", 0)
        evaluate_algorithms = payload.get("evaluate_algorithms") or []
        seed_algorithms = payload.get("seed_algorithms") or []
        if not lease_id:
            return None
        if not isinstance(evaluate_algorithms, list) or not isinstance(seed_algorithms, list):
            return None
        try:
            window_value = int(window_number or 0)
        except Exception:
            window_value = 0
        try:
            budget_value = int(evolve_budget or 0)
        except Exception:
            budget_value = 0

        return PoolLeaseAssignment(
            lease_id=str(lease_id),
            window_number=window_value,
            evolve_budget=budget_value,
            evaluate_algorithms=list(evaluate_algorithms),
            seed_algorithms=list(seed_algorithms),
        )

    def submit_evolution(self, *, batch_id: str, evolved_function: str, parent_functions: List[Dict[str, Any]]) -> bool:
        r = self._session.post(
            f"{self.base_url}/api/v1/tasks/{batch_id}/submit_evolution",
            headers=_pool_auth_headers(self.wallet),
            json={"evolved_function": str(evolved_function), "parent_functions": list(parent_functions or [])},
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            detail = self._response_detail(r)
            raise RuntimeError(f"Pool /tasks/{batch_id}/submit_evolution failed: HTTP {r.status_code} ({detail})")
        return True

    def submit_evaluation(self, *, batch_id: str, evaluations: List[Dict[str, Any]]) -> bool:
        r = self._session.post(
            f"{self.base_url}/api/v1/tasks/{batch_id}/submit_evaluation",
            headers=_pool_auth_headers(self.wallet),
            json={"evaluations": list(evaluations or [])},
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            detail = self._response_detail(r)
            raise RuntimeError(f"Pool /tasks/{batch_id}/submit_evaluation failed: HTTP {r.status_code} ({detail})")
        return True

    def submit_lease(
        self,
        *,
        lease_id: str,
        evaluations: List[Dict[str, Any]],
        evolutions: List[Dict[str, Any]],
        gossip: Optional[Dict[str, Any]] = None,
    ) -> bool:
        body: Dict[str, Any] = {"evaluations": list(evaluations or []), "evolutions": list(evolutions or [])}
        if gossip is not None:
            body["gossip"] = dict(gossip)
        r = self._session.post(
            f"{self.base_url}/api/v1/tasks/{lease_id}/submit_lease",
            headers=_pool_auth_headers(self.wallet),
            json=body,
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            detail = self._response_detail(r)
            raise RuntimeError(f"Pool /tasks/{lease_id}/submit_lease failed: HTTP {r.status_code} ({detail})")
        return True


class SidecarJobClient:
    def __init__(self, sidecar_url: str, run_id: str, *, timeout_s: float = 2.0) -> None:
        self.sidecar_url = str(sidecar_url).rstrip("/")
        self.run_id = str(run_id)
        self.timeout_s = max(0.1, float(timeout_s))
        self._session = requests.Session()
        self._results_cursor = 0

    def enqueue(self, *, kind: str, payload: Dict[str, Any]) -> Optional[str]:
        r = self._session.post(
            f"{self.sidecar_url}/jobs/enqueue",
            json={"run_id": self.run_id, "kind": str(kind), "payload": dict(payload or {})},
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            return None
        try:
            data = r.json() or {}
        except Exception:
            return None
        job_id = data.get("job_id")
        return str(job_id) if job_id else None

    def poll_results(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        try:
            r = self._session.get(
                f"{self.sidecar_url}/jobs/results",
                params={"run_id": self.run_id, "cursor": int(self._results_cursor), "limit": limit},
                timeout=self.timeout_s,
            )
            payload = r.json() or {}
        except Exception:
            return []
        items = payload.get("items") or []
        cursor = payload.get("cursor")
        if cursor is not None:
            try:
                self._results_cursor = int(cursor)
            except Exception:
                pass
        return list(items) if isinstance(items, list) else []

    def note_submission(self, *, ok: bool, score: float = 0.0) -> None:
        status = "submitted" if ok else "failed"
        try:
            self._session.post(
                f"{self.sidecar_url}/submission_result",
                json={"run_id": self.run_id, "score": float(score), "status": status},
                timeout=self.timeout_s,
            )
        except Exception:
            return


class PoolTaskCoordinator:
    def __init__(
        self,
        *,
        pool_client: PoolApiClient,
        sidecar_jobs: SidecarJobClient,
        log: Callable[[str], None],
        request_interval_s: float = 1.0,
    ) -> None:
        self.pool_client = pool_client
        self.sidecar_jobs = sidecar_jobs
        self.log = log
        self.request_interval_s = max(0.2, float(request_interval_s))

        self._registered = False
        self._active: Optional[Tuple[str, str, str]] = None  # (job_id, batch_id, kind)
        self._last_request_s = 0.0
        self._evolve_streak = 0
        self._last_pool_error: Optional[str] = None
        self._last_pool_error_s: float = 0.0

    def _log_pool_error(self, message: str) -> None:
        now = _now_s()
        msg = str(message)
        if self._last_pool_error == msg and (now - float(self._last_pool_error_s)) < 5.0:
            return
        self._last_pool_error = msg
        self._last_pool_error_s = now
        self.log(msg)

    def tick(self) -> None:
        if not self._registered:
            try:
                self._registered = bool(self.pool_client.register())
            except Exception:
                self._registered = False
            if self._registered:
                self.log("[pool] Registered with pool")

        for ev in self.sidecar_jobs.poll_results(limit=50):
            if not isinstance(ev, dict):
                continue
            job_id = ev.get("job_id")
            status = ev.get("status")
            kind = ev.get("kind")
            payload = ev.get("payload") or {}
            result = ev.get("result") or {}
            if not job_id or not isinstance(payload, dict):
                continue

            active = self._active
            if not active or str(active[0]) != str(job_id):
                continue

            batch_id = str(active[1])
            if status != "completed":
                self.log(f"[pool] Job failed kind={kind} status={status}")
                self._active = None
                self._last_request_s = 0.0
                continue

            ok = False
            if str(active[2]) == "evolve":
                evolved = result.get("evolved_function")
                parents = payload.get("algorithms") or []
                parent_functions = [{"id": p.get("id")} for p in parents if isinstance(p, dict) and p.get("id") is not None]
                if evolved:
                    try:
                        ok = bool(
                            self.pool_client.submit_evolution(
                                batch_id=batch_id,
                                evolved_function=str(evolved),
                                parent_functions=parent_functions,
                            )
                        )
                    except Exception as e:
                        ok = False
                        self._log_pool_error(f"[pool] submit_evolution failed batch_id={batch_id}: {e}")
                self.log(f"[pool] submit_evolution ok={ok} batch_id={batch_id}")

            elif str(active[2]) == "evaluate":
                evaluations = result.get("evaluations") or []
                try:
                    ok = bool(self.pool_client.submit_evaluation(batch_id=batch_id, evaluations=list(evaluations or [])))
                except Exception as e:
                    ok = False
                    self._log_pool_error(f"[pool] submit_evaluation failed batch_id={batch_id}: {e}")
                self.log(f"[pool] submit_evaluation ok={ok} batch_id={batch_id} n={len(evaluations) if isinstance(evaluations, list) else 0}")

            self.sidecar_jobs.note_submission(ok=ok, score=0.0)
            self._active = None
            self._last_request_s = 0.0

        if self._active is not None:
            return
        if not self._registered:
            return
        if (_now_s() - float(self._last_request_s)) < self.request_interval_s:
            return

        self._last_request_s = _now_s()
        assignment = None
        preferred = "evaluate" if int(self._evolve_streak) >= 3 else "evolve"
        try:
            assignment = self.pool_client.request_task(task_type=preferred)
        except Exception as e:
            self._log_pool_error(f"[pool] request_task failed task_type={preferred}: {e}")
            assignment = None

        if assignment is None:
            fallback = "evolve" if preferred == "evaluate" else "evaluate"
            try:
                assignment = self.pool_client.request_task(task_type=fallback)
            except Exception as e:
                self._log_pool_error(f"[pool] request_task failed task_type={fallback}: {e}")
                assignment = None
        if assignment is None:
            return

        kind = str(assignment.task_type)
        if kind not in {"evolve", "evaluate"}:
            return

        payload: Dict[str, Any] = {"batch_id": assignment.batch_id, "algorithms": assignment.algorithms}
        if isinstance(assignment.algorithms, list) and assignment.algorithms:
            first = assignment.algorithms[0]
            if isinstance(first, dict) and first.get("input_dim") is not None:
                payload["input_dim"] = first.get("input_dim")

        job_id = self.sidecar_jobs.enqueue(kind=kind, payload=payload)
        if not job_id:
            self.log(f"[pool] Failed to enqueue sidecar job for kind={kind}")
            return

        self._active = (job_id, assignment.batch_id, kind)
        if kind == "evolve":
            self._evolve_streak = int(self._evolve_streak) + 1
        else:
            self._evolve_streak = 0
        self.log(f"[pool] Enqueued job kind={kind} batch_id={assignment.batch_id} job_id={job_id}")


class PoolLeaseCoordinator:
    def __init__(
        self,
        *,
        pool_client: PoolApiClient,
        sidecar_jobs: SidecarJobClient,
        log: Callable[[str], None],
        request_interval_s: float = 1.0,
    ) -> None:
        self.pool_client = pool_client
        self.sidecar_jobs = sidecar_jobs
        self.log = log
        self.request_interval_s = max(0.2, float(request_interval_s))

        self._registered = False
        self._active: Optional[Tuple[str, str]] = None  # (job_id, lease_id)
        self._last_request_s = 0.0
        self._last_pool_error: Optional[str] = None
        self._last_pool_error_s: float = 0.0

    def _log_pool_error(self, message: str) -> None:
        now = _now_s()
        msg = str(message)
        if self._last_pool_error == msg and (now - float(self._last_pool_error_s)) < 5.0:
            return
        self._last_pool_error = msg
        self._last_pool_error_s = now
        self.log(msg)

    def tick(self) -> None:
        if not self._registered:
            try:
                self._registered = bool(self.pool_client.register())
            except Exception:
                self._registered = False
            if self._registered:
                self.log("[pool] Registered with pool")

        for ev in self.sidecar_jobs.poll_results(limit=50):
            if not isinstance(ev, dict):
                continue
            job_id = ev.get("job_id")
            status = ev.get("status")
            payload = ev.get("payload") or {}
            result = ev.get("result") or {}
            if not job_id or not isinstance(payload, dict):
                continue

            active = self._active
            if not active or str(active[0]) != str(job_id):
                continue

            lease_id = str(active[1])
            if status != "completed":
                self.log(f"[pool] Job failed kind=lease status={status}")
                self._active = None
                self._last_request_s = 0.0
                continue

            evaluations = result.get("evaluations") or []
            evolutions = result.get("evolutions") or []
            try:
                eval_n = len(evaluations) if isinstance(evaluations, list) else 0
            except Exception:
                eval_n = 0
            try:
                evo_n = len(evolutions) if isinstance(evolutions, list) else 0
            except Exception:
                evo_n = 0

            ok = False
            try:
                ok = bool(
                    self.pool_client.submit_lease(
                        lease_id=lease_id,
                        evaluations=list(evaluations or []) if isinstance(evaluations, list) else [],
                        evolutions=list(evolutions or []) if isinstance(evolutions, list) else [],
                        gossip=None,
                    )
                )
            except Exception as e:
                ok = False
                self._log_pool_error(f"[pool] submit_lease failed lease_id={lease_id}: {e}")
            self.log(f"[pool] submit_lease ok={ok} lease_id={lease_id} eval_n={eval_n} evo_n={evo_n}")

            self.sidecar_jobs.note_submission(ok=ok, score=0.0)
            self._active = None
            self._last_request_s = 0.0

        if self._active is not None:
            return
        if not self._registered:
            return
        if (_now_s() - float(self._last_request_s)) < self.request_interval_s:
            return

        self._last_request_s = _now_s()
        assignment = None
        try:
            assignment = self.pool_client.request_lease(gossip_limit=0)
        except Exception as e:
            self._log_pool_error(f"[pool] request_lease failed: {e}")
            assignment = None
        if assignment is None:
            return

        payload: Dict[str, Any] = {
            "lease_id": assignment.lease_id,
            "evaluate_algorithms": assignment.evaluate_algorithms,
            "seed_algorithms": assignment.seed_algorithms,
            "evolve_budget": int(assignment.evolve_budget),
        }
        for src in (assignment.seed_algorithms, assignment.evaluate_algorithms):
            if not src or not isinstance(src, list):
                continue
            first = src[0]
            if isinstance(first, dict) and first.get("input_dim") is not None:
                payload["input_dim"] = first.get("input_dim")
                break

        job_id = self.sidecar_jobs.enqueue(kind="lease", payload=payload)
        if not job_id:
            self.log("[pool] Failed to enqueue sidecar lease job")
            return

        self._active = (job_id, assignment.lease_id)
        self.log(f"[pool] Enqueued job kind=lease lease_id={assignment.lease_id} job_id={job_id}")

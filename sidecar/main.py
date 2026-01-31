from __future__ import annotations

from collections import deque
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field


def _now_s() -> float:
    return float(time.time())

def _as_non_empty_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


@dataclass
class _RunState:
    run_id: str
    created_at_s: float = field(default_factory=_now_s)
    updated_at_s: float = field(default_factory=_now_s)

    status: str = "idle"  # idle | running | stopped
    global_sota: Optional[float] = None
    local_sota: Optional[float] = None

    successful_submissions: int = 0

    # Rolling stores for GUI polling (cursor is a monotonic sequence number).
    logs: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    populations: List[Dict[str, Any]] = field(default_factory=list)
    job_results: List[Dict[str, Any]] = field(default_factory=list)

    # Monotonic cursors for rollover-safe polling.
    next_log_seq: int = 0
    next_candidate_seq: int = 0
    next_population_seq: int = 0
    next_job_result_seq: int = 0

    # Derived metrics from progress events.
    worker_last_iter: Dict[int, int] = field(default_factory=dict)
    worker_last_rate: Dict[int, Optional[float]] = field(default_factory=dict)

    # Pool-style job queue.
    jobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    job_queue: "deque[str]" = field(default_factory=deque)

    def to_state(self) -> Dict[str, Any]:
        tasks_completed = int(sum(int(v) for v in self.worker_last_iter.values()))

        best_candidate = None
        for c in self.candidates:
            try:
                score = float(c.get("validator_score"))
            except Exception:
                continue
            if best_candidate is None or score > float(best_candidate):
                best_candidate = score

        return {
            "run_id": self.run_id,
            "status": self.status,
            "global_sota": self.global_sota,
            "local_sota": self.local_sota,
            "successful_submissions": int(self.successful_submissions),
            "tasks_completed": tasks_completed,
            "best_candidate_score": best_candidate,
            "queued_jobs": int(sum(1 for j in self.jobs.values() if j.get("status") == "queued")),
            "leased_jobs": int(sum(1 for j in self.jobs.values() if j.get("status") == "leased")),
            "completed_jobs": int(sum(1 for j in self.jobs.values() if j.get("status") in {"completed", "failed"})),
            "updated_at_s": float(self.updated_at_s),
        }


class StartRunRequest(BaseModel):
    run_id: Optional[str] = None
    replace: bool = True


class StartRunResponse(BaseModel):
    run_id: str


class SetGlobalSotaRequest(BaseModel):
    run_id: Optional[str] = None
    value: float = Field(..., description="Latest global SOTA from relay")


class SubmissionResultRequest(BaseModel):
    run_id: Optional[str] = None
    score: float
    status: str = Field(..., description="GUI submission status, e.g. submitted/not_submitted/failed")

class CursorResponse(BaseModel):
    cursor: int
    items: List[Dict[str, Any]]

class EnqueueJobRequest(BaseModel):
    run_id: Optional[str] = None
    job_id: Optional[str] = None
    kind: str = Field(..., description="Job kind, e.g. evolve/evaluate")
    payload: Dict[str, Any] = Field(default_factory=dict)

class EnqueueJobResponse(BaseModel):
    run_id: str
    job_id: str
    status: str

class JobLeaseResponse(BaseModel):
    run_id: str
    job: Optional[Dict[str, Any]] = None

class JobResultRequest(BaseModel):
    run_id: Optional[str] = None
    status: str = Field(..., description="ok|error")
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


app = FastAPI(title="BitSota Local Sidecar", version="0.1.0")

_lock = threading.Lock()
_runs: Dict[str, _RunState] = {}
_current_run_id: Optional[str] = None

_MAX_LOG_ITEMS = 5000
_MAX_CANDIDATES = 2000
_MAX_POPULATIONS = 200
_MAX_JOBS = 2000
_MAX_JOB_RESULTS = 2000


def _get_or_create_run(run_id: str) -> _RunState:
    global _current_run_id
    run = _runs.get(run_id)
    if run is None:
        run = _RunState(run_id=str(run_id))
        _runs[run_id] = run
    _current_run_id = str(run_id)
    return run


def _current_run() -> Optional[_RunState]:
    if _current_run_id is None:
        return None
    return _runs.get(_current_run_id)

def _get_run_for_read(run_id: Optional[str]) -> Optional[_RunState]:
    rid = _as_non_empty_str(run_id) or _current_run_id
    if not rid:
        return None
    return _runs.get(rid)

def _append_seq(items: List[Dict[str, Any]], next_seq: int, item: Dict[str, Any], *, max_items: int) -> int:
    item = dict(item)
    item["seq"] = int(next_seq)
    items.append(item)
    if len(items) > int(max_items):
        items[:] = items[-int(max_items) :]
    return int(next_seq) + 1

def _cursor_slice(
    items: List[Dict[str, Any]],
    cursor: int,
    limit: int,
) -> tuple[int, List[Dict[str, Any]]]:
    if not items:
        return int(cursor), []

    base_seq = items[0].get("seq")
    if base_seq is None:
        # Backwards compatibility for runs created before seq cursors existed.
        chunk = items[cursor : cursor + limit]
        return int(cursor) + len(chunk), chunk

    try:
        base_seq_i = int(base_seq)
    except Exception:
        base_seq_i = 0

    idx = int(cursor) - int(base_seq_i)
    if idx < 0:
        idx = 0
    if idx > len(items):
        idx = len(items)

    chunk = items[idx : idx + limit]
    if not chunk:
        # Cursor beyond current window: park at end-of-window (i.e. next seq).
        return int(base_seq_i) + int(idx), []

    last_seq = chunk[-1].get("seq")
    try:
        next_cursor = int(last_seq) + 1
    except Exception:
        next_cursor = int(base_seq_i) + int(idx) + len(chunk)
    return int(next_cursor), chunk

def _sweep_expired_job_leases(run: _RunState) -> None:
    now = _now_s()
    for job in list(run.jobs.values()):
        if job.get("status") != "leased":
            continue
        lease_expires = job.get("lease_expires_at_s")
        try:
            expired = lease_expires is not None and float(lease_expires) <= now
        except Exception:
            expired = True
        if not expired:
            continue

        job["status"] = "queued"
        job["worker_id"] = None
        job["lease_expires_at_s"] = None
        job["updated_at_s"] = now
        run.job_queue.appendleft(str(job["job_id"]))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "ts": _now_s()}


@app.post("/runs/start", response_model=StartRunResponse)
def runs_start(req: StartRunRequest) -> StartRunResponse:
    global _current_run_id
    run_id = (req.run_id or str(uuid4())).strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id must be non-empty")

    with _lock:
        if (not req.replace) and (run_id in _runs):
            _current_run_id = run_id
            _runs[run_id].status = "running"
            _runs[run_id].updated_at_s = _now_s()
            return StartRunResponse(run_id=run_id)

        run = _RunState(run_id=run_id)
        run.status = "running"
        _runs[run_id] = run
        _current_run_id = run_id
        return StartRunResponse(run_id=run_id)


@app.get("/state")
def state(run_id: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        run = _get_run_for_read(run_id)
        if run is None:
            return {
                "run_id": _as_non_empty_str(run_id),
                "status": "idle",
                "global_sota": None,
                "local_sota": None,
                "successful_submissions": 0,
                "tasks_completed": 0,
                "best_candidate_score": None,
                "updated_at_s": _now_s(),
            }
        return run.to_state()


@app.post("/set_global_sota")
def set_global_sota(req: SetGlobalSotaRequest) -> Dict[str, Any]:
    with _lock:
        run_id = _as_non_empty_str(req.run_id)
        run = _runs.get(run_id) if run_id else _current_run()
        if run is None:
            run = _get_or_create_run(str(uuid4()))
            run.status = "running"
        run.global_sota = float(req.value)
        run.updated_at_s = _now_s()
        return {"status": "ok"}


@app.post("/submission_result")
def submission_result(req: SubmissionResultRequest) -> Dict[str, Any]:
    with _lock:
        run_id = _as_non_empty_str(req.run_id)
        run = _runs.get(run_id) if run_id else _current_run()
        if run is None:
            raise HTTPException(status_code=400, detail="No active run (provide run_id or start a run)")
        score = float(req.score)
        status = str(req.status or "")
        if status in {"submitted", "accepted", "success"}:
            run.successful_submissions = int(run.successful_submissions) + 1
            if run.local_sota is None or score > float(run.local_sota):
                run.local_sota = float(score)
        run.updated_at_s = _now_s()
        run.next_log_seq = _append_seq(
            run.logs,
            run.next_log_seq,
            {
                "type": "submission_result",
                "ts": _now_s(),
                "score": float(score),
                "status": status,
            },
            max_items=_MAX_LOG_ITEMS,
        )
        return {"status": "ok"}


@app.post("/ingest_batch")
async def ingest_batch(request: Request) -> Dict[str, Any]:
    """
    Hot path: miners post batches frequently. We intentionally avoid pydantic models here
    to minimize validation overhead and reduce impact on mining throughput.
    """

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id must be provided")

    events = payload.get("events") or []
    if not isinstance(events, list):
        raise HTTPException(status_code=400, detail="events must be a list")

    with _lock:
        run = _get_or_create_run(run_id)
        run.status = "running"

        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_type = str(ev.get("type") or "log")
            server_ts = _now_s()

            if ev_type == "progress":
                worker_id = ev.get("worker_id")
                iteration = ev.get("iteration")
                try:
                    wid = int(worker_id)
                except Exception:
                    wid = 0
                try:
                    it = int(iteration)
                except Exception:
                    it = 0
                run.worker_last_iter[wid] = max(it, int(run.worker_last_iter.get(wid, 0) or 0))
                try:
                    run.worker_last_rate[wid] = (
                        float(ev.get("rate")) if ev.get("rate") is not None else None
                    )
                except Exception:
                    run.worker_last_rate[wid] = None

                # Keep a readable log line for the GUI.
                msg_parts = [f"[w{wid}] Iter {it:04d}:"]
                if ev.get("engine_best") is not None:
                    try:
                        msg_parts.append(f" engine_best={float(ev.get('engine_best')):.4f}")
                    except Exception:
                        msg_parts.append(f" engine_best={ev.get('engine_best')}")
                if ev.get("pop_best") is not None:
                    try:
                        msg_parts.append(f" pop_best={float(ev.get('pop_best')):.4f}")
                    except Exception:
                        msg_parts.append(f" pop_best={ev.get('pop_best')}")
                msg_parts.append(f" rate={ev.get('rate')}")
                run.next_log_seq = _append_seq(
                    run.logs,
                    run.next_log_seq,
                    {
                        "type": "log",
                        "ts": server_ts,
                        "message": "".join(msg_parts),
                    },
                    max_items=_MAX_LOG_ITEMS,
                )
                continue

            if ev_type == "candidate":
                run.next_candidate_seq = _append_seq(
                    run.candidates,
                    run.next_candidate_seq,
                    {**ev, "ts": server_ts},
                    max_items=_MAX_CANDIDATES,
                )
                # Also log it.
                try:
                    wid = int(ev.get("worker_id", 0))
                except Exception:
                    wid = 0
                try:
                    it = int(ev.get("iteration", 0))
                except Exception:
                    it = 0
                try:
                    vs = float(ev.get("validator_score", 0.0))
                except Exception:
                    vs = 0.0
                run.next_log_seq = _append_seq(
                    run.logs,
                    run.next_log_seq,
                    {
                        "type": "log",
                        "ts": server_ts,
                        "message": f"[candidate] w{wid} iter {it}: verified_score={vs:.6f}",
                    },
                    max_items=_MAX_LOG_ITEMS,
                )
                continue

            if ev_type == "population":
                run.next_population_seq = _append_seq(
                    run.populations,
                    run.next_population_seq,
                    {**ev, "ts": server_ts},
                    max_items=_MAX_POPULATIONS,
                )

                try:
                    wid = int(ev.get("worker_id", 0))
                except Exception:
                    wid = 0
                try:
                    it = int(ev.get("iteration", 0))
                except Exception:
                    it = 0
                pop = ev.get("population")
                pop_len = len(pop) if isinstance(pop, list) else 0
                run.next_log_seq = _append_seq(
                    run.logs,
                    run.next_log_seq,
                    {
                        "type": "log",
                        "ts": server_ts,
                        "message": f"[population] w{wid} iter {it}: pop_size={pop_len}",
                    },
                    max_items=_MAX_LOG_ITEMS,
                )
                continue

            if ev_type == "done":
                wid = ev.get("worker_id")
                if wid is None:
                    run.next_log_seq = _append_seq(
                        run.logs,
                        run.next_log_seq,
                        {"type": "log", "ts": server_ts, "message": "[coord] done"},
                        max_items=_MAX_LOG_ITEMS,
                    )
                    run.status = "stopped"
                else:
                    run.next_log_seq = _append_seq(
                        run.logs,
                        run.next_log_seq,
                        {
                            "type": "log",
                            "ts": server_ts,
                            "message": f"[w{wid}] done",
                        },
                        max_items=_MAX_LOG_ITEMS,
                    )
                continue

            if ev_type == "error":
                run.next_log_seq = _append_seq(
                    run.logs,
                    run.next_log_seq,
                    {
                        "type": "log",
                        "ts": server_ts,
                        "message": f"[error] {ev.get('traceback') or ev.get('message') or ev}",
                    },
                    max_items=_MAX_LOG_ITEMS,
                )
                continue

            # Default: store as a log-ish event
            msg = ev.get("message")
            run.next_log_seq = _append_seq(
                run.logs,
                run.next_log_seq,
                {"type": "log", "ts": server_ts, "message": str(msg or ev)},
                max_items=_MAX_LOG_ITEMS,
            )

        run.updated_at_s = _now_s()
        return {"status": "ok", "received": len(events)}


@app.get("/logs", response_model=CursorResponse)
def get_logs(cursor: int = 0, limit: int = 200, run_id: Optional[str] = None) -> CursorResponse:
    cursor = max(0, int(cursor))
    limit = max(1, min(int(limit), 500))
    with _lock:
        run = _get_run_for_read(run_id)
        if run is None:
            return CursorResponse(cursor=cursor, items=[])
        next_cursor, chunk = _cursor_slice(run.logs, cursor, limit)
        return CursorResponse(cursor=next_cursor, items=chunk)


@app.get("/candidates", response_model=CursorResponse)
def get_candidates(cursor: int = 0, limit: int = 200, run_id: Optional[str] = None) -> CursorResponse:
    cursor = max(0, int(cursor))
    limit = max(1, min(int(limit), 500))
    with _lock:
        run = _get_run_for_read(run_id)
        if run is None:
            return CursorResponse(cursor=cursor, items=[])
        next_cursor, chunk = _cursor_slice(run.candidates, cursor, limit)
        return CursorResponse(cursor=next_cursor, items=chunk)


@app.get("/populations", response_model=CursorResponse)
def get_populations(cursor: int = 0, limit: int = 50, run_id: Optional[str] = None) -> CursorResponse:
    cursor = max(0, int(cursor))
    limit = max(1, min(int(limit), 50))
    with _lock:
        run = _get_run_for_read(run_id)
        if run is None:
            return CursorResponse(cursor=cursor, items=[])
        next_cursor, chunk = _cursor_slice(run.populations, cursor, limit)
        return CursorResponse(cursor=next_cursor, items=chunk)


@app.post("/jobs/enqueue", response_model=EnqueueJobResponse)
def enqueue_job(req: EnqueueJobRequest) -> EnqueueJobResponse:
    run_id = _as_non_empty_str(req.run_id) or _current_run_id
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id must be provided (or start a run)")
    kind = _as_non_empty_str(req.kind)
    if not kind:
        raise HTTPException(status_code=400, detail="kind must be non-empty")

    job_id = _as_non_empty_str(req.job_id) or str(uuid4())

    with _lock:
        run = _get_or_create_run(run_id)
        existing = run.jobs.get(job_id)
        if existing is not None:
            raise HTTPException(status_code=409, detail="job_id already exists")
        job = {
            "job_id": job_id,
            "run_id": run_id,
            "kind": kind,
            "payload": dict(req.payload or {}),
            "status": "queued",
            "attempt": 0,
            "worker_id": None,
            "lease_expires_at_s": None,
            "created_at_s": _now_s(),
            "updated_at_s": _now_s(),
        }
        run.jobs[job_id] = job
        run.job_queue.append(job_id)

        if len(run.jobs) > _MAX_JOBS:
            # Best-effort: drop oldest completed jobs to keep memory bounded.
            candidates = [
                j for j in run.jobs.values() if j.get("status") in {"completed", "failed", "canceled"}
            ]
            candidates.sort(key=lambda j: float(j.get("updated_at_s") or 0.0))
            for victim in candidates[: max(0, len(run.jobs) - _MAX_JOBS)]:
                try:
                    del run.jobs[str(victim["job_id"])]
                except Exception:
                    pass

        run.updated_at_s = _now_s()
        return EnqueueJobResponse(run_id=run_id, job_id=job_id, status="queued")


@app.get("/jobs/next")
def lease_next_job(
    run_id: Optional[str] = None,
    worker_id: Optional[str] = None,
    lease_seconds: float = 60.0,
) -> Any:
    run_id = _as_non_empty_str(run_id) or _current_run_id
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id must be provided (or start a run)")
    worker_id = _as_non_empty_str(worker_id) or "worker"

    lease_seconds = max(1.0, float(lease_seconds or 60.0))

    with _lock:
        run = _runs.get(run_id)
        if run is None:
            return Response(status_code=204)
        _sweep_expired_job_leases(run)

        while run.job_queue:
            job_id = run.job_queue.popleft()
            job = run.jobs.get(job_id)
            if not isinstance(job, dict):
                continue
            if job.get("status") != "queued":
                continue
            job["status"] = "leased"
            job["attempt"] = int(job.get("attempt", 0) or 0) + 1
            job["worker_id"] = worker_id
            job["lease_expires_at_s"] = _now_s() + lease_seconds
            job["updated_at_s"] = _now_s()
            run.updated_at_s = _now_s()
            return {"run_id": run_id, "job": dict(job)}

        return Response(status_code=204)


@app.post("/jobs/{job_id}/result")
def submit_job_result(job_id: str, req: JobResultRequest) -> Dict[str, Any]:
    job_id = _as_non_empty_str(job_id)
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id must be non-empty")
    run_id = _as_non_empty_str(req.run_id) or _current_run_id
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id must be provided (or start a run)")
    status = _as_non_empty_str(req.status)
    if status not in {"ok", "error"}:
        raise HTTPException(status_code=400, detail="status must be ok|error")

    with _lock:
        run = _runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        job = run.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        final_status = "completed" if status == "ok" else "failed"
        job["status"] = final_status
        job["lease_expires_at_s"] = None
        job["updated_at_s"] = _now_s()

        result_event = {
            "job_id": job_id,
            "run_id": run_id,
            "kind": job.get("kind"),
            "status": final_status,
            "attempt": job.get("attempt"),
            "worker_id": job.get("worker_id"),
            "payload": job.get("payload"),
            "result": dict(req.result or {}),
            "error": req.error,
            "ts": _now_s(),
        }
        run.next_job_result_seq = _append_seq(
            run.job_results,
            run.next_job_result_seq,
            result_event,
            max_items=_MAX_JOB_RESULTS,
        )
        run.next_log_seq = _append_seq(
            run.logs,
            run.next_log_seq,
            {
                "type": "log",
                "ts": _now_s(),
                "message": f"[job] {job.get('kind')} {job_id} -> {final_status}",
            },
            max_items=_MAX_LOG_ITEMS,
        )
        run.updated_at_s = _now_s()
        return {"status": "ok"}


@app.get("/jobs/results", response_model=CursorResponse)
def get_job_results(cursor: int = 0, limit: int = 200, run_id: Optional[str] = None) -> CursorResponse:
    cursor = max(0, int(cursor))
    limit = max(1, min(int(limit), 500))
    with _lock:
        run = _get_run_for_read(run_id)
        if run is None:
            return CursorResponse(cursor=cursor, items=[])
        next_cursor, chunk = _cursor_slice(run.job_results, cursor, limit)
        return CursorResponse(cursor=next_cursor, items=chunk)

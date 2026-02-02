#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue as queue_module
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import requests

import scripts.miner_local_og as miner_local_og


def _default_sidecar_url() -> str:
    host = os.getenv("BITSOTA_SIDECAR_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.getenv("BITSOTA_SIDECAR_PORT", "8123").strip() or "8123"
    return f"http://{host}:{port}"


class _SidecarSender:
    def __init__(
        self,
        *,
        sidecar_url: str,
        run_id: str,
        batch_size: int = 50,
        flush_seconds: float = 0.25,
        max_queue: int = 5000,
        request_timeout_s: float = 2.0,
    ) -> None:
        self.sidecar_url = str(sidecar_url).rstrip("/")
        self.run_id = str(run_id)
        self.batch_size = max(1, int(batch_size))
        self.flush_seconds = max(0.01, float(flush_seconds))
        self.request_timeout_s = max(0.1, float(request_timeout_s))
        self._queue: queue_module.Queue[Dict[str, Any]] = queue_module.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._session = requests.Session()
        self.dropped = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, flush_timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.0, float(flush_timeout_s)))

    def send(self, ev: Dict[str, Any]) -> None:
        if not isinstance(ev, dict):
            return
        try:
            self._queue.put_nowait(ev)
        except queue_module.Full:
            self.dropped += 1

    def _post_batch(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return
        try:
            self._session.post(
                f"{self.sidecar_url}/ingest_batch",
                json={"run_id": self.run_id, "events": batch},
                timeout=self.request_timeout_s,
            )
        except Exception:
            # Sidecar connectivity must never block mining. Drop on failure.
            return

    def _loop(self) -> None:
        batch: List[Dict[str, Any]] = []
        next_flush = time.time() + self.flush_seconds
        while True:
            timeout = max(0.0, next_flush - time.time())
            try:
                ev = self._queue.get(timeout=timeout)
            except queue_module.Empty:
                ev = None

            if ev is not None:
                batch.append(ev)
                if len(batch) >= self.batch_size:
                    self._post_batch(batch)
                    batch.clear()
                    next_flush = time.time() + self.flush_seconds

            if time.time() >= next_flush:
                if batch:
                    self._post_batch(batch)
                    batch.clear()
                next_flush = time.time() + self.flush_seconds

            if self._stop.is_set():
                # Best-effort flush.
                try:
                    while not self._queue.empty() and len(batch) < self.batch_size:
                        batch.append(self._queue.get_nowait())
                except Exception:
                    pass
                if batch:
                    self._post_batch(batch)
                return


def _parse_sidecar_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sidecar-url", default=os.getenv("BITSOTA_SIDECAR_URL", _default_sidecar_url()))
    parser.add_argument("--run-id", default=os.getenv("BITSOTA_SIDECAR_RUN_ID"))
    parser.add_argument("--sidecar-batch-size", type=int, default=int(os.getenv("BITSOTA_SIDECAR_BATCH_SIZE", "50")))
    parser.add_argument(
        "--sidecar-flush-seconds",
        type=float,
        default=float(os.getenv("BITSOTA_SIDECAR_FLUSH_SECONDS", "0.25")),
    )
    parser.add_argument(
        "--sidecar-timeout-seconds",
        type=float,
        default=float(os.getenv("BITSOTA_SIDECAR_TIMEOUT_SECONDS", "2.0")),
    )
    ns, rest = parser.parse_known_args(argv)
    return ns, rest


def _run_single_worker(args: argparse.Namespace, sender: _SidecarSender, *, task_type: str) -> None:
    stop_event = threading.Event()

    class _OutQueue:
        def put(self, msg: Dict[str, Any]) -> None:
            if isinstance(msg, dict) and msg.get("type") in {"candidate", "population"}:
                msg = dict(msg)
                msg.setdefault("task_type", task_type)
            sender.send(msg)

    out_q = _OutQueue()
    in_q: queue_module.Queue[Any] = queue_module.Queue()
    try:
        miner_local_og._worker_loop(  # type: ignore[attr-defined]
            args,
            worker_id=0,
            out_queue=out_q,
            in_queue=in_q,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        stop_event.set()
        sender.send({"type": "done", "worker_id": 0, "iteration": -1})


def _run_meta(args: argparse.Namespace, sender: _SidecarSender, *, task_type: str) -> None:
    workers = max(1, int(args.workers))
    miner_local_og._warn_about_blas_threads(workers)  # type: ignore[attr-defined]
    miner_local_og._seed_coordinator_rng(getattr(args, "seed", None))  # type: ignore[attr-defined]

    ctx = miner_local_og.mp.get_context("spawn")  # type: ignore[attr-defined]
    out_queue = ctx.Queue()
    stop_event = ctx.Event()

    processes: Dict[int, miner_local_og.mp.Process] = {}  # type: ignore[attr-defined]
    in_queues: Dict[int, Any] = {}
    for worker_id in range(workers):
        in_queue = ctx.Queue()
        in_queues[worker_id] = in_queue
        proc = ctx.Process(
            target=miner_local_og._worker_entrypoint,  # type: ignore[attr-defined]
            args=(args, worker_id, out_queue, in_queue, stop_event),
            name=f"miner-worker-{worker_id}",
        )
        proc.start()
        processes[worker_id] = proc

    done_workers = set()
    migration_iteration = None
    pending_migrations: Dict[int, List[object]] = {}
    pending_migration_counts: Dict[int, int] = {}

    try:
        while len(done_workers) < workers:
            try:
                msg = out_queue.get(timeout=0.5)
            except queue_module.Empty:
                # Detect early worker exits.
                for worker_id, proc in processes.items():
                    if worker_id in done_workers:
                        continue
                    if proc.exitcode is None:
                        continue
                    done_workers.add(worker_id)
                    if proc.exitcode != 0:
                        stop_event.set()
                        sender.send({"type": "error", "worker_id": worker_id, "message": f"worker exited (exitcode={proc.exitcode})"})
                continue

            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type")
            if msg_type == "candidate":
                enriched = dict(msg)
                enriched.setdefault("task_type", task_type)
                sender.send(enriched)
                continue

            if msg_type == "population":
                enriched = dict(msg)
                enriched.setdefault("task_type", task_type)
                sender.send(enriched)
                continue

            if msg_type in {"progress", "error", "done"}:
                sender.send(msg)
                if msg_type == "done":
                    try:
                        done_workers.add(int(msg.get("worker_id")))
                    except Exception:
                        pass
                if msg_type == "error":
                    stop_event.set()
                continue

            if msg_type == "migration_request":
                worker_id = int(msg["worker_id"])
                iteration = int(msg["iteration"])
                migrants = list(msg.get("migrants") or [])
                if migration_iteration is None:
                    migration_iteration = iteration
                if iteration != int(migration_iteration):
                    stop_event.set()
                    sender.send(
                        {
                            "type": "error",
                            "worker_id": worker_id,
                            "message": f"migration desync: got iter {iteration} expected {migration_iteration}",
                        }
                    )
                    continue

                pending_migrations[worker_id] = migrants
                pending_migration_counts[worker_id] = len(migrants)
                if len(pending_migrations) >= workers:
                    pool: List[object] = []
                    for wid in range(workers):
                        pool.extend(pending_migrations.get(wid, []))
                    random.shuffle(pool)
                    cursor = 0
                    for target_worker in range(workers):
                        k = int(pending_migration_counts.get(target_worker, 0) or 0)
                        incoming = pool[cursor : cursor + k]
                        cursor += k
                        q = in_queues.get(target_worker)
                        if q is not None:
                            try:
                                q.put(
                                    {
                                        "type": "migration_response",
                                        "iteration": int(migration_iteration),
                                        "incoming": incoming,
                                    }
                                )
                            except Exception:
                                pass
                    pending_migrations.clear()
                    pending_migration_counts.clear()
                    migration_iteration = None
                continue

            # Unknown: forward as log event
            sender.send({"type": "log", "message": str(msg)})
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for proc in processes.values():
            try:
                proc.join(timeout=10)
            except Exception:
                pass
        for proc in processes.values():
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        for proc in processes.values():
            try:
                proc.join(timeout=5)
            except Exception:
                pass
        sender.send({"type": "done"})


def main() -> None:
    sidecar_ns, rest = _parse_sidecar_args(sys.argv[1:])

    # Parse the miner args using the original miner_local_og parser/config rules.
    original_argv0 = sys.argv[0]
    sys.argv = [original_argv0] + list(rest)
    args = miner_local_og.parse_args()

    # Sidecar miner runs until stopped; treat iterations<=0 as "infinite".
    try:
        if int(getattr(args, "iterations", 0) or 0) <= 0:
            setattr(args, "iterations", 2**31 - 1)
    except Exception:
        pass

    run_id = (sidecar_ns.run_id or str(uuid4())).strip()
    sidecar_url = str(sidecar_ns.sidecar_url).rstrip("/")

    task_type = str(getattr(args, "task_type", miner_local_og.DEFAULT_TASK_TYPE))

    sender = _SidecarSender(
        sidecar_url=sidecar_url,
        run_id=run_id,
        batch_size=int(sidecar_ns.sidecar_batch_size),
        flush_seconds=float(sidecar_ns.sidecar_flush_seconds),
        request_timeout_s=float(sidecar_ns.sidecar_timeout_seconds),
    )
    sender.start()
    sender.send({"type": "start", "run_id": run_id, "task_type": task_type, "ts": time.time()})

    try:
        workers = max(1, int(getattr(args, "workers", 1)))
        if workers == 1:
            _run_single_worker(args, sender, task_type=task_type)
        else:
            _run_meta(args, sender, task_type=task_type)
    finally:
        sender.send({"type": "shutdown", "dropped": int(sender.dropped)})
        sender.stop(flush_timeout_s=2.0)


if __name__ == "__main__":
    main()

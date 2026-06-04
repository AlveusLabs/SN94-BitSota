from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from miner.local_benchmark import run_local_benchmark_worker
from miner.island_model import MigrationCoordinator, set_blas_thread_env


class MultiProcessDirectMiningTask(QRunnable):
    class Signals(QObject):
        log = Signal(str)
        error = Signal(str)
        finished = Signal()
        stopping = Signal()
        stats_updated = Signal(dict)
        best_candidate = Signal(dict)

    def __init__(
        self,
        *,
        worker_config: Dict[str, Any],
        workers: int,
        seed: Optional[int],
        migration_generations: int,
        initial_tasks: int = 0,
        initial_submissions: int = 0,
        initial_best_score: Optional[float] = None,
    ):
        super().__init__()
        self.signals = self.Signals()
        self.setAutoDelete(True)

        self.worker_config = dict(worker_config)
        self.workers = max(1, int(workers))
        self.seed = seed
        self.migration_generations = max(0, int(migration_generations))

        self._stop_requested = False
        self._stop_event: Optional[mp.synchronize.Event] = None

        self.tasks_completed = int(initial_tasks)
        self.successful_submissions = int(initial_submissions)
        self.best_score = initial_best_score
        self._last_generation_by_worker: Dict[int, int] = {}

    def stop(self):
        self._stop_requested = True
        if self._stop_event is not None:
            try:
                self._stop_event.set()
            except Exception:
                pass
        self.signals.stopping.emit()

    @Slot()
    def run(self):
        ctx = mp.get_context("spawn")
        set_blas_thread_env(self.workers)

        out_queue = ctx.Queue()
        stop_event = ctx.Event()
        self._stop_event = stop_event
        if self._stop_requested:
            stop_event.set()

        migration_seed = None
        if self.seed is not None:
            migration_seed = int(self.seed) + 1000003
        coordinator = MigrationCoordinator(workers=self.workers, seed=migration_seed)

        processes: Dict[int, mp.Process] = {}
        in_queues: Dict[int, mp.Queue] = {}
        for worker_id in range(self.workers):
            in_q = ctx.Queue()
            in_queues[worker_id] = in_q

            cfg = dict(self.worker_config)
            cfg["seed"] = self.seed
            cfg["migration_generations"] = self.migration_generations

            proc = ctx.Process(
                target=run_local_benchmark_worker,
                args=(cfg, worker_id, out_queue, in_q, stop_event),
                name=f"gui-miner-worker-{worker_id}",
            )
            proc.start()
            processes[worker_id] = proc

        done_workers = set()

        try:
            while True:
                if stop_event.is_set() and len(done_workers) >= self.workers:
                    break

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
                            self.signals.error.emit(
                                f"Worker {worker_id} exited early (exitcode={proc.exitcode})"
                            )
                    continue

                msg_type = msg.get("type")

                if msg_type == "init":
                    worker_id = int(msg.get("worker_id", -1))
                    generation = msg.get("generation")
                    try:
                        generation_i = int(generation) if generation is not None else None
                    except Exception:
                        generation_i = None
                    if generation_i is not None and generation_i >= 0:
                        self._last_generation_by_worker[worker_id] = generation_i

                    best_verified = msg.get("best_verified_score")
                    if best_verified is not None:
                        try:
                            best_verified_f = float(best_verified)
                        except Exception:
                            best_verified_f = None
                        if best_verified_f is not None and (
                            self.best_score is None or best_verified_f > float(self.best_score)
                        ):
                            self.best_score = best_verified_f

                    self.signals.stats_updated.emit(
                        {
                            "tasks_completed": int(self.tasks_completed),
                            "successful_submissions": int(self.successful_submissions),
                            "best_score": self.best_score,
                        }
                    )
                    continue

                if msg_type == "stats":
                    worker_id = int(msg.get("worker_id", -1))
                    generation = msg.get("generation")
                    try:
                        generation_i = int(generation) if generation is not None else None
                    except Exception:
                        generation_i = None

                    if generation_i is not None and generation_i >= 0:
                        if worker_id in self._last_generation_by_worker:
                            prev = int(self._last_generation_by_worker.get(worker_id, 0))
                            if generation_i > prev:
                                self.tasks_completed += int(generation_i - prev)
                        self._last_generation_by_worker[worker_id] = generation_i

                    best_verified = msg.get("best_verified_score")
                    if best_verified is not None:
                        try:
                            best_verified_f = float(best_verified)
                        except Exception:
                            best_verified_f = None
                        if best_verified_f is not None and (
                            self.best_score is None or best_verified_f > float(self.best_score)
                        ):
                            self.best_score = best_verified_f

                    log_line = msg.get("log")
                    if log_line:
                        self.signals.log.emit(f"[w{worker_id}] {str(log_line)}")

                    self.signals.stats_updated.emit(
                        {
                            "tasks_completed": int(self.tasks_completed),
                            "successful_submissions": int(self.successful_submissions),
                            "best_score": self.best_score,
                        }
                    )
                    continue

                if msg_type == "best_verified":
                    worker_id = int(msg.get("worker_id", -1))
                    verified = msg.get("verified_score")
                    try:
                        verified_f = float(verified) if verified is not None else None
                    except Exception:
                        verified_f = None

                    if verified_f is not None and (
                        self.best_score is None or verified_f > float(self.best_score)
                    ):
                        self.best_score = verified_f

                    log_line = msg.get("log")
                    if log_line:
                        self.signals.log.emit(f"[w{worker_id}] {str(log_line)}")

                    self.signals.stats_updated.emit(
                        {
                            "tasks_completed": int(self.tasks_completed),
                            "successful_submissions": int(self.successful_submissions),
                            "best_score": self.best_score,
                        }
                    )
                    self.signals.best_candidate.emit(dict(msg))
                    continue

                if msg_type == "log":
                    worker_id = int(msg.get("worker_id", -1))
                    line = str(msg.get("message") or "")
                    self.signals.log.emit(f"[w{worker_id}] {line}")
                    continue

                if msg_type == "migration_request":
                    worker_id = int(msg.get("worker_id", -1))
                    iteration = int(msg.get("iteration", -1))
                    migrants = list(msg.get("migrants") or [])
                    try:
                        repartition = coordinator.add_request(
                            worker_id=worker_id, iteration=iteration, migrants=migrants
                        )
                    except Exception as e:
                        stop_event.set()
                        self.signals.error.emit(f"Migration coordinator error: {e}")
                        continue
                    if repartition is not None:
                        for target_worker, incoming in repartition.items():
                            q = in_queues.get(int(target_worker))
                            if q is None:
                                continue
                            q.put(
                                {
                                    "type": "migration_response",
                                    "iteration": int(iteration),
                                    "incoming": incoming,
                                }
                            )
                    continue

                if msg_type == "error":
                    stop_event.set()
                    worker_id = msg.get("worker_id")
                    tb = msg.get("traceback") or msg
                    self.signals.error.emit(f"Worker {worker_id} crashed:\n{tb}")
                    if worker_id is not None:
                        done_workers.add(int(worker_id))
                    continue

                if msg_type == "done":
                    worker_id = msg.get("worker_id")
                    if worker_id is not None:
                        done_workers.add(int(worker_id))
                    continue

        finally:
            stop_event.set()
            for proc in processes.values():
                proc.join(timeout=5)
            for proc in processes.values():
                if proc.is_alive():
                    proc.terminate()
            for proc in processes.values():
                proc.join(timeout=5)

            self.signals.finished.emit()

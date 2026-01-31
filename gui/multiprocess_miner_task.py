from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
import random
import re
import time
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from miner.island_mining import run_direct_mining_worker
from miner.island_model import MigrationCoordinator, set_blas_thread_env


class MiningStatsAccumulator:
    def __init__(
        self,
        *,
        checkpoint_generations: int = 1,
        tasks_completed: int = 0,
        successful_submissions: int = 0,
        best_score: Optional[float] = None,
    ):
        self.tasks_completed = int(tasks_completed)
        self.successful_submissions = int(successful_submissions)
        self.best_score = best_score

        self._regularized_log_every = max(1, int(checkpoint_generations))
        self._last_generation_by_worker: Dict[int, int] = {}
        self._last_regularized_iter_by_worker: Dict[int, int] = {}

        number = r"([-+]?(?:\\d+\\.?\\d*|\\d*\\.?\\d+)(?:[eE][-+]?\\d+)?)"
        self._re_score_verified = re.compile(
            rf"\\bScore:\\s*{number}\\s*\\(verified\\)", re.IGNORECASE
        )
        self._re_verified_score = re.compile(
            rf"\\bverified_score\\b[^0-9\\-\\+]*{number}", re.IGNORECASE
        )
        self._re_gen_line = re.compile(r"^Gen\\s+(\\d+)\\b", re.IGNORECASE)
        self._re_regularized_iter = re.compile(r"\\biter=(\\d+)\\b", re.IGNORECASE)

    def _maybe_update_best_verified(self, verified_score: float) -> bool:
        try:
            verified_score_f = float(verified_score)
        except Exception:
            return False
        if self.best_score is None or verified_score_f > float(self.best_score):
            self.best_score = verified_score_f
            return True
        return False

    def process_log_line(self, msg: str, *, worker_id: Optional[int] = None) -> Optional[Dict[str, object]]:
        """Return updated stats dict (or None if unchanged)."""

        changed = False

        msg = str(msg or "")
        lowered = msg.lower()

        if (
            "Solution submitted to relay" in msg
            or ("SOTA submission #" in msg and "successful" in lowered)
            or ("submission" in lowered and "successful" in lowered)
        ):
            self.successful_submissions += 1
            changed = True

        wid = -1
        if worker_id is not None:
            try:
                wid = int(worker_id)
            except Exception:
                wid = -1

        if msg.startswith("Gen "):
            m_gen = self._re_gen_line.match(msg)
            if m_gen:
                try:
                    generation = int(m_gen.group(1))
                except Exception:
                    generation = None
                if generation is not None:
                    last_seen = int(self._last_generation_by_worker.get(wid, 0))
                    if generation > last_seen:
                        delta = int(generation) - last_seen
                        self._last_generation_by_worker[wid] = int(generation)
                        self.tasks_completed += int(delta)
                        changed = True
        elif msg.startswith("[regularized-evo]"):
            m_iter = self._re_regularized_iter.search(msg)
            if m_iter:
                try:
                    iteration = int(m_iter.group(1))
                except Exception:
                    iteration = None
                if iteration is not None and (
                    iteration == 1 or (iteration % int(self._regularized_log_every)) == 0
                ):
                    last_seen = int(self._last_regularized_iter_by_worker.get(wid, 0))
                    if iteration > last_seen:
                        delta = int(iteration) - last_seen
                        self._last_regularized_iter_by_worker[wid] = int(iteration)
                        self.tasks_completed += int(delta)
                        changed = True

        m = self._re_score_verified.search(msg)
        if m and self._maybe_update_best_verified(m.group(1)):
            changed = True

        m = self._re_verified_score.search(msg)
        if m and self._maybe_update_best_verified(m.group(1)):
            changed = True

        if not changed:
            return None
        return {
            "tasks_completed": int(self.tasks_completed),
            "successful_submissions": int(self.successful_submissions),
            "best_score": self.best_score,
        }


class MultiProcessDirectMiningTask(QRunnable):
    class Signals(QObject):
        log = Signal(str)
        error = Signal(str)
        finished = Signal()
        stopping = Signal()
        stats_updated = Signal(dict)

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
        try:
            self._regularized_log_every = max(
                1, int(self.worker_config.get("checkpoint_generations", 1) or 1)
            )
        except Exception:
            self._regularized_log_every = 1
        self._re_regularized_iter = re.compile(r"\\biter=(\\d+)\\b", re.IGNORECASE)

        self._stop_requested = False
        self._stop_event: Optional[mp.synchronize.Event] = None

        self._stats = MiningStatsAccumulator(
            checkpoint_generations=self._regularized_log_every,
            tasks_completed=initial_tasks,
            successful_submissions=initial_submissions,
            best_score=initial_best_score,
        )
        self.tasks_completed = int(initial_tasks)
        self.successful_submissions = int(initial_submissions)
        self.best_score = initial_best_score

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
                target=run_direct_mining_worker,
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

                if msg_type == "log":
                    worker_id = int(msg.get("worker_id", -1))
                    line = str(msg.get("message") or "")
                    updated = self._stats.process_log_line(line, worker_id=worker_id)
                    if updated is not None:
                        self.tasks_completed = int(updated.get("tasks_completed", self.tasks_completed))
                        self.successful_submissions = int(
                            updated.get("successful_submissions", self.successful_submissions)
                        )
                        best = updated.get("best_score")
                        if best is not None:
                            try:
                                self.best_score = float(best)
                            except Exception:
                                pass
                        self.signals.stats_updated.emit(updated)

                    suppress_log = False
                    if line.startswith("[regularized-evo]") and self._regularized_log_every > 1:
                        m = self._re_regularized_iter.search(line)
                        if m:
                            try:
                                iteration = int(m.group(1))
                            except Exception:
                                iteration = None
                            if (
                                iteration is not None
                                and iteration != 1
                                and (iteration % self._regularized_log_every) != 0
                            ):
                                suppress_log = True

                    if not suppress_log:
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

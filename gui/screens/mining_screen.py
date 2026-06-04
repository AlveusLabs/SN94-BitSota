from PySide6.QtCore import Qt, QTimer, QRunnable, QThreadPool, QObject, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QGridLayout,
    QTextEdit,
    QPushButton,
)
from PySide6.QtSvgWidgets import QSvgWidget
from typing import Optional
import logging
import re

from gui.components import PrimaryButton, SecondaryButton
from gui.components.modal import ConfirmationModal
from gui.components.invite_code_modal import InviteCodeModal
from gui.app_config import get_app_config
from gui.components.coming_soon_modal import ComingSoonModal
from gui.screens.pool_mining_screen import PoolMiningScreen
from gui.resource_path import resource_path
import requests
import time
import uuid

from core.engine_defaults import apply_cpp_defaults_to_engine_params


class GUILogHandler(logging.Handler):
    def __init__(self, log_signal, stats_signal, task):
        super().__init__()
        self.log_signal = log_signal
        self.stats_signal = stats_signal
        self.task = task
        number = r"([-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)"
        self._re_score_verified = re.compile(rf"\bScore:\s*{number}\s*\(verified\)", re.IGNORECASE)
        self._re_verified_score = re.compile(rf"\bverified_score\b[^0-9\-\+]*{number}", re.IGNORECASE)
        self._re_regularized_iter = re.compile(r"\biter=(\d+)\b", re.IGNORECASE)

    def _maybe_update_best_verified(self, verified_score: float):
        try:
            verified_score = float(verified_score)
        except Exception:
            return
        if self.task.best_score is None or verified_score > self.task.best_score:
            self.task.best_score = verified_score
            self.stats_signal.emit(
                {
                    "tasks_completed": self.task.tasks_completed,
                    "successful_submissions": self.task.successful_submissions,
                    "best_score": self.task.best_score,
                }
            )

    def emit(self, record):
        msg = self.format(record)

        suppress_log = False
        if msg.startswith("[regularized-evo]"):
            m = self._re_regularized_iter.search(msg)
            if m:
                try:
                    iteration = int(m.group(1))
                except Exception:
                    iteration = None
                try:
                    log_every = max(1, int(getattr(self.task, "checkpoint_generations", 1) or 1))
                except Exception:
                    log_every = 1
                if (
                    iteration is not None
                    and log_every > 1
                    and iteration != 1
                    and (iteration % log_every) != 0
                ):
                    suppress_log = True

        if not suppress_log:
            self.log_signal.emit(msg)

        if (
            "Solution submitted to relay" in msg
            or ("SOTA submission #" in msg and "successful" in msg.lower())
            or ("submission" in msg.lower() and "successful" in msg.lower())
        ):
            self.task.successful_submissions += 1
            best_verified = None
            try:
                if hasattr(self.task.client, "get_local_best_verified_score"):
                    best_verified = self.task.client.get_local_best_verified_score(self.task.task_type)
                elif hasattr(self.task.client, "_local_best_verified_score"):
                    best_verified = self.task.client._local_best_verified_score.get(self.task.task_type)  # type: ignore[attr-defined]
            except Exception:
                best_verified = None
            if best_verified is not None:
                self._maybe_update_best_verified(best_verified)
            else:
                self.stats_signal.emit(
                    {
                        "tasks_completed": self.task.tasks_completed,
                        "successful_submissions": self.task.successful_submissions,
                        "best_score": self.task.best_score,
                    }
                )

        if msg.startswith("[regularized-evo]") or msg.startswith("Gen ") or "generation" in msg.lower():
            self.task.tasks_completed += 1

        m = self._re_score_verified.search(msg)
        if m:
            self._maybe_update_best_verified(m.group(1))
            return

        m = self._re_verified_score.search(msg)
        if m:
            self._maybe_update_best_verified(m.group(1))
            return


class DirectMiningTask(QRunnable):
    class Signals(QObject):
        log = Signal(str)
        error = Signal(str)
        finished = Signal()
        stopping = Signal()
        stats_updated = Signal(dict)

    def __init__(
        self,
        client,
        task_type: str,
        stop_flag,
        *,
        engine_type: str = "baseline",
        checkpoint_generations: int = 10,
        initial_tasks=0,
        initial_submissions=0,
        initial_best_score=None,
    ):
        super().__init__()
        self.client = client
        self.task_type = task_type
        self.engine_type = str(engine_type or "baseline")
        self.checkpoint_generations = max(1, int(checkpoint_generations))
        self.stop_flag = stop_flag
        self.signals = self.Signals()
        self.setAutoDelete(True)
        self.tasks_completed = initial_tasks
        self.successful_submissions = initial_submissions
        self.best_score = initial_best_score

    def stop(self):
        self.stop_flag.stop()
        if hasattr(self.client, "stop_mining"):
            self.client.stop_mining()
        self.signals.stopping.emit()

    @Slot()
    def run(self):
        tracked_loggers = [
            logging.getLogger("miner"),
            logging.getLogger("core"),
        ]
        previous_levels = {tracked: tracked.level for tracked in tracked_loggers}
        handler = GUILogHandler(self.signals.log, self.signals.stats_updated, self)
        handler.setFormatter(logging.Formatter('%(message)s'))
        handler.setLevel(logging.INFO)
        for tracked in tracked_loggers:
            tracked.addHandler(handler)
            tracked.setLevel(logging.INFO)

        try:
            self.signals.log.emit(
                f"Starting {self.task_type} mining with engine={self.engine_type} checkpoint={self.checkpoint_generations}"
            )

            if hasattr(self.client, "run_continuous_mining"):
                result = self.client.run_continuous_mining(
                    task_type=self.task_type,
                    engine_type=self.engine_type,
                    checkpoint_generations=self.checkpoint_generations,
                )
                self.signals.log.emit(f"Mining session completed: {result}")
            else:
                self.signals.error.emit("Direct client not available")
                return

            if self.stop_flag.is_stopped():
                self.signals.log.emit("Mining stopped by user")
            else:
                self.signals.log.emit("Mining session completed")

        except Exception as e:
            self.signals.error.emit(f"Mining error: {e}")
            self.signals.log.emit(f"ERROR: Mining failed: {e}")
        finally:
            for tracked in tracked_loggers:
                tracked.removeHandler(handler)
                try:
                    tracked.setLevel(previous_levels[tracked])
                except Exception:
                    pass
            self.signals.finished.emit()


class RelaySubmissionTask(QRunnable):
    class Signals(QObject):
        log = Signal(str)
        error = Signal(str)
        result = Signal(dict)

    def __init__(self, client, *, solution_data: dict, prevalidated: dict):
        super().__init__()
        self.client = client
        self.solution_data = dict(solution_data or {})
        self.prevalidated = dict(prevalidated or {})
        self.signals = self.Signals()
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        try:
            if not self.client or not hasattr(self.client, "submit_solution"):
                self.signals.error.emit("Direct client not available for submission")
                return
            result = self.client.submit_solution(self.solution_data, prevalidated=self.prevalidated)
            self.signals.result.emit(dict(result or {}))
        except Exception as e:
            self.signals.error.emit(f"Submission error: {e}")


class MiningScreen(QWidget):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.is_mining = False
        self.mining_task: Optional[object] = None
        self.thread_pool = QThreadPool()
        self.tasks_completed = 0
        self.successful_submissions = 0
        self.best_score = None
        self._global_sota_value: Optional[float] = None
        self._submission_in_flight = False
        self._pending_sota_candidate: Optional[dict] = None
        self._inflight_sota_verified_score: Optional[float] = None
        self._last_submission_feedback: Optional[str] = None
        self._next_submission_time = 0.0
        self.setup_ui()
        self._load_mining_stats()

        self.sota_timer = QTimer()
        self.sota_timer.timeout.connect(self.update_global_sota)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(24)

        from gui.components.tab_switcher import TabSwitcher

        self.tab_switcher = TabSwitcher()
        self.tab_switcher.add_tab("direct", "Direct Mining")
        self.tab_switcher.add_tab("pool", "Pool Mining")
        self.tab_switcher.tab_changed.connect(self._on_mining_tab_changed)
        main_layout.addWidget(self.tab_switcher)

        self.description = QLabel(
            "Connect straight to Bittensor validators, ideal for users who want complete control over their mining operations."
        )
        self.description.setObjectName("mining_description")
        self.description.setWordWrap(True)
        main_layout.addWidget(self.description)

        self.content_stack = QWidget()
        self.content_stack_layout = QVBoxLayout(self.content_stack)
        self.content_stack_layout.setContentsMargins(0, 0, 0, 0)

        self.direct_mining_widget = QWidget()
        direct_layout = QVBoxLayout(self.direct_mining_widget)
        direct_layout.setContentsMargins(0, 0, 0, 0)

        content_box = QWidget()
        content_box.setObjectName("content_box")
        content_layout = QVBoxLayout(content_box)
        content_layout.setContentsMargins(24, 32, 24, 32)
        content_layout.setSpacing(24)

        config_section = self._create_config_section()
        content_layout.addWidget(config_section)

        stats_status_layout = QHBoxLayout()
        stats_status_layout.setSpacing(24)

        miner_stats = self._create_miner_stats()
        stats_status_layout.addWidget(miner_stats, 1)

        mining_status = self._create_mining_status()
        stats_status_layout.addWidget(mining_status, 1)

        content_layout.addLayout(stats_status_layout)

        logs_section = self._create_logs_section()
        content_layout.addWidget(logs_section)

        direct_layout.addWidget(content_box)

        self.pool_mining_widget = PoolMiningScreen(main_window=self.main_window)

        self.content_stack_layout.addWidget(self.direct_mining_widget)
        self.direct_mining_widget.show()
        self.pool_mining_widget.hide()

        main_layout.addWidget(self.content_stack)

    def _create_config_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("mining_config_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title = QLabel("Mining Configuration")
        title.setObjectName("section_title")
        layout.addWidget(title)

        task_label = QLabel("Task Type")
        task_label.setObjectName("form_label")
        layout.addWidget(task_label)

        config_row = QHBoxLayout()
        config_row.setSpacing(16)

        self.task_type_combo = QComboBox()
        self.task_type_combo.setObjectName("form_input")
        cfg = get_app_config()
        self.task_type_map = {"CIFAR-10 Binary": "cifar10_binary"}
        if getattr(cfg, "test_mode", False):
            self.task_type_map.update(
                {
                    "MNIST Binary": "mnist_binary",
                    "Scalar Linear": "scalar_linear",
                }
            )
        self.task_type_combo.addItems(list(self.task_type_map.keys()))
        self.task_type_combo.setEnabled(bool(getattr(cfg, "test_mode", False)))
        if not getattr(cfg, "test_mode", False):
            self.task_type_combo.setToolTip("Task selection is available in test mode only.")
        self.task_type_combo.currentTextChanged.connect(lambda: self.update_global_sota())
        config_row.addWidget(self.task_type_combo, 1)

        workers_label = QLabel("Workers")
        workers_label.setObjectName("form_label")
        config_row.addWidget(workers_label)

        self.workers_combo = QComboBox()
        self.workers_combo.setObjectName("form_input")
        self.workers_combo.setToolTip("Number of independent worker processes to run")
        worker_options = [1, 2, 4, 8]
        default_workers = getattr(cfg, "miner_workers", 1)
        try:
            default_workers = max(1, int(default_workers))
        except Exception:
            default_workers = 1
        if default_workers not in worker_options:
            worker_options.append(default_workers)
        worker_options = sorted(set(worker_options))
        self.workers_combo.addItems([str(v) for v in worker_options])
        self.workers_combo.setCurrentText(str(default_workers))
        self.workers_combo.setEnabled(True)
        config_row.addWidget(self.workers_combo)

        self.save_config_btn = SecondaryButton("Save Configuration", width=200, height=48)
        config_row.addWidget(self.save_config_btn)

        self.start_mining_btn = PrimaryButton("Start Mining", width=200, height=48, icon_path=resource_path("gui/images/play.svg"))
        self.start_mining_btn.clicked.connect(self._toggle_mining)
        config_row.addWidget(self.start_mining_btn)

        layout.addLayout(config_row)

        return section

    def _toggle_mining(self):
        if not self.is_mining:
            self._start_mining()
        else:
            self._stop_mining()

    def _start_mining(self):
        if self.is_mining and self.mining_task:
            self._append_log("ERROR: Mining task still running. Please wait for it to stop.")
            return

        if not self.main_window:
            self._append_log("ERROR: Main window reference not available.")
            return

        if not self.main_window.wallet:
            self._append_log("ERROR: No wallet loaded. Please load a wallet first.")
            return

        if not self.main_window.client:
            self._append_log("ERROR: Client not initialized. Please ensure wallet is properly loaded.")
            return

        if not self.main_window.coldkey_address:
            self._append_log("ERROR: No coldkey address provided. Please provide your coldkey address first.")
            self.main_window._prompt_for_coldkey_address()
            return

        try:
            relay_url = self.main_window._get_relay_endpoint_from_config()
            self._append_log(f"Relay endpoint: {relay_url}")
        except Exception:
            pass

        if not self._check_invite_code():
            self._show_invite_code_modal()
            return

        if not self._send_coldkey_address():
            self._append_log("ERROR: Failed to send coldkey address to relay. Please try again.")
            return

        self.is_mining = True
        self.start_mining_btn.update_icon("gui/images/stop.svg")
        self.start_mining_btn.update_text("Stop Mining")
        self.start_mining_btn.setObjectName("stop_mining_button")
        self.start_mining_btn.setStyleSheet("")
        self.start_mining_btn.style().unpolish(self.start_mining_btn)
        self.start_mining_btn.style().polish(self.start_mining_btn)

        task_display = self.task_type_combo.currentText()
        task_type = self.task_type_map.get(task_display, "cifar10_binary")
        cfg = get_app_config()
        problem_cfg = getattr(self.main_window, "problem_config", None)
        explicit_engine_params = getattr(problem_cfg, "engine_params", None) if problem_cfg is not None else None

        # Ensure defaults (memory + phase op limits) match the C++ reference unless
        # problem_config explicitly overrides them.
        try:
            client = self.main_window.client
            client.engine_params = apply_cpp_defaults_to_engine_params(
                task_type,
                getattr(client, "engine_params", None),
                explicit_engine_params=explicit_engine_params,
            ) or {}
            cache = getattr(client, "_engine_cache", None)
            if isinstance(cache, dict):
                cache.clear()
        except Exception:
            pass

        workers = 1
        if hasattr(self, "workers_combo") and self.workers_combo is not None:
            try:
                workers = max(1, int(self.workers_combo.currentText()))
            except Exception:
                workers = 1

        from gui.multiprocess_miner_task import MultiProcessDirectMiningTask

        miner_task_count = getattr(cfg, "miner_task_count", None)
        validator_task_count = getattr(cfg, "validator_task_count", None)
        validate_every = getattr(cfg, "miner_validate_every_n_generations", 1000)
        engine_type = "baseline"
        checkpoint_generations = 10
        engine_params = None
        env_overrides = None

        if problem_cfg is not None:
            miner_task_count = (
                problem_cfg.miner_task_count
                if problem_cfg.miner_task_count is not None
                else miner_task_count
            )
            validator_task_count = (
                problem_cfg.validator_task_count
                if problem_cfg.validator_task_count is not None
                else validator_task_count
            )
            validate_every = (
                problem_cfg.miner_validate_every_n_generations
                if problem_cfg.miner_validate_every_n_generations is not None
                else validate_every
            )
            if getattr(problem_cfg, "engine_type", None):
                engine_type = str(problem_cfg.engine_type)
            if getattr(problem_cfg, "checkpoint_generations", None):
                checkpoint_generations = int(problem_cfg.checkpoint_generations)
            engine_params = getattr(problem_cfg, "engine_params", None)
            env_overrides = getattr(problem_cfg, "env", None)

        engine_params = apply_cpp_defaults_to_engine_params(
            task_type,
            engine_params if isinstance(engine_params, dict) else None,
            explicit_engine_params=explicit_engine_params,
        )

        seed = getattr(cfg, "miner_seed", None)
        migration_generations = getattr(cfg, "miner_migration_generations", 0)

        sota_threshold = None
        try:
            sota_threshold = self.main_window.get_current_sota()
        except Exception:
            sota_threshold = None

        worker_config = {
            "public_address": getattr(self.main_window.wallet.hotkey, "ss58_address", None),
            "miner_task_count": miner_task_count,
            "validator_task_count": validator_task_count,
            "validate_every_n_generations": validate_every,
            "engine_params": engine_params,
            "env_overrides": env_overrides if isinstance(env_overrides, dict) else None,
            "task_type": task_type,
            "engine_type": engine_type,
            "checkpoint_generations": int(checkpoint_generations),
            "sota_threshold": sota_threshold,
            # Keep the engine itself quiet; the worker emits sparse stats + best events.
            "engine_verbose": False,
        }

        self.mining_task = MultiProcessDirectMiningTask(
            worker_config=worker_config,
            workers=workers,
            seed=seed,
            migration_generations=migration_generations,
            initial_tasks=self.tasks_completed,
            initial_submissions=self.successful_submissions,
            initial_best_score=self.best_score,
        )

        self.mining_task.signals.log.connect(self._append_log)
        self.mining_task.signals.error.connect(self._handle_mining_error)
        self.mining_task.signals.finished.connect(self._on_mining_finished)
        self.mining_task.signals.stats_updated.connect(self._update_stats)
        try:
            self.mining_task.signals.best_candidate.connect(self._on_best_candidate)
        except Exception:
            pass

        self.thread_pool.start(self.mining_task)
        self._append_log(f"Starting mining for task: {task_type}")
        self.update_connection_status(True)
        self.update_global_sota()
        self.sota_timer.start(30000)

    def _stop_mining(self):
        self.is_mining = False
        self.sota_timer.stop()
        self.start_mining_btn.update_icon(resource_path("gui/images/play.svg"))
        self.start_mining_btn.update_text("Start Mining")
        self.start_mining_btn.setObjectName("primary_button")
        self.start_mining_btn.setStyleSheet("")
        self.start_mining_btn.style().unpolish(self.start_mining_btn)
        self.start_mining_btn.style().polish(self.start_mining_btn)

        if self.mining_task:
            self.mining_task.stop()
            self._append_log("Stopping mining...")

    def _check_invite_code(self) -> bool:
        if get_app_config().test_mode:
            self._append_log("Test mode enabled: skipping invite code requirement.")
            return True
        try:
            relay_url = self.main_window._get_relay_endpoint_from_config()
            msg = f"auth:{int(time.time())}"
            sig = self.main_window.wallet.hotkey.sign(msg).hex()

            response = requests.get(
                f"{relay_url}/invitation_code/linked",
                headers={
                    "X-Key": self.main_window.wallet.hotkey.ss58_address,
                    "X-Signature": sig,
                    "X-Timestamp": msg
                },
                timeout=10
            )

            response.raise_for_status()
            result = response.json()
            return result.get("data") is not None
        except Exception as e:
            self._append_log(f"Failed to check invite code status: {e}")
            return False

    def _send_coldkey_address(self) -> bool:
        try:
            relay_url = self.main_window._get_relay_endpoint_from_config()
            msg = f"auth:{int(time.time())}"
            sig = self.main_window.wallet.hotkey.sign(msg).hex()

            response = requests.post(
                f"{relay_url}/coldkey_address/update",
                json={"coldkey_address": self.main_window.coldkey_address},
                headers={
                    "X-Key": self.main_window.wallet.hotkey.ss58_address,
                    "X-Signature": sig,
                    "X-Timestamp": msg
                },
                timeout=10
            )

            response.raise_for_status()
            result = response.json()
            if result.get("status") == "success":
                self._append_log(f"Coldkey address sent to relay successfully")
                return True
            else:
                self._append_log(f"Failed to send coldkey address: {result}")
                return False
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = f" Response: {e.response.text}"
            except Exception:
                pass
            self._append_log(f"Error sending coldkey address: {e}{detail}")
            return False
        except Exception as e:
            self._append_log(f"Error sending coldkey address: {e}")
            return False

    def _show_invite_code_modal(self):
        relay_url = self.main_window._get_relay_endpoint_from_config()
        coldkey_address = self.main_window.coldkey_address if hasattr(self.main_window, 'coldkey_address') else None
        invite_modal = InviteCodeModal(
            relay_url=relay_url,
            wallet=self.main_window.wallet,
            coldkey_address=coldkey_address,
            parent=self
        )
        invite_modal.code_verified.connect(self._on_invite_code_verified)
        invite_modal.exec()

    def _on_invite_code_verified(self):
        self._append_log("Invite code verified successfully!")
        self._start_mining()

    def _handle_mining_error(self, error_msg: str):
        self._append_log(f"ERROR: {error_msg}")

    def _load_mining_stats(self):
        from gui.wallet_utils_gui import load_mining_stats
        stats = load_mining_stats()
        self.tasks_completed = stats.get("tasks_completed", 0)
        self.successful_submissions = stats.get("successful_submissions", 0)
        self.best_score = stats.get("best_score")

        if hasattr(self, 'tasks_completed_label'):
            self.tasks_completed_label.setText(str(self.tasks_completed))
            self.successful_submissions_label.setText(str(self.successful_submissions))
            if self.best_score is not None:
                self.best_score_label.setText(f"{self.best_score:.4f}")
            else:
                self.best_score_label.setText("-")

    def _save_mining_stats(self):
        from gui.wallet_utils_gui import save_mining_stats
        save_mining_stats(self.tasks_completed, self.successful_submissions, self.best_score)

    def _update_stats(self, stats: dict):
        tasks = stats.get("tasks_completed", 0)
        submissions = stats.get("successful_submissions", 0)
        best_score = stats.get("best_score")

        self.tasks_completed = tasks
        self.successful_submissions = submissions
        if best_score is not None:
            self.best_score = best_score

        self.tasks_completed_label.setText(str(tasks))
        self.successful_submissions_label.setText(str(submissions))
        if best_score is not None:
            self.best_score_label.setText(f"{best_score:.4f}")
        else:
            self.best_score_label.setText("-")

        self._save_mining_stats()
        try:
            self._maybe_submit_pending_candidate()
        except Exception:
            pass

    def _maybe_submit_pending_candidate(self):
        if self._submission_in_flight:
            return
        if not self._pending_sota_candidate:
            return
        if time.time() < float(self._next_submission_time or 0.0):
            return
        if not self.main_window or not getattr(self.main_window, "client", None):
            return

        candidate = dict(self._pending_sota_candidate)
        try:
            verified_score = float(candidate.get("verified_score"))
        except Exception:
            self._pending_sota_candidate = None
            self._last_submission_feedback = None
            return

        sota_threshold = self._global_sota_value
        if sota_threshold is None:
            try:
                raw = candidate.get("sota_threshold")
                sota_threshold = float(raw) if raw is not None else None
            except Exception:
                sota_threshold = None
        if sota_threshold is None or verified_score < float(sota_threshold):
            # No longer a breaker (or we can't evaluate), drop it.
            self._pending_sota_candidate = None
            self._last_submission_feedback = None
            return

        algorithm_dsl = candidate.get("algorithm_dsl")
        if not algorithm_dsl:
            self._pending_sota_candidate = None
            self._last_submission_feedback = None
            return

        task_type = str(candidate.get("task_type") or "cifar10_binary")
        engine_type = str(candidate.get("engine_type") or "baseline")
        try:
            input_dim = int(candidate.get("input_dim") or 0)
        except Exception:
            input_dim = 0

        mining_score = candidate.get("mining_score")
        try:
            mining_score_f = float(mining_score) if mining_score is not None else 0.0
        except Exception:
            mining_score_f = 0.0

        solution_data = {
            "task_id": f"gui-sota-{uuid.uuid4()}",
            "task_type": task_type,
            "algorithm_dsl": str(algorithm_dsl),
            "eval_score": float(mining_score_f),
            "input_dim": int(input_dim),
            "metadata": {
                "generation": int(candidate.get("generation", -1)),
                "engine_type": engine_type,
                "source": "gui_offline_runner",
                "log_all_task_scores": True,
            },
        }

        prevalidated = {
            "verified_score": float(verified_score),
            "sota_threshold": float(sota_threshold),
        }

        attempt_msg = (
            f"Attempting SOTA submission: verified_score={float(verified_score):.4f} "
            f"sota_threshold={float(sota_threshold):.4f}"
        )
        if self._last_submission_feedback != attempt_msg:
            self._last_submission_feedback = attempt_msg
            self._append_log(attempt_msg)

        self._submission_in_flight = True
        self._inflight_sota_verified_score = float(verified_score)
        self._next_submission_time = 0.0

        submit_task = RelaySubmissionTask(
            self.main_window.client,
            solution_data=solution_data,
            prevalidated=prevalidated,
        )
        submit_task.signals.result.connect(self._on_submission_result)
        submit_task.signals.error.connect(self._on_submission_error)
        self.thread_pool.start(submit_task)

    def _on_best_candidate(self, candidate: dict):
        try:
            verified_score = candidate.get("verified_score")
            verified_score_f = float(verified_score) if verified_score is not None else None
        except Exception:
            verified_score_f = None
        if verified_score_f is None:
            return

        sota_threshold = self._global_sota_value
        if sota_threshold is None:
            try:
                raw = candidate.get("sota_threshold")
                sota_threshold = float(raw) if raw is not None else None
            except Exception:
                sota_threshold = None
        if sota_threshold is None:
            return

        if float(verified_score_f) < float(sota_threshold):
            return

        algo = candidate.get("algorithm_dsl")
        if not algo:
            return

        existing = self._pending_sota_candidate
        if existing is None:
            self._pending_sota_candidate = {
                **dict(candidate),
                "verified_score": float(verified_score_f),
                "sota_threshold": float(sota_threshold),
            }
        else:
            try:
                prev = float(existing.get("verified_score", -float("inf")))
            except Exception:
                prev = -float("inf")
            if float(verified_score_f) > float(prev):
                self._pending_sota_candidate = {
                    **dict(candidate),
                    "verified_score": float(verified_score_f),
                    "sota_threshold": float(sota_threshold),
                }
                self._last_submission_feedback = None

        self._maybe_submit_pending_candidate()

    def _on_submission_error(self, error_msg: str):
        self._submission_in_flight = False
        self._inflight_sota_verified_score = None
        self._next_submission_time = time.time() + 5.0
        self._append_log(f"ERROR: {error_msg}")

    def _on_submission_result(self, result: dict):
        self._submission_in_flight = False
        inflight_score = self._inflight_sota_verified_score
        self._inflight_sota_verified_score = None
        status = (result or {}).get("status")
        if status == "submitted":
            self.successful_submissions += 1
            if self.mining_task is not None and hasattr(self.mining_task, "successful_submissions"):
                try:
                    self.mining_task.successful_submissions = int(self.successful_submissions)
                except Exception:
                    pass
            verified = result.get("verified_score")
            if verified is not None:
                try:
                    self.best_score = max(float(self.best_score or -float("inf")), float(verified))
                except Exception:
                    pass
            self._append_log(f"Solution submitted to relay: {result}")
            self._next_submission_time = 0.0

            if self._pending_sota_candidate is not None and inflight_score is not None:
                try:
                    pending_score = float(self._pending_sota_candidate.get("verified_score", -float("inf")))
                except Exception:
                    pending_score = -float("inf")
                if pending_score <= float(inflight_score):
                    self._pending_sota_candidate = None
        else:
            reason = (result or {}).get("reason")
            if status == "not_submitted" and reason == "submission_cooldown":
                cooldown = result.get("cooldown_remaining_seconds")
                try:
                    cooldown_f = float(cooldown) if cooldown is not None else 5.0
                except Exception:
                    cooldown_f = 5.0
                self._next_submission_time = time.time() + max(0.5, cooldown_f)
                msg = f"Submission cooldown ({cooldown_f:.1f}s); will retry."
                if self._last_submission_feedback != msg:
                    self._last_submission_feedback = msg
                    self._append_log(msg)
            elif status == "not_submitted" and reason in {"below_sota_threshold", "below_local_best"}:
                self._pending_sota_candidate = None
                self._next_submission_time = 0.0
                self._append_log(f"Submission skipped ({reason}): {result}")
            else:
                self._next_submission_time = time.time() + 5.0
                self._append_log(f"Submission result: {result}")

        # Keep persisted GUI stats consistent with the mining runner.
        if self.mining_task is not None and hasattr(self.mining_task, "tasks_completed"):
            try:
                self.tasks_completed = int(self.mining_task.tasks_completed)
            except Exception:
                pass

        self._update_stats(
            {
                "tasks_completed": int(self.tasks_completed),
                "successful_submissions": int(self.successful_submissions),
                "best_score": self.best_score,
            }
        )

    def _on_mining_finished(self):
        self._submission_in_flight = False
        self._pending_sota_candidate = None
        self._inflight_sota_verified_score = None
        self._next_submission_time = 0.0
        if self.mining_task:
            final_stats = {
                "tasks_completed": self.mining_task.tasks_completed,
                "successful_submissions": self.mining_task.successful_submissions,
                "best_score": self.mining_task.best_score
            }
            self._update_stats(final_stats)
            self._save_mining_stats()
            self.mining_task = None

        self.is_mining = False
        self.sota_timer.stop()
        self.start_mining_btn.update_icon(resource_path("gui/images/play.svg"))
        self.start_mining_btn.update_text("Start Mining")
        self.start_mining_btn.setObjectName("primary_button")
        self.start_mining_btn.setStyleSheet("")
        self.start_mining_btn.style().unpolish(self.start_mining_btn)
        self.start_mining_btn.style().polish(self.start_mining_btn)
        self.update_connection_status(False)
        self._append_log("Mining stopped.")

    def _on_mining_tab_changed(self, tab_id: str):
        if tab_id == "pool":
            modal = ComingSoonModal(
                "Pool Mining Screen",
                "The Pool Mining screen is coming soon! This screen will allow you to join mining pools for simplified setup and shared resources. Pool mining is ideal for miners who want a streamlined experience with automated task distribution and reward payouts.",
                parent=self
            )
            modal.exec()
            self.tab_switcher.set_active_tab("direct")
        else:
            self._switch_to_direct()

    def _switch_to_pool(self):
        self.direct_mining_widget.hide()
        self.pool_mining_widget.show()
        self.content_stack_layout.addWidget(self.pool_mining_widget)
        self.description.setText(
            "Join a Mining Pool for simplified setup and shared resources. Ideal for beginners."
        )

    def _switch_to_direct(self):
        self.pool_mining_widget.hide()
        self.direct_mining_widget.show()
        self.description.setText(
            "Connect straight to Bittensor validators, ideal for users who want complete control over their mining operations."
        )

    def _create_miner_stats(self) -> QWidget:
        stats = QWidget()
        stats.setObjectName("stats_box")
        layout = QVBoxLayout(stats)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Miner Stats")
        title.setObjectName("section_title")
        layout.addWidget(title)

        stats_grid = QGridLayout()
        stats_grid.setSpacing(12)

        label = QLabel("Tasks Completed")
        label.setObjectName("stat_label")
        stats_grid.addWidget(label, 0, 0)
        self.tasks_completed_label = QLabel("0")
        self.tasks_completed_label.setObjectName("stat_value")
        self.tasks_completed_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        stats_grid.addWidget(self.tasks_completed_label, 0, 1)

        label = QLabel("Successful Submissions")
        label.setObjectName("stat_label")
        stats_grid.addWidget(label, 1, 0)
        self.successful_submissions_label = QLabel("0")
        self.successful_submissions_label.setObjectName("stat_value")
        self.successful_submissions_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        stats_grid.addWidget(self.successful_submissions_label, 1, 1)

        label = QLabel("Best Local Score")
        label.setObjectName("stat_label")
        stats_grid.addWidget(label, 2, 0)
        self.best_score_label = QLabel("-")
        self.best_score_label.setObjectName("stat_value")
        self.best_score_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        stats_grid.addWidget(self.best_score_label, 2, 1)

        layout.addLayout(stats_grid)
        layout.addStretch()

        return stats

    def _create_mining_status(self) -> QWidget:
        status = QWidget()
        status.setObjectName("stats_box")
        layout = QVBoxLayout(status)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Mining Status")
        title.setObjectName("section_title")
        layout.addWidget(title)

        status_grid = QGridLayout()
        status_grid.setSpacing(12)

        label = QLabel("Global SOTA")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 0, 0)
        self.global_sota_label = QLabel("-")
        self.global_sota_label.setObjectName("stat_value")
        self.global_sota_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.global_sota_label, 0, 1)

        label = QLabel("Wallet")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 1, 0)
        self.wallet_status_label = QLabel("Not Connected")
        self.wallet_status_label.setObjectName("stat_value")
        self.wallet_status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.wallet_status_label, 1, 1)

        label = QLabel("Connection")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 2, 0)
        self.connection_status_label = QLabel("Disconnected")
        self.connection_status_label.setObjectName("stat_value")
        self.connection_status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.connection_status_label, 2, 1)

        layout.addLayout(status_grid)
        layout.addStretch()

        return status

    def _create_logs_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title = QLabel("Mining Logs")
        title.setObjectName("section_title")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.setObjectName("clear_logs_button")
        self.clear_logs_btn.clicked.connect(self._clear_logs)
        header_layout.addWidget(self.clear_logs_btn)

        layout.addLayout(header_layout)

        self.logs_text = QTextEdit()
        self.logs_text.setObjectName("logs_text")
        self.logs_text.setReadOnly(True)
        self.logs_text.setMinimumHeight(200)
        layout.addWidget(self.logs_text)

        return section

    def _clear_logs(self):
        self.logs_text.clear()

    def _append_log(self, message: str):
        self.logs_text.append(message)

    def update_wallet_status(self, wallet_name: str):
        self.wallet_status_label.setText(wallet_name)
        if hasattr(self, 'pool_mining_widget') and self.pool_mining_widget:
            self.pool_mining_widget.update_wallet_status(wallet_name)

    def update_connection_status(self, connected: bool):
        status_text = "Connected" if connected else "Disconnected"
        self.connection_status_label.setText(status_text)
        if connected:
            self.connection_status_label.setStyleSheet("color: #51cf66;")
        else:
            self.connection_status_label.setStyleSheet("color: #74c0fc;")

    def update_global_sota(self):
        if not self.main_window:
            return

        try:
            sota = self.main_window.get_current_sota()
            if sota is not None:
                self._global_sota_value = float(sota)
                self.global_sota_label.setText(f"{sota:.4f}")
            else:
                self._global_sota_value = None
                self.global_sota_label.setText("-")
        except Exception as e:
            print(f"Error fetching SOTA: {e}")
            self._global_sota_value = None
            self.global_sota_label.setText("-")

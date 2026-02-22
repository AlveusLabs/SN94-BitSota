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
from typing import Optional, Any
import logging
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from uuid import uuid4

from gui.components import PrimaryButton, SecondaryButton
from gui.components.modal import ConfirmationModal
from gui.components.invite_code_modal import InviteCodeModal
from gui.app_config import get_app_config
from gui.components.coming_soon_modal import ComingSoonModal
from gui.screens.pool_mining_screen import PoolMiningScreen
from gui.resource_path import resource_path
import requests
import time


_CPP_DEFAULT_ENGINE_PARAMS_BY_TASK = {
    # Mirrors cpp/automl_zero/run_baseline.sh memory sizes and per-phase op budgets.
    "cifar10_binary": {
        "scalar_count": 5,
        "vector_count": 9,
        "matrix_count": 2,
        "phase_max_sizes": {"setup": 7, "predict": 11, "learn": 23},
    },
    # Mirrors cpp/automl_zero/run_demo.sh memory sizes and fixed phase sizes.
    "scalar_linear": {
        "scalar_count": 4,
        "vector_count": 3,
        "matrix_count": 1,
        "phase_max_sizes": {"setup": 10, "predict": 2, "learn": 8},
    },
}


def _apply_cpp_defaults_to_engine_params(
    task_type: str,
    engine_params: Optional[dict],
    *,
    explicit_engine_params: Optional[dict] = None,
) -> Optional[dict]:
    """
    Apply C++-aligned defaults for memory sizes + phase op limits.

    Values from `explicit_engine_params` (typically problem_config.engine_params)
    are treated as user overrides and are not overwritten.
    """

    base: dict = dict(engine_params) if isinstance(engine_params, dict) else {}
    explicit: dict = dict(explicit_engine_params) if isinstance(explicit_engine_params, dict) else {}
    defaults = _CPP_DEFAULT_ENGINE_PARAMS_BY_TASK.get(str(task_type), {})
    if not defaults:
        return base or None

    for key in ("scalar_count", "vector_count", "matrix_count"):
        if key in explicit:
            continue
        if key in defaults:
            base[key] = int(defaults[key])

    default_phase_sizes = defaults.get("phase_max_sizes")
    if isinstance(default_phase_sizes, dict) and "phase_max_sizes" not in explicit:
        base["phase_max_sizes"] = dict(default_phase_sizes)

    return base or None


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
        self._re_gen_line = re.compile(r"^Gen\s+(\d+)\b", re.IGNORECASE)
        self._last_generation_seen = 0
        self._last_regularized_iter_seen = 0

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

        updated_tasks = False
        if msg.startswith("Gen "):
            m_gen = self._re_gen_line.match(msg)
            if m_gen:
                try:
                    generation = int(m_gen.group(1))
                except Exception:
                    generation = None
                if generation is not None and generation > int(self._last_generation_seen):
                    delta = int(generation) - int(self._last_generation_seen)
                    self._last_generation_seen = int(generation)
                    self.task.tasks_completed += int(delta)
                    updated_tasks = True
        elif msg.startswith("[regularized-evo]") and not suppress_log:
            m_iter = self._re_regularized_iter.search(msg)
            if m_iter:
                try:
                    iteration = int(m_iter.group(1))
                except Exception:
                    iteration = None
                if iteration is not None and iteration > int(self._last_regularized_iter_seen):
                    delta = int(iteration) - int(self._last_regularized_iter_seen)
                    self._last_regularized_iter_seen = int(iteration)
                    self.task.tasks_completed += int(delta)
                    updated_tasks = True

        # Emit periodic stats updates on progress logs (avoid per-generation GUI updates).
        if updated_tasks and msg.startswith("Gen "):
            self.stats_signal.emit(
                {
                    "tasks_completed": self.task.tasks_completed,
                    "successful_submissions": self.task.successful_submissions,
                    "best_score": self.task.best_score,
                }
            )

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
                profile_out = os.getenv("BITSOTA_GUI_PROFILE_ONE_GEN_OUT", "").strip()
                if profile_out:
                    import cProfile

                    out_path = os.path.expanduser(profile_out)
                    out_dir = os.path.dirname(out_path)
                    if out_dir:
                        try:
                            os.makedirs(out_dir, exist_ok=True)
                        except Exception:
                            pass

                    client = self.client
                    original_get_engine = getattr(client, "_get_engine", None)
                    if callable(original_get_engine):
                        counter = {"n": 0}

                        def _wrapped_get_engine(task_type: str, engine_type: str = "baseline"):
                            engine = original_get_engine(task_type, engine_type)
                            if getattr(engine, "_bitsota_profile_wrapped", False):
                                return engine
                            original_evolve = getattr(engine, "evolve_generation", None)
                            if not callable(original_evolve):
                                return engine

                            def _wrapped_evolve_generation(*args, **kwargs):
                                out = original_evolve(*args, **kwargs)
                                counter["n"] += 1
                                if counter["n"] >= 1:
                                    try:
                                        client.stop_signal = True
                                    except Exception:
                                        pass
                                return out

                            try:
                                engine.evolve_generation = _wrapped_evolve_generation  # type: ignore[method-assign]
                                engine._bitsota_profile_wrapped = True  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            return engine

                        try:
                            client._get_engine = _wrapped_get_engine  # type: ignore[method-assign]
                        except Exception:
                            pass

                    self.signals.log.emit(
                        f"[profiling] Capturing 1 generation to {out_path} (BITSOTA_GUI_PROFILE_ONE_GEN_OUT)"
                    )
                    prof = cProfile.Profile()
                    prof.enable()
                    try:
                        result = client.run_continuous_mining(
                            task_type=self.task_type,
                            engine_type=self.engine_type,
                            checkpoint_generations=self.checkpoint_generations,
                        )
                    finally:
                        prof.disable()
                        try:
                            prof.dump_stats(out_path)
                        except Exception as e:
                            self.signals.log.emit(f"[profiling] Failed to write pstats: {e}")
                        if callable(original_get_engine):
                            try:
                                client._get_engine = original_get_engine  # type: ignore[method-assign]
                            except Exception:
                                pass
                else:
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


class MiningScreen(QWidget):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.is_mining = False
        self.sidecar_process: Optional[subprocess.Popen] = None
        self.miner_process: Optional[subprocess.Popen] = None
        self.sidecar_url: Optional[str] = None
        self.sidecar_run_id: Optional[str] = None
        self.sidecar_log_cursor = 0
        self.sidecar_candidate_cursor = 0
        self.sidecar_population_cursor = 0
        self._pool_coordinator = None
        self._pool_task_type: Optional[str] = None
        self.thread_pool = QThreadPool()
        self.tasks_completed = 0
        self.successful_submissions = 0
        self.best_score = None
        self._auto_start_enabled = self._env_flag("BITSOTA_AUTO_START_MINING")
        self._auto_start_task_type = os.getenv("BITSOTA_AUTO_TASK_TYPE", "").strip()
        self._auto_start_workers = os.getenv("BITSOTA_AUTO_WORKERS", "").strip()
        self._auto_start_attempts = 0
        self._auto_start_max_attempts = 30
        self.setup_ui()
        self._load_mining_stats()

        self.sota_timer = QTimer()
        self.sota_timer.timeout.connect(self._refresh_global_sota_from_relay)

        self.sidecar_poll_timer = QTimer()
        self.sidecar_poll_timer.timeout.connect(self._poll_sidecar)
        if self._auto_start_enabled:
            QTimer.singleShot(1200, self._maybe_autostart_from_env)

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
        # Allow selecting supported task types regardless of test_mode so profiling/local
        # runs can match `scripts/miner_local_og.py` behavior.
        self.task_type_map = {
            "CIFAR-10 Binary": "cifar10_binary",
            "MNIST Binary": "mnist_binary",
            "Scalar Linear": "scalar_linear",
            "Pool": "pool",
            "Pool Lease": "pool_lease",
        }
        self.task_type_combo.addItems(list(self.task_type_map.keys()))
        self.task_type_combo.setEnabled(True)
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
        if self.is_mining:
            self._append_log("ERROR: Mining is already running.")
            return

        if not self.main_window:
            self._append_log("ERROR: Main window reference not available.")
            return

        if not self.main_window.wallet:
            self._append_log("ERROR: No wallet loaded. Please load a wallet first.")
            return

        task_display = self.task_type_combo.currentText()
        task_type = self.task_type_map.get(task_display, "cifar10_binary")
        self._pool_task_type = str(task_type)
        is_pool = str(task_type) in {"pool", "pool_lease"}
        no_relay_test = self._no_relay_test_enabled()

        if not is_pool:
            if not self.main_window.client and not no_relay_test:
                self._append_log("ERROR: Client not initialized. Please ensure wallet is properly loaded.")
                return

            if not self.main_window.coldkey_address and not no_relay_test:
                self._append_log(
                    "ERROR: No coldkey address provided. Please provide your coldkey address first."
                )
                self.main_window._prompt_for_coldkey_address()
                return

            try:
                relay_url = self.main_window._get_relay_endpoint_from_config()
                self._append_log(f"Relay endpoint: {relay_url}")
            except Exception:
                pass

            if no_relay_test:
                self._append_log("NO_RELAY_TEST enabled: skipping relay invite/coldkey checks and relay submissions.")
            else:
                if not self._check_invite_code():
                    self._show_invite_code_modal()
                    return

                if not self._send_coldkey_address():
                    self._append_log(
                        "ERROR: Failed to send coldkey address to relay. Please try again."
                    )
                    return

        workers = 1
        if hasattr(self, "workers_combo") and self.workers_combo is not None:
            try:
                workers = max(1, int(self.workers_combo.currentText()))
            except Exception:
                workers = 1

        try:
            sidecar_url = self._ensure_sidecar_running()
            self.sidecar_url = sidecar_url

            run_id = self._sidecar_start_run(sidecar_url)
            self.sidecar_run_id = run_id
            self.sidecar_log_cursor = 0
            self.sidecar_candidate_cursor = 0
            self.sidecar_population_cursor = 0

            global_sota = None
            if not is_pool:
                global_sota = self._refresh_global_sota_from_relay()
                if global_sota is not None:
                    self._sidecar_set_global_sota(sidecar_url, global_sota)

            if is_pool:
                self._start_pool_miner_process(
                    sidecar_url=sidecar_url,
                    run_id=run_id,
                    workers=workers,
                )
                self._start_pool_task_driver(
                    sidecar_url=sidecar_url,
                    run_id=run_id,
                    pool_mode=str(task_type),
                )
            else:
                self._start_miner_process(
                    sidecar_url=sidecar_url,
                    run_id=run_id,
                    task_type=task_type,
                    workers=workers,
                    global_sota=global_sota,
                )
        except Exception as e:
            self._append_log(f"ERROR: Failed to start mining processes: {e}")
            self._stop_mining_processes()
            return

        self.is_mining = True
        self.start_mining_btn.update_icon("gui/images/stop.svg")
        self.start_mining_btn.update_text("Stop Mining")
        self.start_mining_btn.setObjectName("stop_mining_button")
        self.start_mining_btn.setStyleSheet("")
        self.start_mining_btn.style().unpolish(self.start_mining_btn)
        self.start_mining_btn.style().polish(self.start_mining_btn)

        self._append_log(f"Starting mining for task: {task_type} (workers={workers})")
        self.update_connection_status(True)
        if not is_pool:
            self.sota_timer.start(30000)
        self.sidecar_poll_timer.start(1000)

    def _stop_mining(self):
        self.is_mining = False
        self.sota_timer.stop()
        self.sidecar_poll_timer.stop()
        self._stop_mining_processes()
        self.start_mining_btn.update_icon(resource_path("gui/images/play.svg"))
        self.start_mining_btn.update_text("Start Mining")
        self.start_mining_btn.setObjectName("primary_button")
        self.start_mining_btn.setStyleSheet("")
        self.start_mining_btn.style().unpolish(self.start_mining_btn)
        self.start_mining_btn.style().polish(self.start_mining_btn)
        self.update_connection_status(False)
        self._append_log("Mining stopped.")

    def _check_invite_code(self) -> bool:
        if self._no_relay_test_enabled():
            self._append_log("NO_RELAY_TEST enabled: skipping invite code requirement.")
            return True
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
        if self._no_relay_test_enabled():
            self._append_log("NO_RELAY_TEST enabled: skipping coldkey update call to relay.")
            return True
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

    def _on_mining_finished(self):
        self.is_mining = False
        self.sota_timer.stop()
        self.sidecar_poll_timer.stop()
        self._stop_mining_processes()
        self.start_mining_btn.update_icon(resource_path("gui/images/play.svg"))
        self.start_mining_btn.update_text("Start Mining")
        self.start_mining_btn.setObjectName("primary_button")
        self.start_mining_btn.setStyleSheet("")
        self.start_mining_btn.style().unpolish(self.start_mining_btn)
        self.start_mining_btn.style().polish(self.start_mining_btn)
        self.update_connection_status(False)
        self._append_log("Mining stopped.")

    @staticmethod
    def _get_sidecar_url() -> str:
        host = os.getenv("BITSOTA_SIDECAR_HOST", "127.0.0.1").strip() or "127.0.0.1"
        port = os.getenv("BITSOTA_SIDECAR_PORT", "8123").strip() or "8123"
        return f"http://{host}:{port}"

    @staticmethod
    def _env_flag(name: str) -> bool:
        raw = os.getenv(name, "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _maybe_autostart_from_env(self) -> None:
        if not self._auto_start_enabled or self.is_mining:
            return

        wallet = getattr(self.main_window, "wallet", None) if self.main_window else None
        if wallet is None:
            self._auto_start_attempts += 1
            if self._auto_start_attempts < self._auto_start_max_attempts:
                QTimer.singleShot(1000, self._maybe_autostart_from_env)
            return

        if self._auto_start_task_type:
            self._set_task_type_by_value(self._auto_start_task_type)

        if self._auto_start_workers:
            try:
                workers = max(1, int(self._auto_start_workers))
                self.workers_combo.setCurrentText(str(workers))
            except Exception:
                pass

        self._append_log(
            f"Auto-start enabled via env (task={self._auto_start_task_type or 'default'})."
        )
        self._start_mining()

    def _set_task_type_by_value(self, task_type: str) -> None:
        wanted = str(task_type or "").strip().lower()
        if not wanted:
            return
        for label, value in self.task_type_map.items():
            if str(value).strip().lower() == wanted:
                self.task_type_combo.setCurrentText(label)
                return

    @staticmethod
    def _is_frozen() -> bool:
        return bool(getattr(sys, "frozen", False))

    @staticmethod
    def _resolve_bundled_executable(*names: str) -> Optional[str]:
        if not names:
            return None

        search_dirs = []
        try:
            search_dirs.append(Path(sys.executable).resolve().parent)
        except Exception:
            pass
        try:
            base = getattr(sys, "_MEIPASS", None)
            if base:
                search_dirs.append(Path(base))
        except Exception:
            pass

        seen = set()
        for directory in search_dirs:
            key = str(directory)
            if key in seen:
                continue
            seen.add(key)
            for name in names:
                candidate = directory / str(name)
                if candidate.exists() and candidate.is_file():
                    return str(candidate)
        return None

    def _sidecar_launch_cmd(self) -> list[str]:
        if self._is_frozen():
            sidecar_bin = self._resolve_bundled_executable(
                "BitSotaSidecar.exe",
                "BitSotaSidecar",
            )
            if not sidecar_bin:
                raise RuntimeError("Bundled sidecar executable not found")
            return [
                sidecar_bin,
                "--host",
                os.getenv("BITSOTA_SIDECAR_HOST", "127.0.0.1"),
                "--port",
                os.getenv("BITSOTA_SIDECAR_PORT", "8123"),
            ]

        return [
            sys.executable,
            "-m",
            "sidecar",
            "--host",
            os.getenv("BITSOTA_SIDECAR_HOST", "127.0.0.1"),
            "--port",
            os.getenv("BITSOTA_SIDECAR_PORT", "8123"),
        ]

    def _miner_launch_prefix(self, *, pool: bool) -> list[str]:
        if self._is_frozen():
            miner_name = "BitSotaPoolMiner" if pool else "BitSotaMiner"
            miner_bin = self._resolve_bundled_executable(
                f"{miner_name}.exe",
                miner_name,
            )
            if not miner_bin:
                raise RuntimeError(f"Bundled executable not found: {miner_name}")
            return [miner_bin]

        if pool:
            return [sys.executable, "-m", "scripts.pool_miner_sidecar"]
        return [sys.executable, "-m", "scripts.miner_local_og_sidecar"]

    def _ensure_sidecar_running(self) -> str:
        url = self._get_sidecar_url().rstrip("/")
        try:
            r = requests.get(f"{url}/health", timeout=0.5)
            r.raise_for_status()
            return url
        except Exception:
            pass

        cmd = self._sidecar_launch_cmd()
        self.sidecar_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
            start_new_session=True,
        )
        try:
            self._append_log(f"Sidecar started (pid={int(self.sidecar_process.pid)}).")
        except Exception:
            pass

        deadline = time.time() + 5.0
        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(f"{url}/health", timeout=0.5)
                r.raise_for_status()
                return url
            except Exception as e:
                last_err = e
                time.sleep(0.1)
        raise RuntimeError(f"Sidecar failed to start: {last_err}")

    def _sidecar_start_run(self, sidecar_url: str) -> str:
        run_id = str(uuid4())
        r = requests.post(
            f"{sidecar_url.rstrip('/')}/runs/start",
            json={"run_id": run_id, "replace": True},
            timeout=1.0,
        )
        r.raise_for_status()
        payload = r.json() or {}
        return str(payload.get("run_id") or run_id)

    def _sidecar_set_global_sota(self, sidecar_url: str, value: float) -> None:
        try:
            payload = {"value": float(value)}
            if self.sidecar_run_id:
                payload["run_id"] = str(self.sidecar_run_id)
            requests.post(
                f"{sidecar_url.rstrip('/')}/set_global_sota",
                json=payload,
                timeout=1.0,
            )
        except Exception:
            return

    def _start_pool_miner_process(
        self,
        *,
        sidecar_url: str,
        run_id: str,
        workers: int,
    ) -> None:
        cfg = get_app_config()
        cmd = [
            *self._miner_launch_prefix(pool=True),
            "--sidecar-url",
            str(sidecar_url),
            "--run-id",
            str(run_id),
            "--workers",
            str(int(max(1, workers))),
            "--mode",
            "real",
            "--lease-evolve-generations",
            str(int(max(1, getattr(cfg, "pool_lease_evolve_generations", 1000)))),
        ]
        self.miner_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
            start_new_session=True,
        )
        try:
            self._append_log(f"Pool miner started (pid={int(self.miner_process.pid)}).")
        except Exception:
            pass

    def _start_pool_task_driver(self, *, sidecar_url: str, run_id: str, pool_mode: str) -> None:
        if not self.main_window or not self.main_window.wallet:
            raise RuntimeError("Wallet unavailable for pool driver")

        from gui.pool_task_driver import PoolApiClient, PoolLeaseCoordinator, PoolTaskCoordinator, SidecarJobClient

        cfg = get_app_config()
        pool_endpoint = str(getattr(cfg, "pool_endpoint", "") or "").strip()
        if not pool_endpoint:
            raise RuntimeError("Pool endpoint missing in config")

        pool_client = PoolApiClient(pool_endpoint, self.main_window.wallet, timeout_s=2.0)
        sidecar_jobs = SidecarJobClient(sidecar_url, run_id, timeout_s=0.5)
        if str(pool_mode) == "pool_lease":
            self._pool_coordinator = PoolLeaseCoordinator(
                pool_client=pool_client,
                sidecar_jobs=sidecar_jobs,
                log=self._append_log,
                request_interval_s=1.0,
            )
        else:
            self._pool_coordinator = PoolTaskCoordinator(
                pool_client=pool_client,
                sidecar_jobs=sidecar_jobs,
                log=self._append_log,
                request_interval_s=1.0,
            )
        self._append_log(f"[pool] Pool endpoint: {pool_endpoint}")

    def _start_miner_process(
        self,
        *,
        sidecar_url: str,
        run_id: str,
        task_type: str,
        workers: int,
        global_sota: Optional[float],
    ) -> None:
        cfg = get_app_config()
        problem_cfg = getattr(self.main_window, "problem_config", None)
        problem_config_path = getattr(cfg, "problem_config_path", None) or getattr(problem_cfg, "source_path", None)
        initial_population_path = getattr(cfg, "population_state_path", None)
        if isinstance(initial_population_path, str):
            initial_population_path = initial_population_path.strip() or None
        if not initial_population_path:
            try:
                from gui.wallet_utils_gui import get_population_state_file

                initial_population_path = str(get_population_state_file())
            except Exception:
                initial_population_path = None

        snapshot_every = None
        try:
            snapshot_every = int(getattr(problem_cfg, "checkpoint_generations", None) or 0)
        except Exception:
            snapshot_every = None
        if not snapshot_every:
            try:
                snapshot_every = int(getattr(cfg, "miner_validate_every_n_generations", 0) or 0)
            except Exception:
                snapshot_every = None

        cmd = [
            *self._miner_launch_prefix(pool=False),
            "--sidecar-url",
            sidecar_url,
            "--run-id",
            run_id,
            "--task-type",
            str(task_type),
            "--workers",
            str(int(max(1, workers))),
            "--iterations",
            str(int(getattr(problem_cfg, "miner_iterations", 0) or 0)),
        ]
        if global_sota is not None:
            cmd.extend(["--sota-threshold", str(float(global_sota))])
        if problem_config_path:
            cmd.extend(["--config", str(problem_config_path)])
        if snapshot_every and int(snapshot_every) > 0:
            cmd.extend(["--population-snapshot-every", str(int(snapshot_every))])
        if initial_population_path:
            try:
                p = os.path.expanduser(str(initial_population_path))
            except Exception:
                p = None
            if p and os.path.exists(p):
                cmd.extend(["--initial-population-path", str(p)])

        env = os.environ.copy()
        env_overrides = getattr(problem_cfg, "env", None) if problem_cfg is not None else None
        if isinstance(env_overrides, dict):
            for k, v in env_overrides.items():
                if k:
                    env[str(k)] = str(v)

        self.miner_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        try:
            self._append_log(f"Miner started (pid={int(self.miner_process.pid)}).")
        except Exception:
            pass

    @staticmethod
    def _terminate_process(proc: Optional[subprocess.Popen], *, timeout_s: float = 5.0) -> None:
        if proc is None:
            return

        try:
            if os.name == "posix" and getattr(proc, "pid", None):
                try:
                    os.killpg(int(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=max(0.1, float(timeout_s)))
            return
        except Exception:
            pass

        try:
            if os.name == "posix" and getattr(proc, "pid", None):
                try:
                    os.killpg(int(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
            else:
                proc.kill()
        except Exception:
            pass

        try:
            proc.wait(timeout=max(0.1, float(timeout_s)))
        except Exception:
            pass

    def _stop_mining_processes(self) -> None:
        proc = self.miner_process
        self.miner_process = None
        self._terminate_process(proc, timeout_s=5.0)

        sidecar_proc = self.sidecar_process
        self.sidecar_process = None
        self._terminate_process(sidecar_proc, timeout_s=5.0)

        self.sidecar_url = None
        self.sidecar_run_id = None
        self.sidecar_log_cursor = 0
        self.sidecar_candidate_cursor = 0
        self.sidecar_population_cursor = 0
        self._pool_coordinator = None
        self._pool_task_type = None

    def _refresh_global_sota_from_relay(self) -> Optional[float]:
        if self._no_relay_test_enabled():
            return None
        if not self.main_window:
            return None
        sota = None
        try:
            sota = self.main_window.get_current_sota()
        except Exception:
            sota = None
        if sota is None:
            return None
        if self.sidecar_url:
            self._sidecar_set_global_sota(self.sidecar_url, float(sota))
        return float(sota)

    def _poll_sidecar(self) -> None:
        if not self.sidecar_url:
            return

        if self.miner_process is not None and self.miner_process.poll() is not None:
            self._append_log("Miner process exited.")
            self._on_mining_finished()
            return

        base = self.sidecar_url.rstrip("/")
        try:
            st_params = {"run_id": str(self.sidecar_run_id)} if self.sidecar_run_id else None
            st = requests.get(f"{base}/state", params=st_params, timeout=0.5).json()
        except Exception:
            self.update_connection_status(False)
            return

        self.update_connection_status(True)

        try:
            tasks = int(st.get("tasks_completed", 0) or 0)
        except Exception:
            tasks = 0
        try:
            subs = int(st.get("successful_submissions", 0) or 0)
        except Exception:
            subs = 0
        local_sota = st.get("local_sota")
        global_sota = st.get("global_sota")

        self._update_stats(
            {
                "tasks_completed": tasks,
                "successful_submissions": subs,
                "best_score": local_sota,
            }
        )

        try:
            if global_sota is not None:
                self.global_sota_label.setText(f"{float(global_sota):.4f}")
            else:
                self.global_sota_label.setText("-")
        except Exception:
            self.global_sota_label.setText("-")

        # Logs
        try:
            r = requests.get(
                f"{base}/logs",
                params={
                    "run_id": str(self.sidecar_run_id),
                    "cursor": int(self.sidecar_log_cursor),
                    "limit": 200,
                }
                if self.sidecar_run_id
                else {"cursor": int(self.sidecar_log_cursor), "limit": 200},
                timeout=0.5,
            )
            payload = r.json() or {}
            items = payload.get("items") or []
            cursor = payload.get("cursor")
            for it in items:
                msg = None
                if isinstance(it, dict):
                    msg = it.get("message")
                if msg:
                    self._append_log(str(msg))
            if cursor is not None:
                self.sidecar_log_cursor = int(cursor)
        except Exception:
            pass

        # Candidates
        try:
            r = requests.get(
                f"{base}/candidates",
                params={
                    "run_id": str(self.sidecar_run_id),
                    "cursor": int(self.sidecar_candidate_cursor),
                    "limit": 50,
                }
                if self.sidecar_run_id
                else {"cursor": int(self.sidecar_candidate_cursor), "limit": 50},
                timeout=0.5,
            )
            payload = r.json() or {}
            items = payload.get("items") or []
            cursor = payload.get("cursor")
            if cursor is not None:
                self.sidecar_candidate_cursor = int(cursor)

            for cand in items:
                if isinstance(cand, dict):
                    self._maybe_submit_candidate(cand, global_sota=global_sota, local_sota=local_sota)
        except Exception:
            pass

        # Populations (for resume)
        try:
            r = requests.get(
                f"{base}/populations",
                params={
                    "run_id": str(self.sidecar_run_id),
                    "cursor": int(self.sidecar_population_cursor),
                    "limit": 10,
                }
                if self.sidecar_run_id
                else {"cursor": int(self.sidecar_population_cursor), "limit": 10},
                timeout=0.5,
            )
            payload = r.json() or {}
            items = payload.get("items") or []
            cursor = payload.get("cursor")
            if cursor is not None:
                self.sidecar_population_cursor = int(cursor)
            for snap in items:
                if isinstance(snap, dict):
                    self._persist_population_snapshot(snap)
        except Exception:
            pass

        coordinator = self._pool_coordinator
        if coordinator is not None:
            try:
                coordinator.tick()
            except Exception:
                pass

    def _persist_population_snapshot(self, snap: dict) -> None:
        cfg = get_app_config()
        path = getattr(cfg, "population_state_path", None)
        if isinstance(path, str):
            path = path.strip() or None
        if not path:
            try:
                from gui.wallet_utils_gui import get_population_state_file

                path = str(get_population_state_file())
            except Exception:
                return

        try:
            resolved = os.path.expanduser(str(path))
        except Exception:
            resolved = str(path)

        pop = snap.get("population")
        if not isinstance(pop, list) or not pop:
            return

        worker_id = snap.get("worker_id", 0)
        iteration = snap.get("iteration", 0)
        task_type = snap.get("task_type")
        engine = snap.get("engine")

        try:
            wid = str(int(worker_id))
        except Exception:
            wid = str(worker_id)

        state: dict = {}
        try:
            if os.path.exists(resolved):
                with open(resolved, "r") as f:
                    state = json.load(f)
        except Exception:
            state = {}

        if not isinstance(state, dict):
            state = {}

        # If task/engine changed, start a fresh state file.
        if task_type and state.get("task_type") not in {None, task_type}:
            state = {}
        if engine and state.get("engine") not in {None, engine}:
            state = {}

        workers = state.get("workers")
        if not isinstance(workers, dict):
            workers = {}

        workers[wid] = {"iteration": iteration, "population": pop}
        state["version"] = 1
        if task_type:
            state["task_type"] = task_type
        if engine:
            state["engine"] = engine
        if self.sidecar_run_id:
            state["run_id"] = self.sidecar_run_id
        state["updated_at_s"] = time.time()
        state["workers"] = workers

        try:
            out_dir = os.path.dirname(resolved)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(resolved, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            return

    def _maybe_submit_candidate(self, cand: dict, *, global_sota: Any, local_sota: Any) -> None:
        no_relay_test = self._no_relay_test_enabled()

        if not self.main_window or not self.main_window.client:
            if not no_relay_test:
                return

        try:
            verified_score = float(cand.get("validator_score", -float("inf")))
        except Exception:
            return

        g = None
        l = None
        try:
            if global_sota is not None:
                g = float(global_sota)
        except Exception:
            g = None
        try:
            if local_sota is not None:
                l = float(local_sota)
        except Exception:
            l = None

        threshold = max(v for v in [g, l] if v is not None) if (g is not None or l is not None) else None
        if threshold is not None and verified_score <= float(threshold):
            return

        algo_dsl = cand.get("algorithm_dsl")
        input_dim = cand.get("input_dim")
        if not algo_dsl or input_dim is None:
            return

        task_type = str(cand.get("task_type") or "cifar10_binary")
        eval_score = cand.get("eval_score")
        worker_id = cand.get("worker_id")
        iteration = cand.get("iteration")

        task_id = f"{self.sidecar_run_id or 'run'}:{worker_id}:{iteration}"
        solution_data = {
            "task_id": task_id,
            "task_type": task_type,
            "algorithm_dsl": str(algo_dsl),
            "input_dim": int(input_dim),
            "eval_score": float(eval_score) if eval_score is not None else float(verified_score),
            "worker_id": worker_id,
            "iteration": iteration,
        }

        prevalidated = {"verified_score": float(verified_score)}
        if g is not None:
            prevalidated["sota_threshold"] = float(g)

        self._append_log(f"[submit] Candidate verified_score={float(verified_score):.6f} threshold={threshold}")
        if no_relay_test:
            status = "success"
            self._append_log("[submit] NO_RELAY_TEST enabled: skipping relay submit, updating local sidecar state only.")
        else:
            result = self.main_window.client.submit_solution(solution_data, prevalidated=prevalidated)
            status = str(result.get("status") or "")
        self._append_log(f"[submit] status={status}")

        if self.sidecar_url:
            try:
                payload = {"score": float(verified_score), "status": status}
                if self.sidecar_run_id:
                    payload["run_id"] = str(self.sidecar_run_id)
                requests.post(
                    f"{self.sidecar_url.rstrip('/')}/submission_result",
                    json=payload,
                    timeout=1.0,
                )
            except Exception:
                pass

    @staticmethod
    def _no_relay_test_enabled() -> bool:
        raw = os.getenv("NO_RELAY_TEST", "").strip()
        if not raw:
            raw = os.getenv("BITSOTA_NO_RELAY_TEST", "").strip()
        return raw.lower() in {"1", "true", "yes", "on"}

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

        label = QLabel("Local SOTA")
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
        self.logs_text.document().setMaximumBlockCount(5000)
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
            sota = self._refresh_global_sota_from_relay()
            if sota is None:
                self.global_sota_label.setText("-")
                return
            # When mining is running the label is driven by sidecar polling,
            # but we still set it here for the idle case.
            if not self.sidecar_url:
                self.global_sota_label.setText(f"{float(sota):.4f}")
        except Exception as e:
            print(f"Error fetching SOTA: {e}")
            self.global_sota_label.setText("-")

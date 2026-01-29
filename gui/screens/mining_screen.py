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
from gui.components.modals.base import ConfirmationModal
from gui.app_config import get_app_config
from gui.screens.pool_mining_screen import PoolMiningScreen
from gui.resource_path import resource_path
import requests
import time


class GUILogHandler(logging.Handler):
    def __init__(self, log_signal, stats_signal, task):
        super().__init__()
        self.log_signal = log_signal
        self.stats_signal = stats_signal
        self.task = task
        number = r"([-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)"
        self._re_score_verified = re.compile(rf"\bScore:\s*{number}\s*\(verified\)", re.IGNORECASE)
        self._re_verified_score = re.compile(rf"\bverified_score\b[^0-9\-\+]*{number}", re.IGNORECASE)

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

    def __init__(self, client, task_type: str, stop_flag, initial_tasks=0, initial_submissions=0, initial_best_score=None):
        super().__init__()
        self.client = client
        self.task_type = task_type
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
            self.signals.log.emit(f"Starting {self.task_type} mining with baseline engine")

            if hasattr(self.client, "run_continuous_mining"):
                result = self.client.run_continuous_mining(
                    task_type=self.task_type,
                    engine_type="baseline",
                    checkpoint_generations=10,
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
        self.mining_task: Optional[DirectMiningTask] = None
        self.thread_pool = QThreadPool()
        self.tasks_completed = 0
        self.successful_submissions = 0
        self.best_score = None
        self.setup_ui()
        self._load_mining_stats()

        self.sota_timer = QTimer()
        self.sota_timer.timeout.connect(self.update_global_sota)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.content_stack = QWidget()
        self.content_stack_layout = QVBoxLayout(self.content_stack)
        self.content_stack_layout.setContentsMargins(0, 0, 0, 0)

        self.direct_mining_widget = QWidget()
        direct_layout = QVBoxLayout(self.direct_mining_widget)
        direct_layout.setContentsMargins(0, 0, 0, 0)

        # White container contains all content
        content_box = QWidget()
        content_box.setObjectName("content_box")
        content_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content_layout = QVBoxLayout(content_box)
        content_layout.setSpacing(0)

        # Tab switcher - centered display
        from gui.components.common.tab_switcher import TabSwitcher
        
        tab_container = QWidget()
        tab_container_layout = QHBoxLayout(tab_container)
        tab_container_layout.setContentsMargins(0, 0, 0, 0)
        tab_container_layout.addStretch()
        
        self.tab_switcher = TabSwitcher()
        self.tab_switcher.add_tab("direct", "Direct Mining")
        self.tab_switcher.add_tab("pool", "Pool Mining")
        self.tab_switcher.tab_changed.connect(self._on_mining_tab_changed)
        tab_container_layout.addWidget(self.tab_switcher)
        
        tab_container_layout.addStretch()
        content_layout.addWidget(tab_container)
        
        content_layout.addSpacing(12)

        # Description text - centered display
        self.description = QLabel(
            "Connect straight to Bittensor validators, ideal for users who want complete control over their mining operations."
        )
        self.description.setObjectName("mining_description")
        self.description.setWordWrap(True)
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.description)
        
        content_layout.addSpacing(24)

        config_section = self._create_config_section()
        content_layout.addWidget(config_section)
        
        content_layout.addSpacing(24)

        stats_status_layout = QHBoxLayout()
        stats_status_layout.setSpacing(24)

        miner_stats = self._create_miner_stats()
        stats_status_layout.addWidget(miner_stats, 1)

        mining_status = self._create_mining_status()
        stats_status_layout.addWidget(mining_status, 1)

        content_layout.addLayout(stats_status_layout)
        
        content_layout.addSpacing(24)

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
        title.setObjectName("config_section_title")
        layout.addWidget(title)

        # Task Type and buttons in the same row
        config_row = QHBoxLayout()
        config_row.setSpacing(24)

        # Task Type area
        task_container = QWidget()
        task_layout = QVBoxLayout(task_container)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(8)

        task_label = QLabel("Task Type")
        task_label.setObjectName("form_label")
        task_layout.addWidget(task_label)

        self.task_type_combo = QComboBox()
        self.task_type_combo.setObjectName("form_input")
        self.task_type_combo.setFixedHeight(48)
        self.task_type_map = {
            "CIFAR-10 Binary Classification": "cifar10_binary",
        }
        self.task_type_combo.addItems(list(self.task_type_map.keys()))
        self.task_type_combo.setEnabled(False)
        self.task_type_combo.currentTextChanged.connect(lambda: self.update_global_sota())
        task_layout.addWidget(self.task_type_combo)

        config_row.addWidget(task_container, 1)

        # Button container - right aligned
        button_container = QHBoxLayout()
        button_container.setSpacing(16)

        self.save_config_btn = SecondaryButton("Save Configuration", width=200, height=48)
        button_container.addWidget(self.save_config_btn)

        self.start_mining_btn = PrimaryButton("Start Mining", width=200, height=48, icon_path=resource_path("gui/images/play.svg"))
        self.start_mining_btn.clicked.connect(self._toggle_mining)
        button_container.addWidget(self.start_mining_btn)

        config_row.addLayout(button_container)

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

        from gui.stop_flag import StopFlag
        stop_flag = StopFlag()

        self.mining_task = DirectMiningTask(
            client=self.main_window.client,
            task_type=task_type,
            stop_flag=stop_flag,
            initial_tasks=self.tasks_completed,
            initial_submissions=self.successful_submissions,
            initial_best_score=self.best_score
        )

        self.mining_task.signals.log.connect(self._append_log)
        self.mining_task.signals.error.connect(self._handle_mining_error)
        self.mining_task.signals.finished.connect(self._on_mining_finished)
        self.mining_task.signals.stats_updated.connect(self._update_stats)

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
        self.main_window.modal_manager.show_invite_code(
            relay_url=relay_url,
            wallet=self.main_window.wallet,
            coldkey_address=coldkey_address
        )

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
            self.main_window.modal_manager.show_coming_soon(
                "Pool Mining Screen",
                "The Pool Mining screen is coming soon! This screen will allow you to join mining pools for simplified setup and shared resources. Pool mining is ideal for miners who want a streamlined experience with automated task distribution and reward payouts."
            )
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
        title.setObjectName("config_section_title")
        layout.addWidget(title)

        stats_container = QVBoxLayout()
        stats_container.setSpacing(12)

        # Total Score
        row = self._create_stat_row("Total Score", "182.8")
        self.total_score_label = row[1]
        stats_container.addLayout(row[0])
        
        stats_container.addWidget(self._create_divider())

        # Evaluation Tasks Completed
        row = self._create_stat_row("Evaluation Tasks Completed", "0")
        self.eval_tasks_label = row[1]
        stats_container.addLayout(row[0])

        # Evaluation Score
        row = self._create_stat_row("Score", "-")
        self.eval_score_label = row[1]
        stats_container.addLayout(row[0])

        # Evolution Tasks Completed
        row = self._create_stat_row("Evolution Tasks Completed", "0")
        self.evo_tasks_label = row[1]
        stats_container.addLayout(row[0])

        # Evolution Score
        row = self._create_stat_row("Score", "-")
        self.evo_score_label = row[1]
        stats_container.addLayout(row[0])

        # Keep old label references for compatibility
        self.tasks_completed_label = self.eval_tasks_label
        self.successful_submissions_label = self.evo_tasks_label
        self.best_score_label = self.eval_score_label

        layout.addLayout(stats_container)

        return stats
    
    def _create_stat_row(self, label_text: str, value_text: str):
        """Create statistics row"""
        row_layout = QHBoxLayout()
        row_layout.setSpacing(0)
        
        label = QLabel(label_text)
        label.setObjectName("stat_label")
        row_layout.addWidget(label)
        
        row_layout.addStretch()
        
        value = QLabel(value_text)
        value.setObjectName("stat_value")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        row_layout.addWidget(value)
        
        return (row_layout, value)
    
    def _create_divider(self) -> QWidget:
        """Create divider"""
        divider = QWidget()
        divider.setObjectName("stat_divider")
        divider.setFixedHeight(1)
        return divider

    def _create_mining_status(self) -> QWidget:
        status = QWidget()
        status.setObjectName("stats_box")
        layout = QVBoxLayout(status)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Mining Status")
        title.setObjectName("config_section_title")
        layout.addWidget(title)

        status_container = QVBoxLayout()
        status_container.setSpacing(12)

        # SOTA
        row = self._create_stat_row("SOTA", "-")
        self.global_sota_label = row[1]
        status_container.addLayout(row[0])
        
        status_container.addWidget(self._create_divider())

        # Wallet
        row = self._create_stat_row("Wallet", "Not Connected")
        self.wallet_status_label = row[1]
        status_container.addLayout(row[0])

        # Status (with status indicator)
        status_row = QHBoxLayout()
        status_row.setSpacing(0)
        
        status_label = QLabel("Status")
        status_label.setObjectName("stat_label")
        status_row.addWidget(status_label)
        status_row.addStretch()
        
        # Status indicator container
        self.status_indicator_container = QWidget()
        status_indicator_layout = QHBoxLayout(self.status_indicator_container)
        status_indicator_layout.setContentsMargins(4, 2, 4, 2)
        status_indicator_layout.setSpacing(4)
        
        self.status_dot = QWidget()
        self.status_dot.setObjectName("status_dot_idle")
        self.status_dot.setFixedSize(6, 6)
        status_indicator_layout.addWidget(self.status_dot)
        
        self.mining_status_label = QLabel("Idle")
        self.mining_status_label.setObjectName("status_text_idle")
        status_indicator_layout.addWidget(self.mining_status_label)
        
        status_row.addWidget(self.status_indicator_container)
        status_container.addLayout(status_row)

        # Connection (with status indicator)
        connection_row = QHBoxLayout()
        connection_row.setSpacing(0)
        
        connection_label = QLabel("Connection")
        connection_label.setObjectName("stat_label")
        connection_row.addWidget(connection_label)
        connection_row.addStretch()
        
        # Connection status indicator container
        self.connection_indicator_container = QWidget()
        connection_indicator_layout = QHBoxLayout(self.connection_indicator_container)
        connection_indicator_layout.setContentsMargins(4, 2, 4, 2)
        connection_indicator_layout.setSpacing(4)
        
        self.connection_dot = QWidget()
        self.connection_dot.setObjectName("status_dot_disconnected")
        self.connection_dot.setFixedSize(6, 6)
        connection_indicator_layout.addWidget(self.connection_dot)
        
        self.connection_status_label = QLabel("Disconnected")
        self.connection_status_label.setObjectName("status_text_disconnected")
        connection_indicator_layout.addWidget(self.connection_status_label)
        
        connection_row.addWidget(self.connection_indicator_container)
        status_container.addLayout(connection_row)

        # Tasks
        row = self._create_stat_row("Tasks", "0")
        self.tasks_label = row[1]
        status_container.addLayout(row[0])

        # Runtime
        row = self._create_stat_row("Runtime", "0h 0m 0s")
        self.runtime_label = row[1]
        status_container.addLayout(row[0])

        # Submissions
        row = self._create_stat_row("Submissions", "0")
        self.submissions_label = row[1]
        status_container.addLayout(row[0])

        layout.addLayout(status_container)

        return status

    def _create_logs_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("logs_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title = QLabel("Mining Logs")
        title.setObjectName("logs_title")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.setObjectName("clear_logs_button")
        self.clear_logs_btn.setFixedHeight(32)
        self.clear_logs_btn.clicked.connect(self._clear_logs)
        header_layout.addWidget(self.clear_logs_btn)

        layout.addLayout(header_layout)

        self.logs_text = QTextEdit()
        self.logs_text.setObjectName("logs_text")
        self.logs_text.setReadOnly(True)
        self.logs_text.setFixedHeight(140)
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
        if connected:
            self.connection_status_label.setText("Connected")
            self.connection_status_label.setObjectName("status_text_connected")
            self.connection_dot.setObjectName("status_dot_connected")
            self.mining_status_label.setText("Running")
            self.mining_status_label.setObjectName("status_text_running")
            self.status_dot.setObjectName("status_dot_running")
        else:
            self.connection_status_label.setText("Disconnected")
            self.connection_status_label.setObjectName("status_text_disconnected")
            self.connection_dot.setObjectName("status_dot_disconnected")
            self.mining_status_label.setText("Idle")
            self.mining_status_label.setObjectName("status_text_idle")
            self.status_dot.setObjectName("status_dot_idle")
        
        # Refresh styles
        self.connection_status_label.style().unpolish(self.connection_status_label)
        self.connection_status_label.style().polish(self.connection_status_label)
        self.connection_dot.style().unpolish(self.connection_dot)
        self.connection_dot.style().polish(self.connection_dot)
        self.mining_status_label.style().unpolish(self.mining_status_label)
        self.mining_status_label.style().polish(self.mining_status_label)
        self.status_dot.style().unpolish(self.status_dot)
        self.status_dot.style().polish(self.status_dot)

    def update_global_sota(self):
        if not self.main_window:
            return

        try:
            sota = self.main_window.get_current_sota()
            if sota is not None:
                self.global_sota_label.setText(f"{sota:.4f}")
            else:
                self.global_sota_label.setText("-")
        except Exception as e:
            print(f"Error fetching SOTA: {e}")
            self.global_sota_label.setText("-")

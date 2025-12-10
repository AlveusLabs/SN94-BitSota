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

from gui.components import PrimaryButton, SecondaryButton
from gui.components.invite_code_modal import InviteCodeModal
from gui.components.coming_soon_modal import ComingSoonModal
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

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

        if "Solution submitted to relay" in msg or "SOTA broken" in msg or ("submission" in msg.lower() and "successful" in msg.lower()):
            self.task.successful_submissions += 1
            self.stats_signal.emit({
                "tasks_completed": self.task.tasks_completed,
                "successful_submissions": self.task.successful_submissions,
                "best_score": self.task.best_score
            })

        if "Gen:" in msg or ("generation" in msg.lower() and "Gen" not in msg):
            self.task.tasks_completed += 1

        if "score:" in msg.lower() or "Score:" in msg:
            try:
                parts = msg.lower().split("score:")
                if len(parts) > 1:
                    score_str = parts[1].split()[0].strip(',')
                    score = float(score_str)
                    if self.task.best_score is None or score > self.task.best_score:
                        self.task.best_score = score
                        self.stats_signal.emit({
                            "tasks_completed": self.task.tasks_completed,
                            "successful_submissions": self.task.successful_submissions,
                            "best_score": self.task.best_score
                        })
            except:
                pass


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
        logger = logging.getLogger("miner.client")
        handler = GUILogHandler(self.signals.log, self.signals.stats_updated, self)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

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
            logger.removeHandler(handler)
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
        self.task_type_map = {
            "CIFAR-10 Binary": "cifar10_binary",
        }
        self.task_type_combo.addItems(list(self.task_type_map.keys()))
        self.task_type_combo.setEnabled(False)
        self.task_type_combo.currentTextChanged.connect(lambda: self.update_global_sota())
        config_row.addWidget(self.task_type_combo, 1)

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
                self.global_sota_label.setText(f"{sota:.4f}")
            else:
                self.global_sota_label.setText("-")
        except Exception as e:
            print(f"Error fetching SOTA: {e}")
            self.global_sota_label.setText("-")

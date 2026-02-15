# PySide6 imports
from PySide6.QtCore import Qt, QTimer, QRunnable, QThreadPool, QObject, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter

# Standard library imports
from typing import Optional
import logging
import re
import time

# Third-party imports
import requests

# Local application imports
from gui.components import PrimaryButton, SelectInput
from gui.app_config import get_app_config
from gui.resource_path import resource_path
from gui.stop_flag import StopFlag
from gui.wallet_utils_gui import load_mining_stats, save_mining_stats

# Constants
SOTA_UPDATE_INTERVAL_MS = 30000  # 30 seconds
HTTP_TIMEOUT_SECONDS = 10
CHECKPOINT_GENERATIONS = 10


class GUILogHandler(logging.Handler):
    """Custom logging handler for mining GUI that extracts statistics from logs"""

    def __init__(self, log_signal, stats_signal, task):
        super().__init__()
        self.log_signal = log_signal
        self.stats_signal = stats_signal
        self.task = task

        # Regex patterns for extracting scores from log messages
        number = r"([-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)"
        self._re_score_verified = re.compile(
            rf"\bScore:\s*{number}\s*\(verified\)", re.IGNORECASE
        )
        self._re_verified_score = re.compile(
            rf"\bverified_score\b[^0-9\-\+]*{number}", re.IGNORECASE
        )

    def _emit_stats_update(self):
        """Emit current task statistics"""
        self.stats_signal.emit(
            {
                "tasks_completed": self.task.tasks_completed,
                "successful_submissions": self.task.successful_submissions,
                "best_score": self.task.best_score,
            }
        )

    def _maybe_update_best_verified(self, verified_score: float):
        """Update best verified score if higher than current"""
        try:
            verified_score = float(verified_score)
        except Exception:
            return

        if self.task.best_score is None or verified_score > self.task.best_score:
            self.task.best_score = verified_score
            self._emit_stats_update()

    def emit(self, record):
        """Process log records and extract mining statistics"""
        msg = self.format(record)
        self.log_signal.emit(msg)

        # Check for successful submission
        if (
            "Solution submitted to relay" in msg
            or ("SOTA submission #" in msg and "successful" in msg.lower())
            or ("submission" in msg.lower() and "successful" in msg.lower())
        ):
            self.task.successful_submissions += 1
            best_verified = self._get_best_verified_score()

            if best_verified is not None:
                self._maybe_update_best_verified(best_verified)
            else:
                self._emit_stats_update()

        # Check for generation/task completion
        if (
            msg.startswith("[regularized-evo]")
            or msg.startswith("Gen ")
            or "generation" in msg.lower()
        ):
            self.task.tasks_completed += 1

        # Check for verified scores in message
        m = self._re_score_verified.search(msg)
        if m:
            self._maybe_update_best_verified(m.group(1))
            return

        m = self._re_verified_score.search(msg)
        if m:
            self._maybe_update_best_verified(m.group(1))
            return

    def _get_best_verified_score(self) -> Optional[float]:
        """Get best verified score from client"""
        try:
            if hasattr(self.task.client, "get_local_best_verified_score"):
                return self.task.client.get_local_best_verified_score(self.task.task_type)
            elif hasattr(self.task.client, "_local_best_verified_score"):
                return self.task.client._local_best_verified_score.get(self.task.task_type)  # type: ignore[attr-defined]
        except Exception:
            pass
        return None


class DirectMiningTask(QRunnable):
    """Background task for running direct mining operations"""

    class Signals(QObject):
        """Qt signals for communicating with the main thread"""

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
        initial_tasks=0,
        initial_submissions=0,
        initial_best_score=None,
    ):
        super().__init__()
        self.client = client
        self.task_type = task_type
        self.stop_flag = stop_flag
        self.signals = self.Signals()
        self.setAutoDelete(True)

        # Initialize statistics
        self.tasks_completed = initial_tasks
        self.successful_submissions = initial_submissions
        self.best_score = initial_best_score

    def stop(self):
        """Stop the mining task gracefully"""
        self.stop_flag.stop()
        if hasattr(self.client, "stop_mining"):
            self.client.stop_mining()
        self.signals.stopping.emit()

    @Slot()
    def run(self):
        """Run the mining task in a background thread"""
        # Setup logging
        tracked_loggers = [
            logging.getLogger("miner"),
            logging.getLogger("core"),
        ]
        previous_levels = {tracked: tracked.level for tracked in tracked_loggers}

        handler = GUILogHandler(self.signals.log, self.signals.stats_updated, self)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)

        for tracked in tracked_loggers:
            tracked.addHandler(handler)
            tracked.setLevel(logging.INFO)

        try:
            self.signals.log.emit(
                f"Starting {self.task_type} mining with baseline engine"
            )

            # Run continuous mining
            if hasattr(self.client, "run_continuous_mining"):
                result = self.client.run_continuous_mining(
                    task_type=self.task_type,
                    engine_type="baseline",
                    checkpoint_generations=CHECKPOINT_GENERATIONS,
                )
                self.signals.log.emit(f"Mining session completed: {result}")
            else:
                self.signals.error.emit("Direct client not available")
                return

            # Check stop status
            if self.stop_flag.is_stopped():
                self.signals.log.emit("Mining stopped by user")
            else:
                self.signals.log.emit("Mining session completed")

        except Exception as e:
            self.signals.error.emit(f"Mining error: {e}")
            self.signals.log.emit(f"ERROR: Mining failed: {e}")

        finally:
            # Cleanup logging handlers
            for tracked in tracked_loggers:
                tracked.removeHandler(handler)
                try:
                    tracked.setLevel(previous_levels[tracked])
                except Exception:
                    pass
            self.signals.finished.emit()


class DirectMiningScreen(QWidget):
    """Direct mining screen with configuration, stats, status, and logs.

    Contains all UI and business logic for the Direct Mining tab.
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window

        # Mining state
        self.is_mining = False
        self.mining_task: Optional[DirectMiningTask] = None
        self.thread_pool = QThreadPool()

        # Mining statistics
        self.tasks_completed = 0
        self.successful_submissions = 0
        self.best_score = None

        # Setup UI and load data
        self.setup_ui()
        self._load_mining_stats()

        # Timer for periodic SOTA updates
        self.sota_timer = QTimer()
        self.sota_timer.timeout.connect(self.update_global_sota)

    # ========== Utility Methods ==========

    def _refresh_widget_style(self, widget: QWidget):
        """Force widget style refresh"""
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _update_button_state(self, is_mining: bool):
        """Update mining button appearance based on mining state"""
        if is_mining:
            self.start_mining_btn.update_icon("gui/images/stop.svg")
            self.start_mining_btn.update_text("Stop Mining")
            self.start_mining_btn.setObjectName("stop_mining_button")
            self.start_mining_btn.setStyleSheet("")
        else:
            self.start_mining_btn.update_icon(resource_path("gui/images/play.svg"))
            self.start_mining_btn.update_text("Start Mining")
            self.start_mining_btn.setObjectName("primary_button")
            # Restore the component's own stylesheet (not in global theme)
            self.start_mining_btn.setStyleSheet(PrimaryButton.get_stylesheet())

        self._refresh_widget_style(self.start_mining_btn)

    def _create_auth_headers(self) -> dict:
        """Create authentication headers for API requests"""
        msg = f"auth:{int(time.time())}"
        sig = self.main_window.wallet.hotkey.sign(msg).hex()
        return {
            "X-Key": self.main_window.wallet.hotkey.ss58_address,
            "X-Signature": sig,
            "X-Timestamp": msg,
        }

    # ========== UI Setup Methods ==========

    def setup_ui(self):
        """Initialize the Direct Mining tab content.

        Layout:
        - Mining Configuration (track selector + start button)
        - Miner Stats + Mining Status panels (side-by-side)
        - Mining Logs (auto-expands to fill remaining height)
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Mining configuration section
        config_section = self._create_config_section()
        layout.addWidget(config_section)
        layout.addSpacing(24)

        # Stats and status panels (side-by-side)
        stats_status_container = QWidget()
        stats_status_layout = QHBoxLayout(stats_status_container)
        stats_status_layout.setContentsMargins(0, 0, 0, 0)
        stats_status_layout.setSpacing(24)

        miner_stats = self._create_miner_stats()
        stats_status_layout.addWidget(miner_stats, 1)

        mining_status = self._create_mining_status()
        stats_status_layout.addWidget(mining_status, 1)

        layout.addWidget(stats_status_container)
        layout.addSpacing(24)

        # Logs section (auto-expand to fill remaining height)
        logs_section = self._create_logs_section()
        layout.addWidget(logs_section, 1)

    def _create_config_section(self) -> QWidget:
        """Create mining configuration section with track selector and mining button"""
        section = QWidget()
        section.setObjectName("mining_config_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title row (text + icon)
        title_row = QHBoxLayout()
        title_row.setSpacing(5)
        title_row.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Mining Configuration")
        title.setObjectName("config_section_title")
        title_row.addWidget(title)

        # Config sliders icon (20x20)
        config_icon_label = QLabel()
        config_icon_label.setFixedSize(20, 20)
        renderer = QSvgRenderer(resource_path("gui/images/config_sliders.svg"))
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        config_icon_label.setPixmap(pixmap)
        title_row.addWidget(config_icon_label)

        title_row.addStretch()
        layout.addLayout(title_row)

        # Configuration row (track selector + button)
        config_row = QHBoxLayout()
        config_row.setSpacing(24)

        # Track selector
        task_container = QWidget()
        task_layout = QVBoxLayout(task_container)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(8)

        task_label = QLabel("Track")
        task_label.setObjectName("form_label")
        task_layout.addWidget(task_label)

        self.task_type_combo = SelectInput(height=48)
        self.task_type_map = {"CIFAR-10 Binary": "cifar10_binary"}
        self.task_type_combo.addItems(list(self.task_type_map.keys()))
        self.task_type_combo.setEnabled(False)
        self.task_type_combo.currentTextChanged.connect(
            lambda: self.update_global_sota()
        )
        task_layout.addWidget(self.task_type_combo)

        config_row.addWidget(task_container, 1)

        # Mining button (aligned to bottom of row)
        self.start_mining_btn = PrimaryButton(
            "Start Mining",
            width=200,
            height=48,
            icon_path=resource_path("gui/images/play.svg"),
            icon_size=20,
        )
        self.start_mining_btn.clicked.connect(self._toggle_mining)
        config_row.addWidget(self.start_mining_btn, 0, Qt.AlignmentFlag.AlignBottom)

        layout.addLayout(config_row)
        return section

    # ========== Mining Control Methods ==========

    def _toggle_mining(self):
        """Toggle between starting and stopping mining"""
        if not self.is_mining:
            self._start_mining()
        else:
            self._stop_mining()

    def _start_mining(self):
        """Start the mining process after validation checks"""
        # Check if already mining
        if self.is_mining and self.mining_task:
            self._append_log(
                "ERROR: Mining task still running. Please wait for it to stop."
            )
            return

        # Validate main window and dependencies
        if not self.main_window:
            self._append_log("ERROR: Main window reference not available.")
            return

        if not self.main_window.wallet:
            self._append_log("ERROR: No wallet loaded. Please load a wallet first.")
            return

        if not self.main_window.client:
            self._append_log(
                "ERROR: Client not initialized. Please ensure wallet is properly loaded."
            )
            return

        if not self.main_window.coldkey_address:
            self._append_log(
                "ERROR: No coldkey address provided. Please provide your coldkey address first."
            )
            self.main_window._prompt_for_coldkey_address()
            return

        # Log relay endpoint
        try:
            relay_url = self.main_window._get_relay_endpoint_from_config()
            self._append_log(f"Relay endpoint: {relay_url}")
        except Exception:
            pass

        # Check invite code and send coldkey address
        if not self._check_invite_code():
            self._show_invite_code_modal()
            return

        if not self._send_coldkey_address():
            self._append_log(
                "ERROR: Failed to send coldkey address to relay. Please try again."
            )
            return

        # Update mining state and button
        self.is_mining = True
        self._update_button_state(True)

        # Get task configuration
        task_display = self.task_type_combo.currentText()
        task_type = self.task_type_map.get(task_display, "cifar10_binary")
        stop_flag = StopFlag()

        # Create and configure mining task
        self.mining_task = DirectMiningTask(
            client=self.main_window.client,
            task_type=task_type,
            stop_flag=stop_flag,
            initial_tasks=self.tasks_completed,
            initial_submissions=self.successful_submissions,
            initial_best_score=self.best_score,
        )

        # Connect signals
        self.mining_task.signals.log.connect(self._append_log)
        self.mining_task.signals.error.connect(self._handle_mining_error)
        self.mining_task.signals.finished.connect(self._on_mining_finished)
        self.mining_task.signals.stats_updated.connect(self._update_stats)

        # Start mining task
        self.thread_pool.start(self.mining_task)
        self._append_log(f"Starting mining for task: {task_type}")

        # Update UI state
        self.update_connection_status(True)
        self.update_global_sota()
        self.sota_timer.start(SOTA_UPDATE_INTERVAL_MS)

    def _stop_mining(self):
        """Stop the currently running mining task"""
        self.is_mining = False
        self.sota_timer.stop()
        self._update_button_state(False)

        if self.mining_task:
            self.mining_task.stop()
            self._append_log("Stopping mining...")

    # ========== Validation and Authentication Methods ==========

    def _check_invite_code(self) -> bool:
        """Check if user has a valid invite code"""
        if get_app_config().test_mode:
            self._append_log("Test mode enabled: skipping invite code requirement.")
            return True

        try:
            relay_url = self.main_window._get_relay_endpoint_from_config()
            headers = self._create_auth_headers()

            response = requests.get(
                f"{relay_url}/invitation_code/linked",
                headers=headers,
                timeout=HTTP_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            result = response.json()
            return result.get("data") is not None
        except Exception as e:
            self._append_log(f"Failed to check invite code status: {e}")
            return False

    def _send_coldkey_address(self) -> bool:
        """Send coldkey address to relay server"""
        try:
            relay_url = self.main_window._get_relay_endpoint_from_config()
            headers = self._create_auth_headers()

            response = requests.post(
                f"{relay_url}/coldkey_address/update",
                json={"coldkey_address": self.main_window.coldkey_address},
                headers=headers,
                timeout=HTTP_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            result = response.json()

            if result.get("status") == "success":
                self._append_log("Coldkey address sent to relay successfully")
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
        """Display invite code input modal"""
        relay_url = self.main_window._get_relay_endpoint_from_config()
        coldkey_address = (
            self.main_window.coldkey_address
            if hasattr(self.main_window, "coldkey_address")
            else None
        )
        self.main_window.modal_manager.show_invite_code(
            relay_url=relay_url,
            wallet=self.main_window.wallet,
            coldkey_address=coldkey_address,
        )

    def _on_invite_code_verified(self):
        """Handle successful invite code verification"""
        self._append_log("Invite code verified successfully!")
        self._start_mining()

    # ========== Mining Statistics Methods ==========

    def _load_mining_stats(self):
        """Load mining statistics from persistent storage"""
        stats = load_mining_stats()
        self.tasks_completed = stats.get("tasks_completed", 0)
        self.successful_submissions = stats.get("successful_submissions", 0)
        self.best_score = stats.get("best_score")

        if hasattr(self, "tasks_completed_label"):
            self._update_stats_labels(
                self.tasks_completed, self.successful_submissions, self.best_score
            )

    def _save_mining_stats(self):
        """Save mining statistics to persistent storage"""
        save_mining_stats(
            self.tasks_completed, self.successful_submissions, self.best_score
        )

    def _update_stats_labels(
        self, tasks: int, submissions: int, best_score: Optional[float]
    ):
        """Update statistics display labels"""
        self.tasks_completed_label.setText(str(tasks))
        self.successful_submissions_label.setText(str(submissions))
        if best_score is not None:
            self.best_score_label.setText(f"{best_score:.4f}")
        else:
            self.best_score_label.setText("-")

    def _update_stats(self, stats: dict):
        """Update mining statistics from task signals"""
        tasks = stats.get("tasks_completed", 0)
        submissions = stats.get("successful_submissions", 0)
        best_score = stats.get("best_score")

        self.tasks_completed = tasks
        self.successful_submissions = submissions
        if best_score is not None:
            self.best_score = best_score

        self._update_stats_labels(tasks, submissions, self.best_score)
        self._save_mining_stats()

    # ========== Event Handlers ==========

    def _handle_mining_error(self, error_msg: str):
        """Handle mining task errors"""
        self._append_log(f"ERROR: {error_msg}")

    def _on_mining_finished(self):
        """Handle mining task completion"""
        # Save final statistics
        if self.mining_task:
            final_stats = {
                "tasks_completed": self.mining_task.tasks_completed,
                "successful_submissions": self.mining_task.successful_submissions,
                "best_score": self.mining_task.best_score,
            }
            self._update_stats(final_stats)
            self._save_mining_stats()
            self.mining_task = None

        # Update UI state
        self.is_mining = False
        self.sota_timer.stop()
        self._update_button_state(False)
        self.update_connection_status(False)
        self._append_log("Mining stopped.")

    # ========== UI Component Creation Methods ==========

    def _create_miner_stats(self) -> QWidget:
        """Create miner statistics panel showing tasks and scores"""
        stats = QWidget()
        stats.setObjectName("stats_box")
        stats.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        stats.setFixedHeight(310)

        layout = QVBoxLayout(stats)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Miner Stats")
        title.setObjectName("config_section_title")
        layout.addWidget(title)

        # Stats container
        stats_container = QVBoxLayout()
        stats_container.setSpacing(12)

        # Total Score
        row = self._create_stat_row("Total Score", "0")
        self.total_score_label = row[1]
        stats_container.addLayout(row[0])
        stats_container.addWidget(self._create_divider())

        # Evaluation section
        row = self._create_stat_row("Evaluation Tasks Completed", "0")
        self.eval_tasks_label = row[1]
        stats_container.addLayout(row[0])

        row = self._create_stat_row("Score", "-")
        self.eval_score_label = row[1]
        stats_container.addLayout(row[0])

        # Evolution section
        row = self._create_stat_row("Evolution Tasks Completed", "0")
        self.evo_tasks_label = row[1]
        stats_container.addLayout(row[0])

        row = self._create_stat_row("Score", "-")
        self.evo_score_label = row[1]
        stats_container.addLayout(row[0])

        # Label references for compatibility
        self.tasks_completed_label = self.eval_tasks_label
        self.successful_submissions_label = self.evo_tasks_label
        self.best_score_label = self.eval_score_label

        layout.addLayout(stats_container)
        layout.addStretch()
        return stats

    @staticmethod
    def _create_stat_row(label_text: str, value_text: str):
        """Create a statistics row with label and value"""
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

    @staticmethod
    def _create_divider() -> QWidget:
        """Create divider line"""
        divider = QWidget()
        divider.setObjectName("stat_divider")
        divider.setFixedHeight(1)
        divider.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        return divider

    def _create_status_indicator(
        self, text: str, dot_object_name: str, text_object_name: str
    ):
        """Create a status indicator with dot and text"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        dot = QWidget()
        dot.setObjectName(dot_object_name)
        dot.setFixedSize(6, 6)
        layout.addWidget(dot)

        label = QLabel(text)
        label.setObjectName(text_object_name)
        layout.addWidget(label)

        return container, dot, label

    def _create_mining_status(self) -> QWidget:
        """Create mining status panel showing connection state and runtime info"""
        status = QWidget()
        status.setObjectName("stats_box")
        status.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        status.setFixedHeight(310)

        layout = QVBoxLayout(status)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Title
        title = QLabel("Mining Status")
        title.setObjectName("config_section_title")
        layout.addWidget(title)

        # Status container
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

        # Status indicator
        status_row = QHBoxLayout()
        status_row.setSpacing(0)
        status_label = QLabel("Status")
        status_label.setObjectName("stat_label")
        status_row.addWidget(status_label)
        status_row.addStretch()

        (
            self.status_indicator_container,
            self.status_dot,
            self.mining_status_label,
        ) = self._create_status_indicator(
            "Idle", "status_dot_idle", "status_text_idle"
        )
        status_row.addWidget(self.status_indicator_container)
        status_container.addLayout(status_row)

        # Connection indicator
        connection_row = QHBoxLayout()
        connection_row.setSpacing(0)
        connection_label = QLabel("Connection")
        connection_label.setObjectName("stat_label")
        connection_row.addWidget(connection_label)
        connection_row.addStretch()

        (
            self.connection_indicator_container,
            self.connection_dot,
            self.connection_status_label,
        ) = self._create_status_indicator(
            "Disconnected", "status_dot_disconnected", "status_text_disconnected"
        )
        connection_row.addWidget(self.connection_indicator_container)
        status_container.addLayout(connection_row)

        # Tasks, Runtime, Submissions
        row = self._create_stat_row("Tasks", "0")
        self.tasks_label = row[1]
        status_container.addLayout(row[0])

        row = self._create_stat_row("Runtime", "0h 0m 0s")
        self.runtime_label = row[1]
        status_container.addLayout(row[0])

        row = self._create_stat_row("Submissions", "0")
        self.submissions_label = row[1]
        status_container.addLayout(row[0])

        layout.addLayout(status_container)
        layout.addStretch()
        return status

    def _create_logs_section(self) -> QWidget:
        """Create mining logs section with clear button"""
        section = QWidget()
        section.setObjectName("logs_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header
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

        # Logs text area
        self.logs_text = QTextEdit()
        self.logs_text.setObjectName("logs_text")
        self.logs_text.setReadOnly(True)
        layout.addWidget(self.logs_text)

        return section

    # ========== Log Management Methods ==========

    def _clear_logs(self):
        """Clear all mining logs"""
        self.logs_text.clear()

    def _append_log(self, message: str):
        """Append a message to the mining logs"""
        self.logs_text.append(message)

    # ========== Public Update Methods ==========

    def update_wallet_status(self, wallet_name: str):
        """Update wallet status display"""
        self.wallet_status_label.setText(wallet_name)

    def update_connection_status(self, connected: bool):
        """Update connection and mining status indicators"""
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
        for widget in [
            self.connection_status_label,
            self.connection_dot,
            self.mining_status_label,
            self.status_dot,
        ]:
            self._refresh_widget_style(widget)

    def update_global_sota(self):
        """Fetch and update the global SOTA (State of the Art) score"""
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

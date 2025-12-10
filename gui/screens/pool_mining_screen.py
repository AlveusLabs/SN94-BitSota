from PySide6.QtCore import Qt, QTimer, QRunnable, QThreadPool, QObject, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGridLayout,
    QTextEdit,
    QPushButton,
)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import QPainter
from typing import Optional
import time

from gui.components import PrimaryButton, SecondaryButton
from gui.resource_path import resource_path


class PoolMiningTask(QRunnable):
    class Signals(QObject):
        log = Signal(str)
        error = Signal(str)
        finished = Signal()
        stopping = Signal()
        stats_updated = Signal(dict)

    def __init__(self, pool_client, stop_flag):
        super().__init__()
        self.pool_client = pool_client
        self.stop_flag = stop_flag
        self.signals = self.Signals()
        self.setAutoDelete(True)
        self.start_time = time.time()
        self.tasks_completed = 0

    def stop(self):
        self.stop_flag.stop()
        self.signals.stopping.emit()

    @Slot()
    def run(self):
        try:
            self.signals.log.emit("[Pool] Registering with pool...")

            if not self.pool_client.register():
                self.signals.error.emit("Failed to register with pool")
                return

            self.signals.log.emit("[Pool] Registered successfully, requesting tasks...")

            while not self.stop_flag.is_stopped():
                has_pending = self.pool_client.check_pending_evaluations()

                if has_pending:
                    self.signals.log.emit("[Pool] Pending evaluations found, helping evaluate others...")
                    task = self.pool_client.request_task(task_type="evaluate")
                else:
                    task = self.pool_client.request_task(task_type="evolve" if self.tasks_completed % 3 == 0 else "evaluate")

                if not task:
                    self.signals.log.emit("[Pool] No tasks available, waiting...")
                    time.sleep(10)
                    continue

                task_type = task.get('task_type')
                batch_id = task.get('batch_id')
                algorithms = task.get('algorithms', [])

                if not algorithms:
                    continue

                self.signals.log.emit(f"[Pool] Processing {task_type} task with {len(algorithms)} algorithms")

                if task_type == 'evolve':
                    for algo in algorithms:
                        algorithm_dsl = algo.get('algorithm_dsl')
                        algo_task_type = algo.get('task_type', 'cifar10_binary')
                        input_dim = algo.get('input_dim', 16)

                        if not algorithm_dsl:
                            continue

                        evolved_dsl = self.pool_client.evolve_algorithm(
                            algorithm_dsl, algo_task_type, input_dim, generations=5
                        )

                        if evolved_dsl:
                            parent_ids = [{"id": algo.get('id')}]
                            if self.pool_client.submit_evolution(batch_id, evolved_dsl, parent_ids):
                                self.tasks_completed += 1
                                self.signals.log.emit(f"[Pool] Evolution submitted ({self.tasks_completed} total)")

                elif task_type == 'evaluate':
                    evaluations = []
                    for algo in algorithms:
                        algorithm_dsl = algo.get('algorithm_dsl')
                        algo_task_type = algo.get('task_type', 'cifar10_binary')
                        input_dim = algo.get('input_dim', 16)
                        algorithm_id = algo.get('id')

                        if not algorithm_dsl:
                            continue

                        score = self.pool_client.evaluate_algorithm(algorithm_dsl, algo_task_type, input_dim)

                        if score is not None:
                            evaluations.append({"algorithm_id": algorithm_id, "score": score})

                    if evaluations:
                        if self.pool_client.submit_evaluation(batch_id, evaluations, evaluation_metrics=None):
                            self.tasks_completed += len(evaluations)
                            self.signals.log.emit(f"[Pool] Submitted {len(evaluations)} evaluations")

                runtime = int(time.time() - self.start_time)
                self.signals.stats_updated.emit({
                    'tasks_completed': self.tasks_completed,
                    'runtime': runtime
                })

            if self.stop_flag.is_stopped():
                self.signals.log.emit(f"[Pool] Mining stopped. Completed {self.tasks_completed} tasks")

        except Exception as e:
            self.signals.error.emit(f"Pool mining error: {e}")
            self.signals.log.emit(f"[ERROR] Pool mining failed: {e}")
        finally:
            self.signals.finished.emit()


class ProgressBoxes(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.setMinimumHeight(12)
        self.setMaximumHeight(12)

    def setValue(self, value: int):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        box_count = 29
        box_width = 6
        box_spacing = 2
        total_width = (box_width + box_spacing) * box_count - box_spacing

        start_x = (self.width() - total_width) // 2
        filled_boxes = int((self.value / 100.0) * box_count)

        from PySide6.QtGui import QColor
        for i in range(box_count):
            x = start_x + i * (box_width + box_spacing)
            y = 0

            if i < filled_boxes:
                painter.fillRect(x, y, box_width, 12, QColor(21, 0, 73))
            else:
                painter.fillRect(x, y, box_width, 12, QColor(21, 0, 73, int(0.6 * 255)))


class PoolMiningScreen(QWidget):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.pool_endpoint = "https://api.bitsota.com"
        self.is_mining = False
        self.mining_task: Optional[PoolMiningTask] = None
        self.thread_pool = QThreadPool()
        self.setup_ui()

        # Disabled until pool mining is released
        # self.update_timer = QTimer()
        # self.update_timer.timeout.connect(self._fetch_pool_data)
        # self.update_timer.start(5000)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(24)

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

        pool_status = self._create_pool_status()
        stats_status_layout.addWidget(pool_status, 1)

        content_layout.addLayout(stats_status_layout)

        logs_section = self._create_logs_section()
        content_layout.addWidget(logs_section)

        main_layout.addWidget(content_box)

    def _create_config_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("mining_config_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title_row = QHBoxLayout()
        title = QLabel("Mining Configuration")
        title.setObjectName("section_title")
        title_row.addWidget(title)

        frame_label = QLabel("Frame 2147213699")
        frame_label.setObjectName("form_label")
        frame_label.setStyleSheet("color: #8EFBFF; font-family: 'JetBrains Mono'; font-size: 12px;")
        title_row.addWidget(frame_label)
        title_row.addStretch()

        layout.addLayout(title_row)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(20)

        pool_contribution = self._create_metric_box("Pool Contribution", "0%", show_progress=True)
        metrics_layout.addWidget(pool_contribution, 1)

        pending_rewards = self._create_metric_box("Pending Rewards", "0 $TAO")
        metrics_layout.addWidget(pending_rewards, 1)

        reputation = self._create_metric_box("Reputation", "0.0")
        metrics_layout.addWidget(reputation, 1)

        layout.addLayout(metrics_layout)

        button_row = QHBoxLayout()
        button_row.addStretch()

        self.join_pool_btn = PrimaryButton("Join Pool", width=200, height=48, icon_path=resource_path("gui/images/play.svg"))
        self.join_pool_btn.clicked.connect(self._toggle_mining)
        button_row.addWidget(self.join_pool_btn)

        layout.addLayout(button_row)

        return section

    def _create_metric_box(self, label: str, value: str, show_progress: bool = False) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel(label)
        title.setObjectName("metric_label")
        header.addWidget(title)

        info_icon = QSvgWidget(resource_path("gui/images/info-circle.svg"))
        info_icon.setFixedSize(16, 16)
        header.addWidget(info_icon)
        header.addStretch()

        layout.addLayout(header)

        value_label = QLabel(value)
        value_label.setStyleSheet("""
            color: #150049;
            font-family: 'JetBrains Mono';
            font-size: 24px;
            font-weight: 600;
        """)

        if label == "Pool Contribution":
            self.pool_contribution_label = value_label
        elif label == "Pending Rewards":
            self.pending_rewards_label = value_label
        elif label == "Reputation":
            self.reputation_label = value_label

        layout.addWidget(value_label)

        if show_progress:
            self.progress_boxes = ProgressBoxes()
            self.progress_boxes.setValue(0)
            layout.addWidget(self.progress_boxes)

        return box

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

        label = QLabel("Total Score")
        label.setObjectName("stat_label")
        stats_grid.addWidget(label, 0, 0)
        self.total_score_label = QLabel("0")
        self.total_score_label.setObjectName("stat_value")
        self.total_score_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        stats_grid.addWidget(self.total_score_label, 0, 1)

        label = QLabel("Evaluation Tasks Completed")
        label.setObjectName("stat_label")
        stats_grid.addWidget(label, 1, 0)
        self.eval_tasks_label = QLabel("0")
        self.eval_tasks_label.setObjectName("stat_value")
        self.eval_tasks_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        stats_grid.addWidget(self.eval_tasks_label, 1, 1)

        label = QLabel("Evolution Tasks Completed")
        label.setObjectName("stat_label")
        stats_grid.addWidget(label, 2, 0)
        self.evol_tasks_label = QLabel("0")
        self.evol_tasks_label.setObjectName("stat_value")
        self.evol_tasks_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        stats_grid.addWidget(self.evol_tasks_label, 2, 1)

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

        label = QLabel("Wallet")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 0, 0)
        self.wallet_status_label = QLabel("Not Connected")
        self.wallet_status_label.setObjectName("stat_value")
        self.wallet_status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.wallet_status_label, 0, 1)

        label = QLabel("Status")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 1, 0)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("stat_value")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.status_label, 1, 1)

        label = QLabel("Connection")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 2, 0)
        self.connection_status_label = QLabel("Disconnected")
        self.connection_status_label.setObjectName("stat_value")
        self.connection_status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.connection_status_label, 2, 1)

        label = QLabel("Runtime")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 3, 0)
        self.runtime_label = QLabel("-")
        self.runtime_label.setObjectName("stat_value")
        self.runtime_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.runtime_label, 3, 1)

        label = QLabel("Resource Contributed")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 4, 0)
        self.resource_contributed_label = QLabel("0")
        self.resource_contributed_label.setObjectName("stat_value")
        self.resource_contributed_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.resource_contributed_label, 4, 1)

        layout.addLayout(status_grid)
        layout.addStretch()

        return status

    def _create_pool_status(self) -> QWidget:
        status = QWidget()
        status.setObjectName("stats_box")
        layout = QVBoxLayout(status)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Pool Status")
        title.setObjectName("section_title")
        layout.addWidget(title)

        status_grid = QGridLayout()
        status_grid.setSpacing(12)

        label = QLabel("SOTA")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 0, 0)
        self.sota_label = QLabel("-")
        self.sota_label.setObjectName("stat_value")
        self.sota_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.sota_label, 0, 1)

        label = QLabel("Total Resource Contributed")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 1, 0)
        self.total_resource_label = QLabel("-")
        self.total_resource_label.setObjectName("stat_value")
        self.total_resource_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.total_resource_label, 1, 1)

        label = QLabel("Active Miners")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 2, 0)
        self.active_miners_label = QLabel("-")
        self.active_miners_label.setObjectName("stat_value")
        self.active_miners_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.active_miners_label, 2, 1)

        label = QLabel("Total Rewards Distributed")
        label.setObjectName("stat_label")
        status_grid.addWidget(label, 3, 0)
        self.total_rewards_label = QLabel("-")
        self.total_rewards_label.setObjectName("stat_value")
        self.total_rewards_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_grid.addWidget(self.total_rewards_label, 3, 1)

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

    def _toggle_mining(self):
        if not self.is_mining:
            self._start_mining()
        else:
            self._stop_mining()

    def _start_mining(self):
        if not self.main_window:
            self._append_log("ERROR: Main window reference not available.")
            return

        if not self.main_window.wallet:
            self._append_log("ERROR: No wallet loaded. Please load a wallet first.")
            return

        try:
            from miner.pool_client import PoolClient
            from gui.stop_flag import StopFlag

            wallet = self.main_window.wallet
            pool_client = PoolClient(self.pool_endpoint, wallet.hotkey.ss58_address, wallet)

            stop_flag = StopFlag()

            self.mining_task = PoolMiningTask(pool_client=pool_client, stop_flag=stop_flag)
            self.mining_task.signals.log.connect(self._append_log)
            self.mining_task.signals.error.connect(self._handle_mining_error)
            self.mining_task.signals.finished.connect(self._on_mining_finished)
            self.mining_task.signals.stats_updated.connect(self._update_stats)

            self.thread_pool.start(self.mining_task)

            self.is_mining = True
            self.join_pool_btn.update_icon("gui/images/stop.svg")
            self.join_pool_btn.update_text("Leave Pool")
            self.join_pool_btn.setObjectName("stop_mining_button")
            self.join_pool_btn.setStyleSheet("")
            self.join_pool_btn.style().unpolish(self.join_pool_btn)
            self.join_pool_btn.style().polish(self.join_pool_btn)

            self.status_label.setText("Mining")
            self.connection_status_label.setText("Connected")
            self.connection_status_label.setStyleSheet("color: #51cf66;")

            wallet_short = wallet.hotkey.ss58_address[:8] + "..." + wallet.hotkey.ss58_address[-8:]
            self.wallet_status_label.setText(wallet_short)

            self._append_log("[Pool] Starting pool mining...")

        except Exception as e:
            self._append_log(f"ERROR: Failed to start pool mining: {e}")

    def _stop_mining(self):
        self.is_mining = False
        self.join_pool_btn.update_icon(resource_path("gui/images/play.svg"))
        self.join_pool_btn.update_text("Join Pool")
        self.join_pool_btn.setObjectName("primary_button")
        self.join_pool_btn.setStyleSheet("")
        self.join_pool_btn.style().unpolish(self.join_pool_btn)
        self.join_pool_btn.style().polish(self.join_pool_btn)

        if self.mining_task:
            self.mining_task.stop()
            self._append_log("[Pool] Stopping mining...")

    def _handle_mining_error(self, error_msg: str):
        self._append_log(f"ERROR: {error_msg}")

    def _on_mining_finished(self):
        self.is_mining = False
        self.join_pool_btn.update_icon(resource_path("gui/images/play.svg"))
        self.join_pool_btn.update_text("Join Pool")
        self.join_pool_btn.setObjectName("primary_button")
        self.join_pool_btn.setStyleSheet("")
        self.join_pool_btn.style().unpolish(self.join_pool_btn)
        self.join_pool_btn.style().polish(self.join_pool_btn)

        self.status_label.setText("Stopped")
        self.connection_status_label.setText("Disconnected")
        self.connection_status_label.setStyleSheet("")
        self.mining_task = None
        self._append_log("[Pool] Mining stopped.")

    def _update_stats(self, stats: dict):
        tasks_completed = stats.get('tasks_completed', 0)
        runtime = stats.get('runtime', 0)

        self.resource_contributed_label.setText(str(tasks_completed))

        hours = runtime // 3600
        minutes = (runtime % 3600) // 60
        seconds = runtime % 60
        self.runtime_label.setText(f"{hours}h {minutes}m {seconds}s")

    def _fetch_pool_data(self):
        if not self.main_window or not self.main_window.wallet:
            return

        try:
            from miner.pool_client import PoolClient

            wallet = self.main_window.wallet
            pool_client = PoolClient(self.pool_endpoint, wallet.hotkey.ss58_address, wallet)

            balance_data = pool_client.get_balance()
            if balance_data:
                pending_rao = balance_data.get('pending_rao', 0)
                reputation = balance_data.get('reputation', 0.0)

                pending_tao = pending_rao / 1e9
                self.pending_rewards_label.setText(f"{pending_tao:.2f} $TAO")
                self.reputation_label.setText(f"{reputation:.1f}")

                if reputation > 0:
                    contribution_pct = min(100, (reputation / 1000) * 100)
                    self.pool_contribution_label.setText(f"{contribution_pct:.2f}%")
                    self.progress_boxes.setValue(int(contribution_pct))

        except Exception as e:
            pass

    def update_wallet_status(self, wallet_name: str):
        if wallet_name:
            wallet_short = wallet_name[:8] + "..." + wallet_name[-8:] if len(wallet_name) > 16 else wallet_name
            self.wallet_status_label.setText(wallet_short)

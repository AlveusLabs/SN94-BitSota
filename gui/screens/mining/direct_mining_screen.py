from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.components import PrimaryButton, SelectInput
from gui.app_config import get_app_config
from gui.resource_path import resource_path
from gui.screens.mining_screen import MiningScreen as LegacyMiningScreen


class _HiddenSelection:
    def __init__(self, text: str):
        self._text = str(text)

    def currentText(self) -> str:
        return self._text

    def setCurrentText(self, text: str):
        self._text = str(text)


class DirectMiningScreen(LegacyMiningScreen):
    """Refactor UI shell backed by the current sidecar/worker launcher."""

    def __init__(self, main_window=None, parent=None):
        self._runtime_started_at: float | None = None
        super().__init__(main_window=main_window, parent=parent)
        self._runtime_timer = QTimer(self)
        self._runtime_timer.timeout.connect(self._update_runtime)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        layout.addWidget(self._create_config_section())

        stats_row = QHBoxLayout()
        stats_row.setSpacing(24)
        stats_row.addWidget(self._create_miner_stats(), 1)
        stats_row.addWidget(self._create_mining_status(), 1)
        layout.addLayout(stats_row)

        layout.addWidget(self._create_logs_section(), 1)

    def _create_config_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("mining_config_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Mining Configuration")
        title.setObjectName("config_section_title")
        title_row.addWidget(title)

        icon_label = QLabel()
        icon_label.setFixedSize(20, 20)
        renderer = QSvgRenderer(resource_path("gui/images/config_sliders.svg"))
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        icon_label.setPixmap(pixmap)
        title_row.addWidget(icon_label)
        title_row.addStretch()
        layout.addLayout(title_row)

        config_row = QHBoxLayout()
        config_row.setSpacing(24)

        task_container = QWidget()
        task_layout = QVBoxLayout(task_container)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(8)

        task_label = QLabel("Track")
        task_label.setObjectName("form_label")
        task_layout.addWidget(task_label)

        self.task_type_combo = SelectInput(height=48)
        self.task_type_map = {
            "CIFAR-10 Binary": "cifar10_binary",
            "MNIST Binary": "mnist_binary",
            "Scalar Linear": "scalar_linear",
        }
        self.task_type_combo.addItems(list(self.task_type_map.keys()))
        self.task_type_combo.currentTextChanged.connect(self.update_global_sota)
        task_layout.addWidget(self.task_type_combo)
        config_row.addWidget(task_container, 1)

        default_workers = max(1, int(getattr(get_app_config(), "miner_workers", 1) or 1))
        self.workers_combo = _HiddenSelection(str(default_workers))

        self.start_mining_btn = PrimaryButton(
            "Start Mining",
            width=200,
            height=48,
            icon_path=resource_path("gui/images/play.svg"),
        )
        self.start_mining_btn.clicked.connect(self._toggle_mining)
        config_row.addWidget(self.start_mining_btn, 0, Qt.AlignmentFlag.AlignBottom)

        layout.addLayout(config_row)
        return section

    def _create_miner_stats(self) -> QWidget:
        stats = QWidget()
        stats.setObjectName("stats_box")
        stats.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        stats.setFixedHeight(300)

        layout = QVBoxLayout(stats)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Miner Stats")
        title.setObjectName("config_section_title")
        layout.addWidget(title)

        rows = QVBoxLayout()
        rows.setSpacing(12)

        row = self._create_stat_row("Tasks Completed", "0")
        self.tasks_completed_label = row[1]
        rows.addLayout(row[0])
        rows.addWidget(self._create_divider())

        row = self._create_stat_row("Successful Submissions", "0")
        self.successful_submissions_label = row[1]
        rows.addLayout(row[0])

        row = self._create_stat_row("Local SOTA", "-")
        self.best_score_label = row[1]
        rows.addLayout(row[0])

        layout.addLayout(rows)
        layout.addStretch()
        return stats

    def _create_mining_status(self) -> QWidget:
        status = QWidget()
        status.setObjectName("stats_box")
        status.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        status.setFixedHeight(300)

        layout = QVBoxLayout(status)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("Mining Status")
        title.setObjectName("config_section_title")
        layout.addWidget(title)

        rows = QVBoxLayout()
        rows.setSpacing(12)

        row = self._create_stat_row("Global SOTA", "-")
        self.global_sota_label = row[1]
        rows.addLayout(row[0])
        rows.addWidget(self._create_divider())

        row = self._create_stat_row("Wallet", "Not Connected")
        self.wallet_status_label = row[1]
        rows.addLayout(row[0])

        row = QHBoxLayout()
        row.addWidget(_build_stat_label("Status"))
        row.addStretch()
        self.status_dot, self.mining_status_label, status_container = self._create_status_indicator(
            "Idle",
            "status_dot_idle",
            "status_text_idle",
        )
        row.addWidget(status_container)
        rows.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(_build_stat_label("Connection"))
        row.addStretch()
        self.connection_dot, self.connection_status_label, conn_container = self._create_status_indicator(
            "Disconnected",
            "status_dot_disconnected",
            "status_text_disconnected",
        )
        row.addWidget(conn_container)
        rows.addLayout(row)

        row = self._create_stat_row("Runtime", "0h 0m 0s")
        self.runtime_label = row[1]
        rows.addLayout(row[0])

        layout.addLayout(rows)
        layout.addStretch()
        return status

    def _create_logs_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("logs_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel("Mining Logs")
        title.setObjectName("logs_title")
        header.addWidget(title)
        header.addStretch()

        self.clear_logs_btn = QPushButton("Clear Logs")
        self.clear_logs_btn.setObjectName("clear_logs_button")
        self.clear_logs_btn.setFixedHeight(32)
        self.clear_logs_btn.clicked.connect(self._clear_logs)
        header.addWidget(self.clear_logs_btn)

        layout.addLayout(header)

        self.logs_text = QTextEdit()
        self.logs_text.setObjectName("logs_text")
        self.logs_text.setReadOnly(True)
        self.logs_text.document().setMaximumBlockCount(5000)
        layout.addWidget(self.logs_text)

        return section

    @staticmethod
    def _create_stat_row(label_text: str, value_text: str):
        row_layout = QHBoxLayout()
        row_layout.setSpacing(0)

        row_layout.addWidget(_build_stat_label(label_text))
        row_layout.addStretch()

        value = QLabel(value_text)
        value.setObjectName("stat_value")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        row_layout.addWidget(value)
        return row_layout, value

    @staticmethod
    def _create_divider() -> QWidget:
        divider = QWidget()
        divider.setObjectName("stat_divider")
        divider.setFixedHeight(1)
        divider.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        return divider

    @staticmethod
    def _refresh_widget_style(widget: QWidget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _create_status_indicator(self, text: str, dot_name: str, text_name: str):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        dot = QWidget()
        dot.setObjectName(dot_name)
        dot.setFixedSize(8, 8)
        dot.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout.addWidget(dot)

        label = QLabel(text)
        label.setObjectName(text_name)
        layout.addWidget(label)
        return dot, label, container

    def _start_mining(self):
        was_mining = self.is_mining
        super()._start_mining()
        if not was_mining and self.is_mining:
            self._runtime_started_at = time.time()
            self._update_runtime()
            self._runtime_timer.start(1000)

    def _stop_mining(self):
        super()._stop_mining()
        self._runtime_timer.stop()

    def _on_mining_finished(self):
        super()._on_mining_finished()
        self._runtime_timer.stop()

    def _update_runtime(self):
        if self._runtime_started_at is None:
            self.runtime_label.setText("0h 0m 0s")
            return
        elapsed = max(0, int(time.time() - self._runtime_started_at))
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        self.runtime_label.setText(f"{hours}h {minutes}m {seconds}s")

    def _update_stats(self, stats: dict):
        tasks = int(stats.get("tasks_completed", 0) or 0)
        submissions = int(stats.get("successful_submissions", 0) or 0)
        best_score = stats.get("best_score")

        self.tasks_completed = tasks
        self.successful_submissions = submissions
        if best_score is not None:
            self.best_score = best_score

        self.tasks_completed_label.setText(str(tasks))
        self.successful_submissions_label.setText(str(submissions))
        if self.best_score is None:
            self.best_score_label.setText("-")
        else:
            self.best_score_label.setText(f"{float(self.best_score):.4f}")
        self._save_mining_stats()

    def update_wallet_status(self, wallet_name: str):
        self.wallet_status_label.setText(wallet_name or "Not Connected")

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

        for widget in (
            self.connection_status_label,
            self.connection_dot,
            self.mining_status_label,
            self.status_dot,
        ):
            self._refresh_widget_style(widget)


def _build_stat_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("stat_label")
    return label

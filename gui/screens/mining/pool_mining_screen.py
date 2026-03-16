from __future__ import annotations

import os
import time
from typing import Optional
from urllib.parse import urlparse

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.app_config import get_app_config
from gui.components import PrimaryButton, SecondaryButton
from gui.resource_path import resource_path
from gui.screens.mining_screen import MiningScreen as LegacyMiningScreen


POOL_GRID_COLUMNS = 2
POOL_GRID_SPACING = 24


def _configured_pools() -> list[dict]:
    cfg = get_app_config()
    endpoint = str(getattr(cfg, "pool_endpoint", "") or "").strip()
    parsed = urlparse(endpoint) if endpoint else None
    host = parsed.netloc or endpoint or "Not configured"
    backend = str(os.getenv("BITSOTA_MINER_BACKEND", "") or "").strip().lower()
    backend_label = "C++" if backend in {"cpp", "cpp_baseline", "automl_zero_cpp"} else "Python"
    workers = max(1, int(getattr(cfg, "miner_workers", 1) or 1))

    return [
        {
            "id": "pool_lease",
            "name": "Lease Pool",
            "mode": "pool_lease",
            "mode_label": "Lease coordinator",
            "endpoint": host,
            "backend": backend_label,
            "workers": workers,
            "recommended": True,
        },
        {
            "id": "pool_task",
            "name": "Task Pool",
            "mode": "pool",
            "mode_label": "Task batches",
            "endpoint": host,
            "backend": backend_label,
            "workers": workers,
            "recommended": False,
        },
    ]


class PoolCard(QWidget):
    join_clicked = Signal(dict)

    def __init__(self, pool_data: dict, parent=None):
        super().__init__(parent)
        self.pool_data = dict(pool_data)
        self.setObjectName("pool_card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title_row = QHBoxLayout()
        title = QLabel(self.pool_data.get("name", "Pool"))
        title.setObjectName("config_section_title")
        title_row.addWidget(title)
        if self.pool_data.get("recommended"):
            badge = QLabel("Recommended")
            badge.setObjectName("pool_recommended_badge")
            title_row.addWidget(badge)
        title_row.addStretch()
        layout.addLayout(title_row)

        rows = QVBoxLayout()
        rows.setSpacing(12)
        _add_stat_row(rows, "Mode", self.pool_data.get("mode_label", "-"))
        rows.addWidget(_create_divider())
        _add_stat_row(rows, "Endpoint", self.pool_data.get("endpoint", "-"))
        _add_stat_row(rows, "Backend", self.pool_data.get("backend", "-"))
        _add_stat_row(rows, "Workers", str(self.pool_data.get("workers", "-")))
        layout.addLayout(rows)
        layout.addStretch()

        join_btn = QPushButton("Join Pool")
        join_btn.setObjectName("pool_join_btn")
        join_btn.setFixedHeight(48)
        join_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        join_btn.clicked.connect(lambda: self.join_clicked.emit(self.pool_data))
        layout.addWidget(join_btn)


class _HiddenTaskSelection:
    def __init__(self, label: str):
        self._label = str(label)

    def currentText(self) -> str:
        return self._label

    def setCurrentText(self, label: str):
        self._label = str(label)


class PoolDetailScreen(LegacyMiningScreen):
    switch_pool_requested = Signal()

    def __init__(self, pool_data: dict, main_window=None, parent=None):
        self.pool_data = dict(pool_data)
        self._runtime_started_at: float | None = None
        super().__init__(main_window=main_window, parent=parent)
        self._runtime_timer = QTimer(self)
        self._runtime_timer.timeout.connect(self._update_runtime)

    def setup_ui(self):
        self.task_type_map = {
            str(self.pool_data.get("mode_label", "Pool")): str(self.pool_data.get("mode") or "pool_lease")
        }

        only_label = next(iter(self.task_type_map.keys()))
        self.task_type_combo = _HiddenTaskSelection(only_label)
        self.workers_combo = _HiddenTaskSelection(str(self.pool_data.get("workers", 1)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        layout.addWidget(self._create_config_section())

        stats_row = QHBoxLayout()
        stats_row.setSpacing(24)
        stats_row.addWidget(self._create_miner_stats(), 1)
        stats_row.addWidget(self._create_mining_status(), 1)
        stats_row.addWidget(self._create_pool_status(), 1)
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

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(24)
        metrics_row.addWidget(self._create_metric_box("Mode", self.pool_data.get("mode_label", "-")), 1)
        metrics_row.addWidget(self._create_metric_box("Backend", self.pool_data.get("backend", "-")), 1)
        metrics_row.addWidget(self._create_metric_box("Workers", str(self.pool_data.get("workers", "-"))), 1)
        layout.addLayout(metrics_row)

        pool_row = QHBoxLayout()
        pool_row.setSpacing(24)

        pool_container = QWidget()
        pool_layout = QVBoxLayout(pool_container)
        pool_layout.setContentsMargins(0, 0, 0, 0)
        pool_layout.setSpacing(8)

        pool_label = QLabel("Current Pool")
        pool_label.setObjectName("form_label")
        pool_layout.addWidget(pool_label)

        pool_display = QWidget()
        pool_display.setObjectName("pool_display")
        pool_display.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pool_display.setFixedHeight(48)
        pool_display_layout = QHBoxLayout(pool_display)
        pool_display_layout.setContentsMargins(14, 8, 14, 8)
        pool_display_layout.setSpacing(8)

        pool_icon = QLabel()
        pool_icon.setFixedSize(20, 20)
        renderer = QSvgRenderer(resource_path("gui/images/pool_nodes.svg"))
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        pool_icon.setPixmap(pixmap)
        pool_display_layout.addWidget(pool_icon)

        pool_name = QLabel(self.pool_data.get("name", "Pool"))
        pool_name.setObjectName("pool_display_name")
        pool_display_layout.addWidget(pool_name)
        pool_display_layout.addStretch()
        pool_layout.addWidget(pool_display)
        pool_row.addWidget(pool_container, 1)

        buttons_wrapper = QWidget()
        buttons_layout = QVBoxLayout(buttons_wrapper)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(0)
        buttons_layout.addStretch()

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(10)
        self.switch_pool_btn = SecondaryButton("Switch Pool", width=200, height=48)
        self.switch_pool_btn.clicked.connect(self._on_switch_pool)
        buttons_row.addWidget(self.switch_pool_btn)

        self.start_mining_btn = PrimaryButton(
            "Start Mining",
            width=200,
            height=48,
            icon_path=resource_path("gui/images/play.svg"),
        )
        self.start_mining_btn.clicked.connect(self._toggle_mining)
        buttons_row.addWidget(self.start_mining_btn)
        buttons_layout.addLayout(buttons_row)
        pool_row.addWidget(buttons_wrapper)

        layout.addLayout(pool_row)
        return section

    def _create_metric_box(self, label: str, value: str) -> QWidget:
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(4)
        title = QLabel(label)
        title.setObjectName("metric_label")
        header.addWidget(title)
        header.addStretch()
        box_layout.addLayout(header)

        value_label = QLabel(value)
        value_label.setObjectName("metric_value")
        value_label.setWordWrap(True)
        box_layout.addWidget(value_label)
        return box

    def _create_miner_stats(self) -> QWidget:
        panel = _create_stats_panel("Miner Stats")
        content = panel.layout().itemAt(1).layout()
        self.tasks_completed_label = _add_stat_row(content, "Tasks Completed", "0")
        content.addWidget(_create_divider())
        self.successful_submissions_label = _add_stat_row(content, "Lease Submissions", "0")
        self.best_score_label = _add_stat_row(content, "Local Best", "-")
        return panel

    def _create_mining_status(self) -> QWidget:
        panel = _create_stats_panel("Mining Status")
        content = panel.layout().itemAt(1).layout()
        self.wallet_status_label = _add_stat_row(content, "Wallet", "Not Connected")

        row = QHBoxLayout()
        row.addWidget(_build_stat_label("Status"))
        row.addStretch()
        self.status_dot, self.mining_status_label, status_container = self._create_status_indicator(
            "Idle",
            "status_dot_idle",
            "status_text_idle",
        )
        row.addWidget(status_container)
        content.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(_build_stat_label("Connection"))
        row.addStretch()
        self.connection_dot, self.connection_status_label, conn_container = self._create_status_indicator(
            "Disconnected",
            "status_dot_disconnected",
            "status_text_disconnected",
        )
        row.addWidget(conn_container)
        content.addLayout(row)

        self.runtime_label = _add_stat_row(content, "Runtime", "0h 0m 0s")
        self.resource_contributed_label = _add_stat_row(content, "Resource Contributed", "0")
        return panel

    def _create_pool_status(self) -> QWidget:
        panel = _create_stats_panel("Pool Status")
        content = panel.layout().itemAt(1).layout()
        self.global_sota_label = _add_stat_row(content, "SOTA", "-")
        content.addWidget(_create_divider())
        self.endpoint_label = _add_stat_row(content, "Endpoint", self.pool_data.get("endpoint", "-"))
        self.backend_label = _add_stat_row(content, "Backend", self.pool_data.get("backend", "-"))
        self.mode_label = _add_stat_row(content, "Mode", self.pool_data.get("mode_label", "-"))
        return panel

    def _create_logs_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("logs_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title = QLabel("Mining Logs")
        title.setObjectName("logs_title")
        header.addWidget(title)
        header.addStretch()

        clear_btn = QPushButton("Clear Logs")
        clear_btn.setObjectName("clear_logs_button")
        clear_btn.setFixedHeight(32)
        clear_btn.clicked.connect(self._clear_logs)
        header.addWidget(clear_btn)
        layout.addLayout(header)

        self.logs_text = QTextEdit()
        self.logs_text.setObjectName("logs_text")
        self.logs_text.setReadOnly(True)
        self.logs_text.document().setMaximumBlockCount(5000)
        layout.addWidget(self.logs_text)
        return section

    def _clear_logs(self):
        self.logs_text.clear()

    def _append_log(self, message: str):
        self.logs_text.append(message)

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

    @staticmethod
    def _refresh_widget_style(widget: QWidget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _on_switch_pool(self):
        if self.is_mining:
            self._append_log("[pool] stop mining before switching pools")
            return
        self.switch_pool_requested.emit()

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
        self.resource_contributed_label.setText(str(tasks))
        if self.best_score is None:
            self.best_score_label.setText("-")
        else:
            self.best_score_label.setText(f"{float(self.best_score):.4f}")

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


class PoolMiningScreen(QWidget):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.pool_cards: list[PoolCard] = []
        self.pool_detail_view: Optional[PoolDetailScreen] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.pool_list_view = self._create_pool_list_view()
        self.stack.addWidget(self.pool_list_view)
        layout.addWidget(self.stack)

    def _create_pool_list_view(self) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        self.pool_grid = QGridLayout()
        self.pool_grid.setHorizontalSpacing(POOL_GRID_SPACING)
        self.pool_grid.setVerticalSpacing(POOL_GRID_SPACING)
        scroll_layout.addLayout(self.pool_grid)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        container_layout.addWidget(scroll)

        self._load_pools(_configured_pools())
        return container

    def _load_pools(self, pools: list[dict]):
        for card in self.pool_cards:
            card.deleteLater()
        self.pool_cards.clear()

        for idx, pool_data in enumerate(pools):
            card = PoolCard(pool_data)
            card.join_clicked.connect(self._on_join_pool)
            row = idx // POOL_GRID_COLUMNS
            col = idx % POOL_GRID_COLUMNS
            self.pool_grid.addWidget(card, row, col)
            self.pool_cards.append(card)

    def _on_join_pool(self, pool_data: dict):
        if self.pool_detail_view is not None:
            self.stack.removeWidget(self.pool_detail_view)
            self.pool_detail_view.deleteLater()

        self.pool_detail_view = PoolDetailScreen(pool_data=pool_data, main_window=self.main_window)
        self.pool_detail_view.switch_pool_requested.connect(self.show_pool_list)
        self.stack.addWidget(self.pool_detail_view)
        self.stack.setCurrentWidget(self.pool_detail_view)

        wallet = getattr(self.main_window, "wallet", None) if self.main_window else None
        if wallet is not None:
            self.pool_detail_view.update_wallet_status(getattr(wallet, "name", "") or "Connected")

    def show_pool_list(self):
        self.stack.setCurrentWidget(self.pool_list_view)

    def update_wallet_status(self, wallet_name: str):
        if self.pool_detail_view is not None:
            self.pool_detail_view.update_wallet_status(wallet_name)


def _create_stats_panel(title_text: str) -> QWidget:
    panel = QWidget()
    panel.setObjectName("stats_box")
    panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(16)

    title = QLabel(title_text)
    title.setObjectName("config_section_title")
    layout.addWidget(title)

    stats_container = QVBoxLayout()
    stats_container.setSpacing(12)
    layout.addLayout(stats_container)
    layout.addStretch()
    return panel


def _add_stat_row(layout: QVBoxLayout, label_text: str, value_text: str) -> QLabel:
    row = QHBoxLayout()
    row.setSpacing(0)
    row.addWidget(_build_stat_label(label_text))
    row.addStretch()

    value = QLabel(str(value_text))
    value.setObjectName("stat_value")
    value.setAlignment(Qt.AlignmentFlag.AlignRight)
    row.addWidget(value)
    layout.addLayout(row)
    return value


def _build_stat_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("stat_label")
    return label


def _create_divider() -> QWidget:
    divider = QWidget()
    divider.setObjectName("stat_divider")
    divider.setFixedHeight(1)
    divider.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    return divider

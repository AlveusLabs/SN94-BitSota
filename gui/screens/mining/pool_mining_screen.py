# PySide6 imports
from PySide6.QtCore import Qt, QTimer, QRunnable, QThreadPool, QObject, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGridLayout,
    QScrollArea,
    QStackedWidget,
    QPushButton,
    QTextEdit,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter, QColor

# Standard library imports
from typing import Optional
import time

# Local application imports
from gui.components import PrimaryButton, SecondaryButton
from gui.theme import BitSOTATheme
from gui.resource_path import resource_path

# Grid layout constants
POOL_GRID_COLUMNS = 3
POOL_GRID_SPACING = 24

# Placeholder pool data (will be replaced by API data)
PLACEHOLDER_POOLS = [
    {"id": "pool1", "name": "Pool 1", "sota": 3, "miners": 2, "staked": 88.8, "rewards": 2},
    {"id": "pool2", "name": "Pool 2", "sota": 2, "miners": 12, "staked": 24.8, "rewards": 3},
    {"id": "pool3", "name": "Pool 3", "sota": 3, "miners": 11, "staked": 121.1, "rewards": 12},
    {"id": "pool4", "name": "Pool 4", "sota": 9, "miners": 3, "staked": 14.2, "rewards": 11},
]

# Mock data for pool detail page
MOCK_POOL_DETAIL = {
    "pool_contribution": "10%",
    "pool_contribution_value": 10,
    "pending_rewards": "0 $TAO",
    "reputation": "0",
    "miner_stats": {
        "total_score": "182.8",
        "eval_tasks": "2",
        "eval_score": "88.8",
        "evo_tasks": "2",
        "evo_score": "82.8",
    },
    "mining_status": {
        "wallet": "wallet_asdasdasdas",
        "status": "Ready",
        "connection": "Disconnected",
        "runtime": "-",
        "resource_contributed": "2",
        "last_payout": "Feb 1, 2025",
        "next_payout": "Feb 1, 2025",
    },
    "pool_status": {
        "sota": "4",
        "total_resource": "-",
        "active_miners": "2",
        "total_rewards": "2",
    },
}


class PoolMiningTask(QRunnable):
    """Background task for pool mining operations"""

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
                    self.signals.log.emit(
                        "[Pool] Pending evaluations found, helping evaluate others..."
                    )
                    task = self.pool_client.request_task(task_type="evaluate")
                else:
                    task = self.pool_client.request_task(
                        task_type="evolve"
                        if self.tasks_completed % 3 == 0
                        else "evaluate"
                    )

                if not task:
                    self.signals.log.emit("[Pool] No tasks available, waiting...")
                    time.sleep(10)
                    continue

                task_type = task.get("task_type")
                batch_id = task.get("batch_id")
                algorithms = task.get("algorithms", [])

                if not algorithms:
                    continue

                self.signals.log.emit(
                    f"[Pool] Processing {task_type} task with {len(algorithms)} algorithms"
                )

                if task_type == "evolve":
                    for algo in algorithms:
                        algorithm_dsl = algo.get("algorithm_dsl")
                        algo_task_type = algo.get("task_type", "cifar10_binary")
                        input_dim = algo.get("input_dim", 16)

                        if not algorithm_dsl:
                            continue

                        evolved_dsl = self.pool_client.evolve_algorithm(
                            algorithm_dsl, algo_task_type, input_dim, generations=5
                        )

                        if evolved_dsl:
                            parent_ids = [{"id": algo.get("id")}]
                            if self.pool_client.submit_evolution(
                                batch_id, evolved_dsl, parent_ids
                            ):
                                self.tasks_completed += 1
                                self.signals.log.emit(
                                    f"[Pool] Evolution submitted ({self.tasks_completed} total)"
                                )

                elif task_type == "evaluate":
                    evaluations = []
                    for algo in algorithms:
                        algorithm_dsl = algo.get("algorithm_dsl")
                        algo_task_type = algo.get("task_type", "cifar10_binary")
                        input_dim = algo.get("input_dim", 16)
                        algorithm_id = algo.get("id")

                        if not algorithm_dsl:
                            continue

                        score = self.pool_client.evaluate_algorithm(
                            algorithm_dsl, algo_task_type, input_dim
                        )

                        if score is not None:
                            evaluations.append(
                                {"algorithm_id": algorithm_id, "score": score}
                            )

                    if evaluations:
                        if self.pool_client.submit_evaluation(
                            batch_id, evaluations, evaluation_metrics=None
                        ):
                            self.tasks_completed += len(evaluations)
                            self.signals.log.emit(
                                f"[Pool] Submitted {len(evaluations)} evaluations"
                            )

                runtime = int(time.time() - self.start_time)
                self.signals.stats_updated.emit(
                    {"tasks_completed": self.tasks_completed, "runtime": runtime}
                )

            if self.stop_flag.is_stopped():
                self.signals.log.emit(
                    f"[Pool] Mining stopped. Completed {self.tasks_completed} tasks"
                )

        except Exception as e:
            self.signals.error.emit(f"Pool mining error: {e}")
            self.signals.log.emit(f"[ERROR] Pool mining failed: {e}")
        finally:
            self.signals.finished.emit()


# =============================================================================
# Pool Card (used in pool list grid)
# =============================================================================


class PoolCard(QWidget):
    """Pool card widget showing pool info and a Join Pool button."""

    join_clicked = Signal(dict)

    @staticmethod
    def get_stylesheet() -> str:
        """Get the stylesheet for PoolCard component."""
        return f"""
            QWidget#pool_card {{
                background-color: rgba(21, 0, 73, 0.05);
                border-radius: 4px;
            }}

            QPushButton#pool_join_btn {{
                background-color: {BitSOTATheme.COLOR1};
                color: {BitSOTATheme.COLOR2_VARIANT};
                border: none;
                border-radius: 4px;
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 16px;
                font-weight: 600;
                line-height: 1.2;
            }}

            QPushButton#pool_join_btn:hover {{
                background-color: rgba(21, 0, 73, 0.9);
            }}

            QPushButton#pool_join_btn:pressed {{
                background-color: rgba(21, 0, 73, 0.8);
            }}
        """

    def __init__(self, pool_data: dict, parent=None):
        super().__init__(parent)
        self.pool_data = pool_data
        self.setObjectName("pool_card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(self.get_stylesheet())
        self._setup_ui()

    def _setup_ui(self):
        """Build the pool card layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Pool name title
        title = QLabel(self.pool_data.get("name", "Pool"))
        title.setObjectName("config_section_title")
        layout.addWidget(title)

        # Stats container
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(12)

        _add_stat_row(stats_layout, "SOTA", str(self.pool_data.get("sota", "-")))

        divider = _create_divider()
        stats_layout.addWidget(divider)

        _add_stat_row(
            stats_layout, "Number of Miners", str(self.pool_data.get("miners", "-"))
        )
        _add_stat_row(
            stats_layout, "Staked  Amount", str(self.pool_data.get("staked", "-"))
        )
        _add_stat_row(
            stats_layout, "TAO Rewards", str(self.pool_data.get("rewards", "-"))
        )

        layout.addLayout(stats_layout)

        # Join Pool button (full width)
        join_btn = QPushButton("Join Pool")
        join_btn.setObjectName("pool_join_btn")
        join_btn.setFixedHeight(48)
        join_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        join_btn.clicked.connect(lambda: self.join_clicked.emit(self.pool_data))
        layout.addWidget(join_btn)


# =============================================================================
# Pool Detail Screen (shown after clicking Join Pool)
# =============================================================================


class PoolDetailScreen(QWidget):
    """Pool detail screen shown after joining a pool.

    Emits switch_pool_requested when the user clicks "Switch Pool".

    Displays:
    - Mining Configuration with metrics (Pool Contribution, Pending Rewards, Reputation)
    - Current Pool selector with Switch Pool and Start Mining buttons
    - Three stats panels: Miner Stats, Mining Status, Pool Status
    - Mining Logs
    """

    switch_pool_requested = Signal()

    def __init__(self, pool_data: dict, main_window=None, parent=None):
        super().__init__(parent)
        self.pool_data = pool_data
        self.main_window = main_window
        self.mock = MOCK_POOL_DETAIL
        self.setup_ui()

    def setup_ui(self):
        """Build the pool detail layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        # Mining Configuration section
        config_section = self._create_config_section()
        layout.addWidget(config_section)

        # Three stats panels side by side
        stats_panels = self._create_stats_panels()
        layout.addWidget(stats_panels)

        # Mining Logs
        logs_section = self._create_logs_section()
        layout.addWidget(logs_section, 1)

    # ========== Mining Configuration ==========

    def _create_config_section(self) -> QWidget:
        """Create mining configuration section with metrics, pool selector, and buttons."""
        section = QWidget()
        section.setObjectName("mining_config_box")
        section.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title row
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Mining Configuration")
        title.setObjectName("config_section_title")
        title_row.addWidget(title)

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

        # Metrics row (3 equal columns)
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(24)

        metrics_row.addWidget(
            self._create_metric_box(
                "Pool Contribution",
                self.mock["pool_contribution"],
                show_progress=True,
            ),
            1,
        )
        metrics_row.addWidget(
            self._create_metric_box(
                "Pending Rewards", self.mock["pending_rewards"]
            ),
            1,
        )
        metrics_row.addWidget(
            self._create_metric_box("Reputation", self.mock["reputation"]),
            1,
        )

        layout.addLayout(metrics_row)

        # Current Pool row + buttons
        pool_row = QHBoxLayout()
        pool_row.setSpacing(24)

        # Current Pool input (read-only)
        pool_input_container = QWidget()
        pool_input_layout = QVBoxLayout(pool_input_container)
        pool_input_layout.setContentsMargins(0, 0, 0, 0)
        pool_input_layout.setSpacing(8)

        pool_label = QLabel("Current Pool")
        pool_label.setObjectName("form_label")
        pool_input_layout.addWidget(pool_label)

        pool_display = QWidget()
        pool_display.setFixedHeight(48)
        pool_display.setStyleSheet(
            f"background-color: {BitSOTATheme.CONTENT_BOX_BG};"
            f"border: 1px solid {BitSOTATheme.BORDER_12};"
            f"border-radius: 4px;"
        )
        pool_display_layout = QHBoxLayout(pool_display)
        pool_display_layout.setContentsMargins(14, 8, 14, 8)
        pool_display_layout.setSpacing(8)

        # Pool nodes icon (20x20)
        pool_icon = QLabel()
        pool_icon.setFixedSize(20, 20)
        pool_icon.setStyleSheet("border: none; background: transparent;")
        try:
            icon_renderer = QSvgRenderer(resource_path("gui/images/pool_nodes.svg"))
            icon_pixmap = QPixmap(20, 20)
            icon_pixmap.fill(Qt.GlobalColor.transparent)
            icon_painter = QPainter(icon_pixmap)
            icon_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            icon_renderer.render(icon_painter)
            icon_painter.end()
            pool_icon.setPixmap(icon_pixmap)
        except Exception:
            pass
        pool_display_layout.addWidget(pool_icon)

        pool_name_label = QLabel(self.pool_data.get("name", "Pool 1"))
        pool_name_label.setStyleSheet(
            f"color: {BitSOTATheme.BLACK100}; font-size: 14px; font-weight: 500; border: none;"
        )
        pool_display_layout.addWidget(pool_name_label)
        pool_display_layout.addStretch()

        pool_input_layout.addWidget(pool_display)
        pool_row.addWidget(pool_input_container, 1)

        # Buttons container (aligned to bottom of row)
        buttons_wrapper = QWidget()
        buttons_layout = QVBoxLayout(buttons_wrapper)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(0)
        buttons_layout.addStretch()

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(10)

        self.switch_pool_btn = SecondaryButton("Switch Pool ", width=200, height=48)
        self.switch_pool_btn.clicked.connect(self._on_switch_pool)
        buttons_row.addWidget(self.switch_pool_btn)

        self.start_mining_btn = PrimaryButton(
            "Start Mining",
            width=200,
            height=48,
            icon_path=resource_path("gui/images/play.svg"),
            icon_size=20,
        )
        self.start_mining_btn.clicked.connect(self._on_start_mining)
        buttons_row.addWidget(self.start_mining_btn)

        buttons_layout.addLayout(buttons_row)
        pool_row.addWidget(buttons_wrapper)
        layout.addLayout(pool_row)

        return section

    def _create_metric_box(
        self, label: str, value: str, show_progress: bool = False
    ) -> QWidget:
        """Create a metric box with label, info icon, and large value text."""
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(8)

        # Header row (label + info icon)
        header = QHBoxLayout()
        header.setSpacing(4)

        title = QLabel(label)
        title.setStyleSheet(
            f"color: {BitSOTATheme.BLACK100}; font-size: 14px; font-weight: 500;"
        )
        header.addWidget(title)

        info_icon = QLabel()
        info_icon.setFixedSize(16, 16)
        try:
            renderer = QSvgRenderer(resource_path("gui/images/info-circle.svg"))
            pixmap = QPixmap(16, 16)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            info_icon.setPixmap(pixmap)
        except Exception:
            pass
        header.addWidget(info_icon)
        header.addStretch()

        box_layout.addLayout(header)

        # Value
        value_label = QLabel(value)
        value_label.setStyleSheet(
            f"color: {BitSOTATheme.BLACK100}; font-size: 24px; font-weight: 600;"
        )
        box_layout.addWidget(value_label)

        # Progress bar (optional)
        if show_progress:
            progress = ProgressBar()
            progress.setValue(self.mock.get("pool_contribution_value", 0))
            box_layout.addWidget(progress)

        return box

    # ========== Stats Panels ==========

    def _create_stats_panels(self) -> QWidget:
        """Create the 3 side-by-side stats panels."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        layout.addWidget(self._create_miner_stats(), 1)
        layout.addWidget(self._create_mining_status(), 1)
        layout.addWidget(self._create_pool_status(), 1)

        return container

    def _create_miner_stats(self) -> QWidget:
        """Create Miner Stats panel."""
        stats = self.mock["miner_stats"]
        panel = _create_stats_panel("Miner Stats")
        content = panel.layout().itemAt(1).layout()  # stats_container

        _add_stat_row(content, "Total Score", stats["total_score"])
        content.addWidget(_create_divider())
        _add_stat_row(content, "Evaluation Tasks Completed", stats["eval_tasks"])
        _add_stat_row(content, "Score", stats["eval_score"])
        _add_stat_row(content, "Evolution Tasks Completed", stats["evo_tasks"])
        _add_stat_row(content, "Score", stats["evo_score"])

        return panel

    def _create_mining_status(self) -> QWidget:
        """Create Mining Status panel."""
        status = self.mock["mining_status"]
        panel = _create_stats_panel("Mining Status")
        content = panel.layout().itemAt(1).layout()

        _add_stat_row(content, "Wallet", status["wallet"])
        _add_colored_stat_row(content, "Status", status["status"], "#E97135")
        _add_colored_stat_row(content, "Connection", status["connection"], "#D02533")
        _add_stat_row(content, "Runtime", status["runtime"])
        _add_stat_row(content, "Resource Contributed", status["resource_contributed"])
        _add_stat_row(content, "Last Payout", status["last_payout"])
        _add_stat_row(content, "Next Payout", status["next_payout"])

        return panel

    def _create_pool_status(self) -> QWidget:
        """Create Pool Status panel."""
        pool = self.mock["pool_status"]
        panel = _create_stats_panel("Pool Status")
        content = panel.layout().itemAt(1).layout()

        _add_stat_row(content, "SOTA", pool["sota"])
        content.addWidget(_create_divider())
        _add_stat_row(content, "Total Resource Contributed", pool["total_resource"])
        _add_stat_row(content, "Active Miners", pool["active_miners"])
        _add_stat_row(content, "Total Rewards Distributed", pool["total_rewards"])

        return panel

    # ========== Mining Logs ==========

    def _create_logs_section(self) -> QWidget:
        """Create mining logs section with clear button."""
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

        clear_btn = QPushButton("Clear Logs")
        clear_btn.setObjectName("clear_logs_button")
        clear_btn.setFixedHeight(32)
        clear_btn.clicked.connect(self._clear_logs)
        header.addWidget(clear_btn)

        layout.addLayout(header)

        self.logs_text = QTextEdit()
        self.logs_text.setObjectName("logs_text")
        self.logs_text.setReadOnly(True)
        self.logs_text.append("[Pool] Interface loaded")
        layout.addWidget(self.logs_text)

        return section

    def _clear_logs(self):
        """Clear mining logs."""
        self.logs_text.clear()

    # ========== Button Handlers (stubs) ==========

    def _on_switch_pool(self):
        """Handle Switch Pool button click — go back to pool list."""
        self.switch_pool_requested.emit()

    def _on_start_mining(self):
        """Handle Start Mining button click. (Not implemented yet)"""
        pass

    def update_wallet_status(self, wallet_name: str):
        """Update wallet status display (compatibility method)."""
        pass


# =============================================================================
# Progress Bar widget (used in Pool Contribution metric)
# =============================================================================


class ProgressBar(QWidget):
    """Segmented progress bar that fills the full container width."""

    BOX_SPACING = 2
    BOX_HEIGHT = 12
    FILLED_COLOR = QColor(21, 0, 73)          # COLOR1 solid
    EMPTY_COLOR = QColor(21, 0, 73, 31)       # COLOR1 ~12% opacity

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.setFixedHeight(self.BOX_HEIGHT)

    def setValue(self, value: int):
        self.value = max(0, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        if w <= 0:
            return

        # Calculate box count and width dynamically to fill full width
        box_count = max(1, (w + self.BOX_SPACING) // (6 + self.BOX_SPACING))
        box_width = (w - (box_count - 1) * self.BOX_SPACING) / box_count
        filled_boxes = int((self.value / 100.0) * box_count)

        for i in range(box_count):
            x = int(i * (box_width + self.BOX_SPACING))
            bw = int((i + 1) * (box_width + self.BOX_SPACING) - self.BOX_SPACING) - x
            color = self.FILLED_COLOR if i < filled_boxes else self.EMPTY_COLOR
            painter.fillRect(x, 0, bw, self.BOX_HEIGHT, color)


# =============================================================================
# Pool Mining Screen (coordinator: pool list <-> pool detail)
# =============================================================================


class PoolMiningScreen(QWidget):
    """Pool mining screen that switches between pool list and pool detail.

    - View 0 (default): Pool cards grid
    - View 1: PoolDetailScreen (shown after clicking Join Pool)
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.pool_cards: list[PoolCard] = []
        self.setup_ui()

    def setup_ui(self):
        """Initialize the pool mining UI with a stacked widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.stack = QStackedWidget()

        # View 0: Pool list (card grid)
        self.pool_list_view = self._create_pool_list_view()
        self.stack.addWidget(self.pool_list_view)

        # View 1: Pool detail (created on demand, placeholder for now)
        self.pool_detail_view: Optional[PoolDetailScreen] = None

        layout.addWidget(self.stack)

    def _create_pool_list_view(self) -> QWidget:
        """Create the scrollable pool cards grid view."""
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        self.pool_grid = QGridLayout()
        self.pool_grid.setHorizontalSpacing(POOL_GRID_SPACING)
        self.pool_grid.setVerticalSpacing(POOL_GRID_SPACING)

        scroll_layout.addLayout(self.pool_grid)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        container_layout.addWidget(scroll)

        # Load placeholder pools
        self._load_pools(PLACEHOLDER_POOLS)

        return container

    def _load_pools(self, pools: list[dict]):
        """Load pool data and create pool cards in the grid."""
        for card in self.pool_cards:
            card.deleteLater()
        self.pool_cards.clear()

        for i, pool_data in enumerate(pools):
            card = PoolCard(pool_data)
            card.join_clicked.connect(self._on_join_pool)
            row = i // POOL_GRID_COLUMNS
            col = i % POOL_GRID_COLUMNS
            self.pool_grid.addWidget(card, row, col)
            self.pool_cards.append(card)

    def _on_join_pool(self, pool_data: dict):
        """Handle Join Pool button click: switch to pool detail view."""
        # Remove previous detail view if exists
        if self.pool_detail_view is not None:
            self.stack.removeWidget(self.pool_detail_view)
            self.pool_detail_view.deleteLater()

        # Create and show pool detail
        self.pool_detail_view = PoolDetailScreen(
            pool_data=pool_data, main_window=self.main_window
        )
        self.pool_detail_view.switch_pool_requested.connect(self.show_pool_list)
        self.stack.addWidget(self.pool_detail_view)
        self.stack.setCurrentWidget(self.pool_detail_view)

    def show_pool_list(self):
        """Switch back to pool list view."""
        self.stack.setCurrentWidget(self.pool_list_view)

    def update_wallet_status(self, wallet_name: str):
        """Update wallet status display (compatibility method)."""
        if self.pool_detail_view:
            self.pool_detail_view.update_wallet_status(wallet_name)


# =============================================================================
# Shared helper functions for stat rows / dividers
# =============================================================================


def _create_stats_panel(title_text: str) -> QWidget:
    """Create a stats panel container with title and a VBoxLayout for content."""
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


def _add_stat_row(parent_layout: QVBoxLayout, label_text: str, value_text: str):
    """Add a stat row (label + right-aligned value) to the parent layout."""
    row = QHBoxLayout()
    row.setSpacing(0)

    label = QLabel(label_text)
    label.setObjectName("stat_label")
    row.addWidget(label)
    row.addStretch()

    value = QLabel(value_text)
    value.setObjectName("stat_value")
    value.setAlignment(Qt.AlignmentFlag.AlignRight)
    row.addWidget(value)

    parent_layout.addLayout(row)


def _add_colored_stat_row(
    parent_layout: QVBoxLayout, label_text: str, value_text: str, color: str
):
    """Add a stat row with a colored value."""
    row = QHBoxLayout()
    row.setSpacing(0)

    label = QLabel(label_text)
    label.setObjectName("stat_label")
    row.addWidget(label)
    row.addStretch()

    value = QLabel(value_text)
    value.setStyleSheet(f"color: {color}; font-size: 14px;")
    value.setAlignment(Qt.AlignmentFlag.AlignRight)
    row.addWidget(value)

    parent_layout.addLayout(row)


def _create_divider() -> QWidget:
    """Create a divider line widget."""
    divider = QWidget()
    divider.setObjectName("stat_divider")
    divider.setFixedHeight(1)
    divider.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    return divider

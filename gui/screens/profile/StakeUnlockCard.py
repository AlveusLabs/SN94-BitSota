from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path
from .StatsCard import StatsCard
from .ChartWidget import ChartWidget


ICON_PATH = resource_path("gui/images/profile/wallet.svg")

STAKE_UNLOCK_CARD_STYLE = """
QWidget#stats_card {
    background-color: #F6F5F8;
    border-radius: 4px;
}
QLabel#card_title {
    font-size: 14px;
    font-weight: 600;
    color: #0C0029;
}
QWidget#value_unit_container {
    border-bottom: 1px dashed black;
}
QLabel#value_label {
    font-size: 36px;
    font-weight: 600;
    color: #0C0029;
    padding: 0;
    margin: 0;
}
QLabel#unit_label {
    font-size: 20px;
    font-weight: 600;
    color: #0C0029;
    padding: 0;
    margin: 0;
}
QLabel#desc_label {
    font-size: 14px;
    font-weight: 400;
    color: rgba(12, 0, 41, 0.8);
}
QLabel#total_label {
    font-size: 14px;
    font-weight: 400;
    color: rgba(12, 0, 41, 0.8);
}
QLabel#axis_label {
    font-size: 12px;
    color: rgba(12, 0, 41, 0.8);
}
QPushButton#stake_btn {
    background-color: #150049;
    color: #71DADE;
    border: none;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton#stake_btn:hover {
    background-color: #6A1B9A;
}
"""


class StakeUnlockCard(StatsCard):
    """
    Stake & Unlock Rate Card
    
    Displays user's stake information and unlock rate chart, containing:
    - Title and icon
    - Three stat values: staked amount, current unlock rate, pool share
    - "Stake" button
    - Total pool stake display
    - Unlock rate curve chart
    
    Signals:
        stake_clicked: Emitted when user clicks the stake button
    """
    
    stake_clicked = Signal()  # Stake button click signal
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize user interface"""
        self.setStyleSheet(STAKE_UNLOCK_CARD_STYLE)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)
        
        # ========== Top Section (Header + Stats, max height 150px) ==========
        top_section = QWidget()
        top_section.setMaximumHeight(115)

        top_layout = QVBoxLayout(top_section)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        
        # ========== Header Area ==========
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        
        # Icon (26x26 SVG)
        icon_widget = QSvgWidget(ICON_PATH)
        icon_widget.setFixedSize(26, 26)
        header_layout.addWidget(icon_widget)
        
        # Title
        title_label = QLabel("STAKE & UNLOCK RATE")
        title_label.setObjectName("card_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        top_layout.addWidget(header)
        top_layout.addSpacing(24)
        
        # ========== Stats Value Row ==========
        stats_row = QWidget()
        stats_layout = QHBoxLayout(stats_row)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(24)
        
        # Your staked amount
        staked_widget, self.staked_label = self._create_stat_widget("0.00", "$TAO", "Your Staked", dashed=True)
        stats_layout.addWidget(staked_widget)
        
        # Current unlock rate
        unlock_widget, self.unlock_label = self._create_stat_widget("0", "%", "Current Unlock Rate")
        stats_layout.addWidget(unlock_widget)
        
        # Pool share
        pool_widget, self.pool_label = self._create_stat_widget("0.0", "%", "Pool Share")
        stats_layout.addWidget(pool_widget)
        
        stats_layout.addStretch()
        
        # Stake button
        self.stake_btn = QPushButton("Stake")
        self.stake_btn.setObjectName("stake_btn")
        self.stake_btn.setFixedSize(QSize(87, 40))
        self.stake_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stake_btn.clicked.connect(self.stake_clicked.emit)
        stats_layout.addWidget(self.stake_btn)
        
        top_layout.addWidget(stats_row)
        
        layout.addWidget(top_section)
        layout.addSpacing(18)
        
        # ========== Total Pool Stake ==========
        self.total_label = QLabel("Total Pool Stake: 0 $TAO")
        self.total_label.setObjectName("total_label")
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.total_label)
        layout.addSpacing(18)
        
        # ========== Chart Area ==========
        chart_widget = QWidget()
        chart_widget.setMinimumHeight(150)
        chart_widget.setStyleSheet("background-color: transparent;")
        
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        
        # Chart content (Y-axis + curve)
        chart_content = QWidget()
        chart_content_layout = QHBoxLayout(chart_content)
        chart_content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Y-axis labels
        y_axis = QWidget()
        y_axis_layout = QVBoxLayout(y_axis)
        y_axis_layout.setContentsMargins(0, 0, 12, 0)
        y_axis_layout.setSpacing(0)
        
        # Add Y-axis tick labels (100%, 75%, 50%, 25%, 0%)
        for pct in ["100%", "75%", "50%", "25%", "0%"]:
            label = QLabel(pct)
            label.setObjectName("axis_label")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            y_axis_layout.addWidget(label)
            if pct != "0%":
                y_axis_layout.addStretch()
        
        chart_content_layout.addWidget(y_axis)
        
        # Chart drawing area
        self.chart_area = ChartWidget()
        chart_content_layout.addWidget(self.chart_area, 1)
        
        chart_layout.addWidget(chart_content)
        
        # X-axis label
        x_label = QLabel("Stake in $TAO →")
        x_label.setObjectName("axis_label")
        x_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        chart_layout.addWidget(x_label)
        
        layout.addWidget(chart_widget, 1)
        
    def _create_stat_widget(self, value: str, unit: str, label: str, dashed: bool = False) -> tuple:
        """
        Create stat value component
        
        Args:
            value: Value
            unit: Unit
            label: Description label
            dashed: Whether to show dashed underline
            
        Returns:
            Tuple of (QWidget, QLabel) - widget and value label reference
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Value row
        value_row = QWidget()
        value_row_layout = QHBoxLayout(value_row)
        value_row_layout.setContentsMargins(0, 0, 0, 0)
        value_row_layout.setSpacing(0)
        value_row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Container for value + unit (shared dashed underline if needed)
        value_unit_container = QWidget()
        if dashed:
            value_unit_container.setObjectName("value_unit_container")
            value_unit_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        value_unit_layout = QHBoxLayout(value_unit_container)
        value_unit_layout.setContentsMargins(0, 0, 0, 0)
        value_unit_layout.setSpacing(6)
        value_unit_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Value label
        value_label = QLabel(value)
        value_label.setObjectName("value_label")
        value_unit_layout.addWidget(value_label)
        
        # Unit label
        unit_label = QLabel(unit)
        unit_label.setObjectName("unit_label")
        value_unit_layout.addWidget(unit_label, 0, Qt.AlignmentFlag.AlignBottom)
        
        value_row_layout.addWidget(value_unit_container)
        
        layout.addWidget(value_row)
        
        # Description label
        desc_label = QLabel(label)
        desc_label.setObjectName("desc_label")
        layout.addWidget(desc_label)
        
        return widget, value_label
    
    def set_values(self, staked: str, unlock_rate: str, pool_share: str, total_pool: str):
        """
        Update all displayed values
        
        Args:
            staked: Your staked amount
            unlock_rate: Current unlock rate percentage
            pool_share: Pool share percentage
            total_pool: Total pool stake amount
        """
        self.staked_label.setText(staked)
        self.unlock_label.setText(unlock_rate)
        self.pool_label.setText(pool_share)
        self.total_label.setText(f"Total Pool Stake: {total_pool} $TAO")
        
        # Update chart max stake if total_pool is a valid number
        try:
            max_stake = float(total_pool.replace(",", ""))
            self.chart_area.set_max_stake(max_stake)
        except ValueError:
            pass

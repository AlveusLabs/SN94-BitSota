from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)
from PySide6.QtSvgWidgets import QSvgWidget

from gui.resource_path import resource_path
from .StatsCard import StatsCard
from .SegmentedProgressBar import SegmentedProgressBar


SMALL_STATS_CARD_STYLE = """
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
}
QLabel#unit_label {
    font-size: 20px;
    font-weight: 600;
    color: #0C0029;
}
QLabel#legend_label {
    font-size: 12px;
    font-weight: 400;
    color: rgba(12, 0, 41, 0.6);
}
QWidget#legend_dot_purple {
    background-color: #3E277C;
    border-radius: 4px;
}
QWidget#legend_dot_cyan {
    background-color: #71DADE;
    border-radius: 4px;
}
"""


class SmallStatsCard(StatsCard):
    """
    Small Stats Card
    
    Smaller card for displaying "Total TAO Rewards" and "Cumulative Runtime", containing:
    - Title and icon
    - Main value display
    - Direct/Pool legend
    - Segmented progress bar
    
    Args:
        title: Card title
        icon_path: Path to SVG icon (relative path for resource_path)
        show_dashed: Whether to show dashed underline on value
    """
    
    def __init__(self, title: str, icon_path: str, show_dashed: bool = True, parent=None):
        super().__init__(parent)
        self.title = title
        self.icon_path = resource_path(icon_path)
        self.show_dashed = show_dashed
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize user interface"""
        self.setStyleSheet(SMALL_STATS_CARD_STYLE)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)
        
        # ========== Header Area ==========
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        
        # Icon (26x26 SVG)
        icon_widget = QSvgWidget(self.icon_path)
        icon_widget.setFixedSize(26, 26)
        header_layout.addWidget(icon_widget)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setObjectName("card_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # ========== Value Area ==========
        value_section = QWidget()
        value_layout = QVBoxLayout(value_section)
        value_layout.setContentsMargins(0, 0, 0, 0)
        value_layout.setSpacing(14)
        
        # Value row
        value_row = QWidget()
        value_row_layout = QHBoxLayout(value_row)
        value_row_layout.setContentsMargins(0, 0, 0, 0)
        value_row_layout.setSpacing(0)
        value_row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Container for value + unit (shared dashed underline)
        self.value_unit_container = QWidget()
        if self.show_dashed:
            self.value_unit_container.setObjectName("value_unit_container")
            self.value_unit_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        value_unit_layout = QHBoxLayout(self.value_unit_container)
        value_unit_layout.setContentsMargins(0, 0, 0, 0)
        value_unit_layout.setSpacing(6)
        value_unit_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Main value label
        self.value_label = QLabel("0.00")
        self.value_label.setObjectName("value_label")
        value_unit_layout.addWidget(self.value_label)
        
        # Unit label
        self.unit_label = QLabel("$TAO")
        self.unit_label.setObjectName("unit_label")
        value_unit_layout.addWidget(self.unit_label, 0, Qt.AlignmentFlag.AlignBottom)
        
        value_row_layout.addWidget(self.value_unit_container)
        value_row_layout.addStretch()
        
        value_layout.addWidget(value_row)
        
        # ========== Legend Area ==========
        legend = QWidget()
        legend_layout = QHBoxLayout(legend)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(16)
        
        # Direct mining legend
        direct_legend = self._create_legend_item("legend_dot_purple", "Direct")
        legend_layout.addWidget(direct_legend)
        
        # Pool mining legend
        pool_legend = self._create_legend_item("legend_dot_cyan", "Pool")
        legend_layout.addWidget(pool_legend)
        legend_layout.addStretch()
        
        value_layout.addWidget(legend)
        
        # ========== Progress Bar ==========
        self.progress_bar = SegmentedProgressBar()
        value_layout.addWidget(self.progress_bar)
        
        layout.addWidget(value_section)
        layout.addStretch()
        
    def _create_legend_item(self, dot_object_name: str, text: str) -> QWidget:
        """
        Create legend item
        
        Args:
            dot_object_name: Object name for dot widget (determines color via stylesheet)
            text: Legend text
            
        Returns:
            QWidget containing dot and text
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        dot = QWidget()
        dot.setObjectName(dot_object_name)
        dot.setFixedSize(8, 8)
        layout.addWidget(dot)
        
        label = QLabel(text)
        label.setObjectName("legend_label")
        layout.addWidget(label)
        
        return widget
    
    def set_value(self, value: str, unit: str = "$TAO"):
        """
        Set the displayed value
        
        Args:
            value: Value text
            unit: Unit text (hide unit label when empty string)
        """
        self.value_label.setText(value)
        self.unit_label.setText(unit)
        self.unit_label.setVisible(bool(unit))  # Hide label when unit is empty

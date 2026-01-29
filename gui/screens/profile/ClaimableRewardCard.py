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
from .SegmentedProgressBar import SegmentedProgressBar


ICON_PATH = resource_path("gui/images/profile/unlock.svg")

CLAIMABLE_REWARD_CARD_STYLE = """
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
QLabel#nominal_label {
    font-size: 14px;
    font-weight: 400;
    color: rgba(12, 0, 41, 0.8);
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
QPushButton#claim_btn {
    background-color: #150049;
    color: #71DADE;
    border: none;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton#claim_btn:hover {
    background-color: #6A1B9A;
}
"""


class ClaimableRewardCard(StatsCard):
    """
    Claimable Reward Card
    
    Displays user's claimable TAO rewards, containing:
    - Title and icon
    - Legend (Liquid Reward / Auto-staked Reward)
    - Main value display (with dashed underline)
    - Nominal rewards amount
    - "Claim" button
    - Segmented progress bar
    
    Signals:
        claim_clicked: Emitted when user clicks the claim button
    """
    
    claim_clicked = Signal()  # Claim button click signal
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize user interface"""
        self.setStyleSheet(CLAIMABLE_REWARD_CARD_STYLE)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)
        
        # ========== Header Area ==========
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        
        # Title container (icon + title text)
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(6)
        
        # Icon (26x26 SVG)
        icon_widget = QSvgWidget(ICON_PATH)
        icon_widget.setFixedSize(26, 26)
        title_layout.addWidget(icon_widget)
        
        # Title text
        title_label = QLabel("CLAIMABLE REWARD")
        title_label.setObjectName("card_title")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        header_layout.addWidget(title_container)
        header_layout.addStretch()
        
        # ========== Legend Area ==========
        legend = QWidget()
        legend_layout = QHBoxLayout(legend)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(16)
        
        # Liquid reward legend (dark purple dot + text)
        liquid_legend = self._create_legend_item("legend_dot_purple", "Liquid Reward")
        legend_layout.addWidget(liquid_legend)
        
        # Auto-staked reward legend (cyan dot + text)
        auto_legend = self._create_legend_item("legend_dot_cyan", "Auto-staked Reward")
        legend_layout.addWidget(auto_legend)
        
        header_layout.addWidget(legend)
        layout.addWidget(header)
        layout.addSpacing(24)
        
        # ========== Value Display Area ==========
        value_row = QWidget()
        value_layout = QHBoxLayout(value_row)
        value_layout.setContentsMargins(0, 0, 0, 0)
        value_layout.setSpacing(0)
        
        # Value container
        value_container = QWidget()
        value_container_layout = QVBoxLayout(value_container)
        value_container_layout.setContentsMargins(0, 0, 0, 0)
        value_container_layout.setSpacing(8)
        
        # Main value row (value + unit with shared dashed underline)
        value_row_inner = QWidget()
        value_inner_layout = QHBoxLayout(value_row_inner)
        value_inner_layout.setContentsMargins(0, 0, 0, 0)
        value_inner_layout.setSpacing(0)
        value_inner_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Container for value + unit (shared dashed underline)
        value_unit_container = QWidget()
        value_unit_container.setObjectName("value_unit_container")
        value_unit_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        value_unit_layout = QHBoxLayout(value_unit_container)
        value_unit_layout.setContentsMargins(0, 0, 0, 0)
        value_unit_layout.setSpacing(6)
        value_unit_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Main value label
        self.value_label = QLabel("800.12")
        self.value_label.setObjectName("value_label")
        value_unit_layout.addWidget(self.value_label)
        
        # Unit label
        unit_label = QLabel("$TAO")
        unit_label.setObjectName("unit_label")
        value_unit_layout.addWidget(unit_label, 0, Qt.AlignmentFlag.AlignBottom)
        
        value_inner_layout.addWidget(value_unit_container)
        value_inner_layout.addStretch()
        
        value_container_layout.addWidget(value_row_inner)
        
        # Nominal rewards label
        self.nominal_label = QLabel("Nominal Rewards: 40,000")
        self.nominal_label.setObjectName("nominal_label")
        value_container_layout.addWidget(self.nominal_label)
        
        value_layout.addWidget(value_container)
        value_layout.addStretch()
        
        # ========== Claim Button ==========
        self.claim_btn = QPushButton("Claim")
        self.claim_btn.setObjectName("claim_btn")
        self.claim_btn.setFixedSize(QSize(87, 40))
        self.claim_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.claim_btn.clicked.connect(self.claim_clicked.emit)
        value_layout.addWidget(self.claim_btn)
        
        layout.addWidget(value_row)
        layout.addSpacing(16)
        
        # ========== Progress Bar ==========
        self.progress_bar = SegmentedProgressBar()
        self.progress_bar.set_ratios(1, 0)
        layout.addWidget(self.progress_bar)
        
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
        
        # Color dot
        dot = QWidget()
        dot.setObjectName(dot_object_name)
        dot.setFixedSize(8, 8)
        layout.addWidget(dot)
        
        # Text label
        label = QLabel(text)
        label.setObjectName("legend_label")
        layout.addWidget(label)
        
        return widget
    
    def set_value(self, value: str, nominal: str = "40,000"):
        """
        Set the displayed values
        
        Args:
            value: Claimable reward value
            nominal: Nominal rewards amount
        """
        self.value_label.setText(value)
        self.nominal_label.setText(f"Nominal Rewards: {nominal}")

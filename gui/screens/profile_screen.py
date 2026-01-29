from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QFrame,
    QScrollArea,
    QSizePolicy,
)
from PySide6.QtGui import QPainter, QColor, QPen

from gui.components import PrimaryButton, TabSwitcher



class SegmentedProgressBar(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(16)
        self.liquid_ratio = 0.55
        self.auto_staked_ratio = 0.45
        
    def set_ratios(self, liquid: float, auto_staked: float):
        self.liquid_ratio = liquid
        self.auto_staked_ratio = auto_staked
        self.update()  # Trigger repaint
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  # Enable anti-aliasing
        
        width = self.width()
        height = self.height()
        segment_width = 6  # Width of each segment
        gap = 2  # Gap between segments
        total_segments = int(width / (segment_width + gap))  # Calculate total segments
        
        # Calculate segments for liquid reward
        liquid_segments = int(total_segments * self.liquid_ratio)
        
        # Define colors
        dark_color = QColor("#150049")  # Dark purple - Liquid reward
        cyan_color = QColor("#71DADE")  # Cyan - Auto-staked reward
        
        # Draw each segment
        x = 0
        for i in range(total_segments):
            if i < liquid_segments:
                # First segments use dark purple (liquid reward)
                painter.fillRect(int(x), 0, segment_width, height, dark_color)
            else:
                # Remaining segments use cyan (auto-staked reward)
                painter.fillRect(int(x), 0, segment_width, height, cyan_color)
            x += segment_width + gap


class StatsCard(QWidget):
    """
    Stats Card Base Class
    
    Base class for all stats cards providing unified background style:
    - Background color: #F6F5F8 (light gray)
    - Border radius: 4px
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stats_card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget#stats_card {
                background-color: #F6F5F8;
                border-radius: 4px;
            }
        """)


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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)  # Padding
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
        
        # Icon placeholder (32x32 circle)
        icon_widget = QWidget()
        icon_widget.setFixedSize(32, 32)
        icon_widget.setStyleSheet("background-color: #3E277C; border-radius: 16px;")
        title_layout.addWidget(icon_widget)
        
        # Title text
        title_label = QLabel("CLAIMABLE REWARD")
        title_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #0C0029;
            text-transform: uppercase;
        """)
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
        liquid_legend = self._create_legend_item("#3E277C", "Liquid Reward")
        legend_layout.addWidget(liquid_legend)
        
        # Auto-staked reward legend (cyan dot + text)
        auto_legend = self._create_legend_item("#71DADE", "Auto-staked Reward")
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
        
        # Main value row (value + unit)
        value_row_inner = QWidget()
        value_inner_layout = QHBoxLayout(value_row_inner)
        value_inner_layout.setContentsMargins(0, 0, 0, 0)
        value_inner_layout.setSpacing(6)
        value_inner_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Main value label (with dashed underline)
        self.value_label = QLabel("800.12")
        self.value_label.setStyleSheet("""
            font-size: 36px;
            font-weight: 600;
            color: #0C0029;
            border-bottom: 1px dashed black;
        """)
        value_inner_layout.addWidget(self.value_label)
        
        # Unit label
        unit_label = QLabel("$TAO")
        unit_label.setStyleSheet("""
            font-size: 20px;
            font-weight: 600;
            color: #0C0029;
        """)
        value_inner_layout.addWidget(unit_label, 0, Qt.AlignmentFlag.AlignBottom)
        value_inner_layout.addStretch()
        
        value_container_layout.addWidget(value_row_inner)
        
        # Nominal rewards label
        self.nominal_label = QLabel("Nominal Rewards: 40,000")
        self.nominal_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 400;
            color: rgba(12, 0, 41, 0.8);
        """)
        value_container_layout.addWidget(self.nominal_label)
        
        value_layout.addWidget(value_container)
        value_layout.addStretch()
        
        # ========== Claim Button ==========
        self.claim_btn = PrimaryButton("Claim", width=100, height=40)
        self.claim_btn.clicked.connect(self.claim_clicked.emit)
        value_layout.addWidget(self.claim_btn)
        
        layout.addWidget(value_row)
        layout.addSpacing(16)
        
        # ========== Progress Bar ==========
        self.progress_bar = SegmentedProgressBar()
        layout.addWidget(self.progress_bar)
        
    def _create_legend_item(self, color: str, text: str) -> QWidget:
        """
        Create legend item
        
        Args:
            color: Dot color (hex color value)
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
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
        layout.addWidget(dot)
        
        # Text label
        label = QLabel(text)
        label.setStyleSheet("""
            font-size: 12px;
            font-weight: 400;
            color: rgba(12, 0, 41, 0.6);
        """)
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
        icon_color: Icon background color
        show_dashed: Whether to show dashed underline on value
    """
    
    def __init__(self, title: str, icon_color: str = "#3E277C", show_dashed: bool = True, parent=None):
        super().__init__(parent)
        self.title = title
        self.icon_color = icon_color
        self.show_dashed = show_dashed
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)
        
        # ========== Header Area ==========
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        
        # Icon
        icon_widget = QWidget()
        icon_widget.setFixedSize(32, 32)
        icon_widget.setStyleSheet(f"background-color: {self.icon_color}; border-radius: 16px;")
        header_layout.addWidget(icon_widget)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #0C0029;
            text-transform: uppercase;
        """)
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
        value_row_layout.setSpacing(6)
        value_row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Main value label (dashed underline based on show_dashed)
        self.value_label = QLabel("500.25")
        dashed_style = "border-bottom: 1px dashed black;" if self.show_dashed else ""
        self.value_label.setStyleSheet(f"""
            font-size: 36px;
            font-weight: 600;
            color: #0C0029;
            {dashed_style}
        """)
        value_row_layout.addWidget(self.value_label)
        
        # Unit label
        self.unit_label = QLabel("$TAO")
        self.unit_label.setStyleSheet("""
            font-size: 20px;
            font-weight: 600;
            color: #0C0029;
        """)
        value_row_layout.addWidget(self.unit_label, 0, Qt.AlignmentFlag.AlignBottom)
        value_row_layout.addStretch()
        
        value_layout.addWidget(value_row)
        
        # ========== Legend Area ==========
        legend = QWidget()
        legend_layout = QHBoxLayout(legend)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(16)
        
        # Direct mining legend
        direct_legend = self._create_legend_item("#3E277C", "Direct")
        legend_layout.addWidget(direct_legend)
        
        # Pool mining legend
        pool_legend = self._create_legend_item("#71DADE", "Pool")
        legend_layout.addWidget(pool_legend)
        legend_layout.addStretch()
        
        value_layout.addWidget(legend)
        
        # ========== Progress Bar ==========
        self.progress_bar = SegmentedProgressBar()
        value_layout.addWidget(self.progress_bar)
        
        layout.addWidget(value_section)
        layout.addStretch()
        
    def _create_legend_item(self, color: str, text: str) -> QWidget:
        """
        Create legend item
        
        Args:
            color: Dot color
            text: Legend text
            
        Returns:
            QWidget containing dot and text
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        dot = QWidget()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
        layout.addWidget(dot)
        
        label = QLabel(text)
        label.setStyleSheet("""
            font-size: 12px;
            font-weight: 400;
            color: rgba(12, 0, 41, 0.6);
        """)
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)
        
        # ========== Header Area ==========
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        
        # Icon
        icon_widget = QWidget()
        icon_widget.setFixedSize(32, 32)
        icon_widget.setStyleSheet("background-color: #3E277C; border-radius: 16px;")
        header_layout.addWidget(icon_widget)
        
        # Title
        title_label = QLabel("STAKE & UNLOCK RATE")
        title_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #0C0029;
            text-transform: uppercase;
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # ========== Stats Value Row ==========
        stats_row = QWidget()
        stats_layout = QHBoxLayout(stats_row)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(24)
        
        # Your staked amount
        staked_widget = self._create_stat_widget("800.24", "$TAO", "Your Staked", dashed=True)
        stats_layout.addWidget(staked_widget)
        
        # Current unlock rate
        unlock_widget = self._create_stat_widget("49", "%", "Current Unlock Rate")
        stats_layout.addWidget(unlock_widget)
        
        # Pool share
        pool_widget = self._create_stat_widget("4.9", "%", "Pool Share")
        stats_layout.addWidget(pool_widget)
        
        stats_layout.addStretch()
        
        # Stake button
        self.stake_btn = PrimaryButton("Stake", width=100, height=40)
        self.stake_btn.clicked.connect(self.stake_clicked.emit)
        stats_layout.addWidget(self.stake_btn)
        
        layout.addWidget(stats_row)
        
        # ========== Total Pool Stake ==========
        total_label = QLabel("Total Pool Stake: 1,860,000 $TAO")
        total_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        total_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 400;
            color: rgba(12, 0, 41, 0.8);
        """)
        layout.addWidget(total_label)
        
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
            label.setStyleSheet("font-size: 12px; color: rgba(12, 0, 41, 0.8);")
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            y_axis_layout.addWidget(label)
            if pct != "0%":
                y_axis_layout.addStretch()
        
        chart_content_layout.addWidget(y_axis)
        
        # Chart drawing area
        chart_area = ChartWidget()
        chart_content_layout.addWidget(chart_area, 1)
        
        chart_layout.addWidget(chart_content)
        
        # X-axis label
        x_label = QLabel("Stake in $TAO →")
        x_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        x_label.setStyleSheet("font-size: 12px; color: rgba(12, 0, 41, 0.8);")
        chart_layout.addWidget(x_label)
        
        layout.addWidget(chart_widget, 1)
        
    def _create_stat_widget(self, value: str, unit: str, label: str, dashed: bool = False) -> QWidget:
        """
        Create stat value component
        
        Args:
            value: Value
            unit: Unit
            label: Description label
            dashed: Whether to show dashed underline
            
        Returns:
            QWidget containing value and label
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Value row
        value_row = QWidget()
        value_layout = QHBoxLayout(value_row)
        value_layout.setContentsMargins(0, 0, 0, 0)
        value_layout.setSpacing(6)
        value_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Value label
        value_label = QLabel(value)
        border_style = "border-bottom: 1px dashed black;" if dashed else ""
        value_label.setStyleSheet(f"""
            font-size: 36px;
            font-weight: 600;
            color: #0C0029;
            {border_style}
        """)
        value_layout.addWidget(value_label)
        
        # Unit label
        unit_label = QLabel(unit)
        unit_label.setStyleSheet("""
            font-size: 20px;
            font-weight: 600;
            color: #0C0029;
        """)
        value_layout.addWidget(unit_label, 0, Qt.AlignmentFlag.AlignBottom)
        
        layout.addWidget(value_row)
        
        # Description label
        desc_label = QLabel(label)
        desc_label.setStyleSheet("""
            font-size: 14px;
            font-weight: 400;
            color: rgba(12, 0, 41, 0.8);
        """)
        layout.addWidget(desc_label)
        
        return widget


class ChartTooltip(QWidget):
    """
    Chart Tooltip
    
    Floating popup displayed when hovering over the chart curve, containing:
    - Stake amount
    - Unlock rate percentage
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize tooltip interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Container (with background and shadow)
        self.container = QWidget()
        self.container.setStyleSheet("""
            QWidget {
                background-color: #150049;
                border-radius: 6px;
            }
        """)
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(12, 8, 12, 8)
        container_layout.setSpacing(4)
        
        # Stake amount label
        self.stake_label = QLabel("Stake: 0 $TAO")
        self.stake_label.setStyleSheet("""
            font-size: 12px;
            font-weight: 500;
            color: #FFFFFF;
            background: transparent;
        """)
        container_layout.addWidget(self.stake_label)
        
        # Unlock rate label
        self.rate_label = QLabel("Unlock Rate: 0%")
        self.rate_label.setStyleSheet("""
            font-size: 12px;
            font-weight: 600;
            color: #71DADE;
            background: transparent;
        """)
        container_layout.addWidget(self.rate_label)
        
        layout.addWidget(self.container)
        
    def update_values(self, stake: float, rate: float):
        """
        Update tooltip displayed values
        
        Args:
            stake: Stake amount (in $TAO)
            rate: Unlock rate percentage (0-100)
        """
        self.stake_label.setText(f"Stake: {stake:,.0f} $TAO")
        self.rate_label.setText(f"Unlock Rate: {rate:.1f}%")


class ChartWidget(QWidget):
    """
    Chart Drawing Component
    
    Draws unlock rate curve chart showing relationship between stake amount and unlock rate.
    Uses logarithmic curve to simulate unlock rate growth trend.
    
    Supports mouse hover interaction:
    - Shows vertical indicator line
    - Shows dot on curve
    - Displays tooltip with current values
    """
    
    # Chart data range
    MAX_STAKE = 1860000  # Maximum stake amount (total pool stake)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)  # Enable mouse tracking
        
        # Mouse hover state
        self.hover_x = -1  # Current mouse X coordinate, -1 means not hovering
        self.hover_y = -1  # Current mouse Y coordinate
        
        # Create tooltip
        self.tooltip = ChartTooltip()
        self.tooltip.hide()
        
    def _calculate_unlock_rate(self, progress: float) -> float:
        """
        Calculate unlock rate based on progress
        
        Args:
            progress: Progress value (0.0 - 1.0)
            
        Returns:
            Unlock rate percentage (0 - 100)
        """
        import math
        if progress <= 0:
            return 0
        return 100 * math.log10(1 + progress * 9) / math.log10(10)
    
    def _calculate_y_from_progress(self, progress: float, height: int) -> float:
        """
        Calculate Y coordinate based on progress
        
        Args:
            progress: Progress value (0.0 - 1.0)
            height: Chart height
            
        Returns:
            Y coordinate value
        """
        rate = self._calculate_unlock_rate(progress)
        return height - (height * rate / 100)
        
    def mouseMoveEvent(self, event):
        """
        Mouse move event - Update hover position and show tooltip
        """
        self.hover_x = event.position().x()
        self.hover_y = event.position().y()
        
        width = self.width()
        height = self.height()
        
        if 0 <= self.hover_x <= width:
            # Calculate values at current position
            progress = self.hover_x / width
            stake = progress * self.MAX_STAKE
            rate = self._calculate_unlock_rate(progress)
            
            # Update tooltip content
            self.tooltip.update_values(stake, rate)
            
            # Calculate tooltip position (show above mouse)
            global_pos = self.mapToGlobal(event.position().toPoint())
            tooltip_x = global_pos.x() - self.tooltip.width() // 2
            tooltip_y = global_pos.y() - self.tooltip.height() - 15
            
            self.tooltip.move(int(tooltip_x), int(tooltip_y))
            self.tooltip.show()
        else:
            self.tooltip.hide()
            
        self.update()  # Trigger repaint
        
    def leaveEvent(self, event):
        """
        Mouse leave event - Hide tooltip and indicator
        """
        self.hover_x = -1
        self.hover_y = -1
        self.tooltip.hide()
        self.update()
        
    def paintEvent(self, event):
        """
        Custom paint event - Draw unlock rate curve
        
        Contains:
        1. Horizontal dashed grid lines
        2. Cyan logarithmic curve
        3. Hover indicator (vertical line + dot)
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # ========== Draw Grid Lines ==========
        pen = QPen(QColor(200, 200, 200))
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)  # Dashed style
        painter.setPen(pen)
        
        # Draw 5 horizontal grid lines
        for i in range(5):
            y = int(i * height / 4)
            painter.drawLine(0, y, width, y)
        
        # ========== Draw Curve ==========
        pen = QPen(QColor("#71DADE"))  # Cyan
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Use QPainterPath to draw smooth logarithmic curve
        from PySide6.QtGui import QPainterPath, QBrush
        import math
        
        path = QPainterPath()
        path.moveTo(0, height)
        
        # Iterate each pixel point, calculate logarithmic curve Y coordinate
        for i in range(width):
            x = i
            progress = i / width
            y = self._calculate_y_from_progress(progress, height)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        
        painter.drawPath(path)
        
        # ========== Draw Hover Indicator ==========
        if 0 <= self.hover_x <= width:
            progress = self.hover_x / width
            point_y = self._calculate_y_from_progress(progress, height)
            
            # Draw vertical indicator line (dashed)
            pen = QPen(QColor("#150049"))
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(self.hover_x), 0, int(self.hover_x), height)
            
            # Draw dot on curve
            painter.setPen(QPen(QColor("#150049"), 2))
            painter.setBrush(QBrush(QColor("#71DADE")))
            painter.drawEllipse(int(self.hover_x) - 6, int(point_y) - 6, 12, 12)


class MiningHistoryTable(QWidget):
    """
    Mining History Table Component
    
    Displays user's mining history records, supporting two modes:
    1. Direct Mining - Shows mining date, rewards, distributed rewards, track, runtime
    2. Pool Mining - Shows start date, rewards, contribution ratio, total rewards, status
    
    Uses TabSwitcher component to switch between two modes
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tab = "direct"  # Currently selected tab
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        
        # ========== Header Area ==========
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        
        # Title
        title = QLabel("Mining History")
        title.setStyleSheet("""
            font-size: 20px;
            font-weight: 600;
            color: #0C0029;
        """)
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Tab switcher
        self.tab_switcher = TabSwitcher()
        self.tab_switcher.add_tab("direct", "Direct Mining")
        self.tab_switcher.add_tab("pool", "Pool Mining")
        self.tab_switcher.tab_changed.connect(self._on_tab_changed)
        header_layout.addWidget(self.tab_switcher)
        
        layout.addWidget(header)
        
        # ========== Table Container ==========
        self.table_widget = QWidget()
        self.table_layout = QVBoxLayout(self.table_widget)
        self.table_layout.setContentsMargins(0, 0, 0, 0)
        self.table_layout.setSpacing(0)
        
        # Default to direct mining table
        self._create_direct_mining_table()
        
        layout.addWidget(self.table_widget)
        
    def _on_tab_changed(self, tab_id: str):
        """
        Tab change handler
        
        Args:
            tab_id: Newly selected tab ID ("direct" or "pool")
        """
        self.current_tab = tab_id
        
        # Clear existing table content
        while self.table_layout.count():
            item = self.table_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Create corresponding table based on selected tab
        if tab_id == "direct":
            self._create_direct_mining_table()
        else:
            self._create_pool_mining_table()
    
    def _create_table_header(self, columns: list, alignments: list = None) -> QWidget:
        """
        Create table header
        
        Args:
            columns: List of column titles
            alignments: List of alignment for each column
            
        Returns:
            Header QWidget
        """
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        for i, col in enumerate(columns):
            label = QLabel(col)
            label.setStyleSheet("""
                font-size: 14px;
                font-weight: 400;
                color: rgba(12, 0, 41, 0.6);
                padding: 16px 0;
            """)
            
            # Set alignment
            if alignments and i < len(alignments):
                label.setAlignment(alignments[i])
            else:
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            
            # Last column doesn't stretch, others distribute evenly
            if i == len(columns) - 1:
                layout.addWidget(label, 0)
            else:
                layout.addWidget(label, 1)
        
        return header
    
    def _create_table_row(self, data: tuple, is_track: bool = False) -> QWidget:
        """
        Create table data row
        
        Args:
            data: Row data tuple
            is_track: Whether it contains Track column (needs special color)
            
        Returns:
            Data row QWidget
        """
        row = QWidget()
        row.setStyleSheet("border-bottom: 1px solid #F0F0F0;")  # Row bottom separator
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        for i, value in enumerate(data):
            label = QLabel(str(value))
            
            # Track column uses purple text
            if is_track and i == 3:
                label.setStyleSheet("""
                    font-size: 14px;
                    font-weight: 400;
                    color: #4A148C;
                    padding: 16px 0;
                """)
            # Last column right aligned
            elif i == len(data) - 1:
                label.setStyleSheet("""
                    font-size: 14px;
                    font-weight: 400;
                    color: #0C0029;
                    padding: 16px 0;
                """)
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            # Other columns default style
            else:
                label.setStyleSheet("""
                    font-size: 14px;
                    font-weight: 400;
                    color: #0C0029;
                    padding: 16px 0;
                """)
            
            # Last column doesn't stretch
            if i == len(data) - 1:
                layout.addWidget(label, 0)
            else:
                layout.addWidget(label, 1)
        
        return row
    
    def _create_direct_mining_table(self):
        """
        Create direct mining history table
        
        Table columns: Mining Date, Rewards, Total Rewards Distributed, Track, Runtime
        """
        # Header definition
        columns = ["Mining Date", "Rewards", "Total Rewards Distributed", "Track", "Runtime"]
        alignments = [
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        ]
        
        # Create header
        header = self._create_table_header(columns, alignments)
        header.setStyleSheet("border-bottom: 1px solid #E8E8E8;")
        self.table_layout.addWidget(header)
        
        # Sample data
        rows_data = [
            ("Feb 6, 2025", "5 $TAO", "750.00 $TAO", "Track 1", "1h 32m 2s"),
            ("Feb 5, 2025", "2 $TAO", "800.00 $TAO", "Track 2", "1h 44m 0s"),
            ("Feb 4, 2025", "2 $TAO", "700.00 $TAO", "Track 1", "1h 27m 2s"),
            ("Feb 3, 2025", "6 $TAO", "300.00 $TAO", "Track 1", "1h 24m 0s"),
            ("Feb 2, 2025", "1 $TAO", "300.00 $TAO", "Track 3", "1h 23m 4s"),
            ("Feb 1, 2025", "2 $TAO", "200.00 $TAO", "Track 1", "1h 29m 0s"),
        ]
        
        # Create data rows
        for row_data in rows_data:
            row = self._create_table_row(row_data, is_track=True)
            self.table_layout.addWidget(row)
    
    def _create_pool_mining_table(self):
        """
        Create pool mining history table
        
        Table columns: Date Started, Rewards, Pool Contribution, Total Pool Rewards, Status
        """
        # Header definition
        columns = ["Date Started", "Rewards", "Pool Contribution", "Total Pool Rewards", "Status"]
        alignments = [
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        ]
        
        # Create header
        header = self._create_table_header(columns, alignments)
        header.setStyleSheet("border-bottom: 1px solid #E8E8E8;")
        self.table_layout.addWidget(header)
        
        # Sample data
        rows_data = [
            ("Feb 6, 2025", "5 $TAO", "5%", "250 $TAO", "Active"),
            ("Feb 5, 2025", "2 $TAO", "3%", "180 $TAO", "Active"),
            ("Feb 4, 2025", "2 $TAO", "4%", "200 $TAO", "Active"),
            ("Feb 3, 2025", "6 $TAO", "6%", "300 $TAO", "Completed"),
            ("Feb 2, 2025", "1 $TAO", "2%", "150 $TAO", "Completed"),
            ("Feb 1, 2025", "2 $TAO", "3%", "200 $TAO", "Completed"),
        ]
        
        # Create data rows
        for row_data in rows_data:
            row = self._create_table_row(row_data)
            self.table_layout.addWidget(row)


class ProfileScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ========== Scroll Area ==========
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)  # Content adapts to size
        scroll.setFrameShape(QFrame.Shape.NoFrame)  # No border
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Disable horizontal scroll
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Hide vertical scrollbar (still scrollable)
        
        # ========== Content Container ==========
        content = QWidget()
        content.setObjectName("content_box")
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(32, 32, 32, 32)  # Padding
        content_layout.setSpacing(24)  # Component spacing

        # ========== Page Title ==========
        title = QLabel("Overview")
        title.setStyleSheet("""
            font-size: 30px;
            font-weight: 600;
            color: #101828;
            letter-spacing: -0.75px;
        """)
        content_layout.addWidget(title)
        
        # ========== Stats Cards Container ==========
        stats_container = QWidget()
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(24)
        
        # ---------- Left Column (Claimable Reward + Two Small Cards) ----------
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(24)
        
        # Claimable reward card
        self.claimable_card = ClaimableRewardCard()
        self.claimable_card.setMinimumHeight(192)
        self.claimable_card.claim_clicked.connect(self._on_claim_clicked)
        left_layout.addWidget(self.claimable_card)
        
        # Small cards row container
        small_cards_row = QWidget()
        small_cards_layout = QHBoxLayout(small_cards_row)
        small_cards_layout.setContentsMargins(0, 0, 0, 0)
        small_cards_layout.setSpacing(24)
        
        # Total TAO rewards card
        self.total_rewards_card = SmallStatsCard("TOTAL $TAO REWARDS")
        self.total_rewards_card.setMinimumHeight(192)
        self.total_rewards_card.set_value("500.25", "$TAO")
        small_cards_layout.addWidget(self.total_rewards_card)
        
        # Cumulative runtime card (cyan icon, no dashed underline)
        self.runtime_card = SmallStatsCard("CUMULATIVE RUNTIME", icon_color="#71DADE", show_dashed=False)
        self.runtime_card.setMinimumHeight(192)
        self.runtime_card.set_value("48h 20m", "")
        small_cards_layout.addWidget(self.runtime_card)
        
        left_layout.addWidget(small_cards_row)
        
        stats_layout.addWidget(left_column, 1)  # stretch=1 means left column takes half width
        
        # ---------- Right Column (Stake & Unlock Rate Card) ----------
        self.stake_card = StakeUnlockCard()
        self.stake_card.setMinimumHeight(408)
        self.stake_card.stake_clicked.connect(self._on_stake_clicked)
        stats_layout.addWidget(self.stake_card, 1)  # stretch=1 means right column takes half width
        
        content_layout.addWidget(stats_container)
        
        # ========== Mining History Table ==========
        self.mining_table = MiningHistoryTable()
        content_layout.addWidget(self.mining_table)
        
        content_layout.addStretch()  # Bottom elastic space
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)
    
    def _on_claim_clicked(self):
        """
        Claim button click handler
        
        TODO: Implement claim reward functionality
        """
        print("Claim clicked")
    
    def _on_stake_clicked(self):
        """
        Stake button click handler
        
        TODO: Implement stake functionality
        """
        print("Stake clicked")
    
    def update_stats(self, claimable: str = "800.12", total_rewards: str = "500.25", 
                     runtime: str = "48h 20m"):
        """
        Update page displayed statistics
        
        Args:
            claimable: Claimable reward amount
            total_rewards: Total TAO reward amount
            runtime: Cumulative runtime
        """
        self.claimable_card.set_value(claimable)
        self.total_rewards_card.set_value(total_rewards, "$TAO")
        self.runtime_card.set_value(runtime, "")

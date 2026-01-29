from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath, QBrush
import math


CHART_TOOLTIP_STYLE = """
QWidget#tooltip_container {
    background-color: #150049;
    border-radius: 6px;
}
QLabel#stake_label {
    font-size: 12px;
    font-weight: 500;
    color: #FFFFFF;
    background: transparent;
}
QLabel#rate_label {
    font-size: 12px;
    font-weight: 600;
    color: #71DADE;
    background: transparent;
}
"""


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
        self.setStyleSheet(CHART_TOOLTIP_STYLE)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Container (with background and shadow)
        self.container = QWidget()
        self.container.setObjectName("tooltip_container")
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(12, 8, 12, 8)
        container_layout.setSpacing(4)
        
        # Stake amount label
        self.stake_label = QLabel("Stake: 0 $TAO")
        self.stake_label.setObjectName("stake_label")
        container_layout.addWidget(self.stake_label)
        
        # Unlock rate label
        self.rate_label = QLabel("Unlock Rate: 0%")
        self.rate_label.setObjectName("rate_label")
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

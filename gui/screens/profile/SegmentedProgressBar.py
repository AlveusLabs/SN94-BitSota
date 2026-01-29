from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor


class SegmentedProgressBar(QWidget):
    
    def __init__(self, liquid_ratio: float = 0.5, auto_staked_ratio: float = 0.5, parent=None):
        super().__init__(parent)
        self.setFixedHeight(16)
        self.liquid_ratio = liquid_ratio
        self.auto_staked_ratio = auto_staked_ratio
        
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

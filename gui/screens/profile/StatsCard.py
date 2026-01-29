from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


STATS_CARD_STYLE = """
QWidget#stats_card {
    background-color: #F6F5F8;
    border-radius: 4px;
}
"""


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
        self.setStyleSheet(STATS_CARD_STYLE)

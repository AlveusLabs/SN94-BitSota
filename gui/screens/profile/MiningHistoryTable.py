from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)

from gui.components import TabSwitcher


MINING_HISTORY_TABLE_STYLE = """
QLabel#table_title {
    font-size: 20px;
    font-weight: 600;
    color: #0C0029;
}
QLabel#header_label {
    font-size: 14px;
    font-weight: 400;
    color: rgba(12, 0, 41, 0.6);
    padding: 16px 0;
}
QLabel#cell_label {
    font-size: 14px;
    font-weight: 400;
    color: #0C0029;
    padding: 16px 0;
}
QLabel#cell_label_track {
    font-size: 14px;
    font-weight: 400;
    color: #4A148C;
    padding: 16px 0;
}
QWidget#table_row {
    border-bottom: 1px solid #F0F0F0;
}
QWidget#table_header {
    border-bottom: 1px solid #E8E8E8;
}
"""


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
        self.setStyleSheet(MINING_HISTORY_TABLE_STYLE)
        
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
        title.setObjectName("table_title")
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
        header.setObjectName("table_header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        for i, col in enumerate(columns):
            label = QLabel(col)
            label.setObjectName("header_label")
            
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
        row.setObjectName("table_row")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        for i, value in enumerate(data):
            label = QLabel(str(value))
            
            # Track column uses purple text
            if is_track and i == 3:
                label.setObjectName("cell_label_track")
            else:
                label.setObjectName("cell_label")
            
            # Last column right aligned
            if i == len(data) - 1:
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
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

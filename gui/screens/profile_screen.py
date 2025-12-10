from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QFrame,
)

from gui.components import PrimaryButton


class ProfileScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(24)

        main_layout.addStretch(1)

        title = QLabel("Total Overview")
        title.setObjectName("section_title")
        main_layout.addWidget(title)

        tab_container = QWidget()
        tab_layout = QHBoxLayout(tab_container)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        self.direct_tab = QPushButton("/ Direct Mining")
        self.direct_tab.setObjectName("tab_switcher_active")
        self.direct_tab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.direct_tab.clicked.connect(lambda: self._switch_tab("direct"))
        tab_layout.addWidget(self.direct_tab)

        self.pool_tab = QPushButton("/ Pool Mining")
        self.pool_tab.setObjectName("tab_switcher_inactive")
        self.pool_tab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pool_tab.clicked.connect(lambda: self._switch_tab("pool"))
        tab_layout.addWidget(self.pool_tab)

        tab_layout.addStretch()

        main_layout.addWidget(tab_container)

        self.table_container = QWidget()
        self.table_layout = QVBoxLayout(self.table_container)
        self.table_layout.setContentsMargins(0, 0, 0, 0)
        self.table_layout.setSpacing(0)

        self.direct_table = self._create_direct_mining_table()
        self.pool_table = self._create_pool_mining_table()

        self.table_layout.addWidget(self.direct_table)
        self.table_layout.addWidget(self.pool_table)
        self.direct_table.show()
        self.pool_table.hide()

        main_layout.addWidget(self.table_container)
        main_layout.addStretch(2)

    def _switch_tab(self, tab_type):
        if tab_type == "direct":
            self.direct_tab.setObjectName("tab_switcher_active")
            self.pool_tab.setObjectName("tab_switcher_inactive")
            self.direct_table.show()
            self.pool_table.hide()
        else:
            self.direct_tab.setObjectName("tab_switcher_inactive")
            self.pool_tab.setObjectName("tab_switcher_active")
            self.direct_table.hide()
            self.pool_table.show()

        self.direct_tab.setStyleSheet("")
        self.pool_tab.setStyleSheet("")
        self.direct_tab.style().unpolish(self.direct_tab)
        self.pool_tab.style().unpolish(self.pool_tab)
        self.direct_tab.style().polish(self.direct_tab)
        self.pool_tab.style().polish(self.pool_tab)

    def _create_direct_mining_table(self) -> QWidget:
        table = QWidget()
        table.setObjectName("content_box")
        table.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(table)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        header = self._create_table_header([
            "Mining Date",
            "Rewards",
            "Total Rewards Distributed",
            "Task Type",
            "Runtime",
            "Claim"
        ])
        layout.addWidget(header)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: rgba(21, 0, 73, 0.12); max-height: 1px;")
        layout.addWidget(separator)

        rows_data = [
            ("Feb 1, 2025", "5 $TAO", "-", "cifar10_binary", "2h 54m 0s", "Claimed"),
            ("Feb 2, 2025", "0.7 $TAO", "-", "cifar10_binary", "2h 54m 0s", "Claimed"),
            ("Feb 3, 2025", "1 $TAO", "-", "cifar10_binary", "2h 54m 0s", "Claim"),
            ("Feb 5, 2025", "2.3 $TAO", "-", "cifar10_binary", "2h 54m 0s", "Claimed"),
        ]

        for row_data in rows_data:
            row = self._create_table_row(row_data)
            layout.addWidget(row)

        return table

    def _create_pool_mining_table(self) -> QWidget:
        table = QWidget()
        table.setObjectName("content_box")
        table.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(table)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        header = self._create_table_header([
            "Date Started",
            "Rewards",
            "Pool Contribution",
            "Total Pool Rewards",
            "Pool Status",
            "Next Estimated Payout",
            "Claim"
        ])
        layout.addWidget(header)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: rgba(21, 0, 73, 0.12); max-height: 1px;")
        layout.addWidget(separator)

        rows_data = [
            ("Feb 1, 2025", "5 $TAO", "5%", "250 $TAO", "-", "Feb 1, 2025", "Claimed"),
            ("Feb 2, 2025", "0.7 $TAO", "5%", "250 $TAO", "-", "Feb 1, 2025", "Claimed"),
            ("Feb 3, 2025", "1 $TAO", "5%", "250 $TAO", "-", "Feb 1, 2025", "Claim"),
            ("Feb 5, 2025", "2.3 $TAO", "5%", "250 $TAO", "-", "Feb 1, 2025", "Claimed"),
        ]

        for row_data in rows_data:
            row = self._create_table_row(row_data)
            layout.addWidget(row)

        return table

    def _create_table_header(self, columns) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(16)

        for i, col in enumerate(columns):
            label = QLabel(col)
            label.setObjectName("form_label")
            if i == len(columns) - 1:
                label.setAlignment(Qt.AlignmentFlag.AlignRight)
                layout.addWidget(label, 0)
            else:
                layout.addWidget(label, 1)

        return header

    def _create_table_row(self, row_data) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(16)

        for i, data in enumerate(row_data):
            if i == len(row_data) - 1:
                if data == "Claim":
                    claim_btn = QPushButton(data)
                    claim_btn.setObjectName("primary_button")
                    claim_btn.setFixedSize(120, 40)
                    claim_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    claim_btn.clicked.connect(self._on_claim_clicked)
                    layout.addWidget(claim_btn, 0, Qt.AlignmentFlag.AlignRight)
                else:
                    label = QLabel(data)
                    label.setObjectName("stat_value")
                    label.setAlignment(Qt.AlignmentFlag.AlignRight)
                    label.setStyleSheet("color: rgba(21, 0, 73, 0.60);")
                    layout.addWidget(label, 0)
            else:
                label = QLabel(data)
                label.setObjectName("stat_value")
                layout.addWidget(label, 1)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: rgba(21, 0, 73, 0.08); max-height: 1px;")

        row_container = QWidget()
        row_layout = QVBoxLayout(row_container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        row_layout.addWidget(row)
        row_layout.addWidget(separator)

        return row_container

    def _on_claim_clicked(self):
        sender = self.sender()
        sender.setText("Claimed")
        sender.setEnabled(False)
        sender.setObjectName("secondary_button")
        sender.setStyleSheet("color: rgba(21, 0, 73, 0.60);")
        sender.style().unpolish(sender)
        sender.style().polish(sender)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QWidget
)
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path


class ComingSoonModal(QDialog):
    def __init__(self, screen_name: str, description: str, parent=None):
        super().__init__(parent)
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(600, 400)
        self.setup_ui(screen_name, description)

    def setup_ui(self, screen_name: str, description: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(24)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)

        title_label = QLabel("Coming Soon")
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        close_btn_widget = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn_widget.setFixedSize(24, 24)
        close_btn_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn_widget.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn_widget)

        layout.addLayout(header_layout)

        layout.addStretch()

        coming_soon_container = QWidget()
        coming_soon_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        coming_soon_container.setStyleSheet("""
            background-color: rgba(21, 0, 73, 0.04);
            border: 1px solid rgba(21, 0, 73, 0.12);
            border-radius: 8px;
        """)
        container_layout = QVBoxLayout(coming_soon_container)
        container_layout.setContentsMargins(24, 32, 24, 32)
        container_layout.setSpacing(16)

        screen_name_label = QLabel(screen_name)
        screen_name_label.setStyleSheet("""
            background: transparent;
            border: none;
            color: #150049;
            text-align: center;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 20px;
            font-weight: 500;
            line-height: 150%;
        """)
        screen_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(screen_name_label)

        description_label = QLabel(description)
        description_label.setStyleSheet("""
            background: transparent;
            border: none;
            color: rgba(21, 0, 73, 0.60);
            text-align: center;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
        """)
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        container_layout.addWidget(description_label)

        layout.addWidget(coming_soon_container)

        layout.addStretch()

        ok_btn = QPushButton("Got it")
        ok_btn.setObjectName("primary_button")
        ok_btn.setFixedHeight(48)
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

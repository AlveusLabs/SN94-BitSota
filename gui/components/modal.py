from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path


class ConfirmationModal(QDialog):
    confirmed = Signal()
    cancelled = Signal()

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(600, 300)
        self.setup_ui(title, message)

    def setup_ui(self, title: str, message: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(24)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)

        title_label = QLabel(title)
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        close_btn_widget = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn_widget.setFixedSize(24, 24)
        close_btn_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn_widget.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn_widget)

        layout.addLayout(header_layout)

        message_label = QLabel(message)
        message_label.setObjectName("modal_message")
        message_label.setWordWrap(True)
        layout.addWidget(message_label)

        layout.addStretch()

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(16)

        yes_btn = QPushButton("Yes")
        yes_btn.setObjectName("secondary_button")
        yes_btn.setFixedHeight(48)
        yes_btn.clicked.connect(self._on_yes)
        buttons_layout.addWidget(yes_btn, 1)

        no_btn = QPushButton("No")
        no_btn.setObjectName("primary_button")
        no_btn.setFixedHeight(48)
        no_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(no_btn, 1)

        layout.addLayout(buttons_layout)

    def _on_yes(self):
        self.confirmed.emit()
        self.accept()

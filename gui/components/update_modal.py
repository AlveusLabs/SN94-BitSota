from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path


class UpdateAvailableModal(QDialog):
    download_clicked = Signal()
    skip_clicked = Signal()

    def __init__(self, update_info: dict, parent=None):
        super().__init__(parent)
        self.update_info = update_info
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(600, 400)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(24)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)

        title_label = QLabel("New Update Available")
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        close_btn_widget = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn_widget.setFixedSize(24, 24)
        close_btn_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn_widget.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn_widget)

        layout.addLayout(header_layout)

        version_info = QLabel(
            f"Current version: {self.update_info['current_version']}\n"
            f"New version: {self.update_info['new_version']}"
        )
        version_info.setObjectName("modal_message")
        layout.addWidget(version_info)

        if self.update_info.get('description'):
            whats_new_label = QLabel("What's new:")
            whats_new_label.setObjectName("modal_subtitle")
            layout.addWidget(whats_new_label)

            description = QLabel(self.update_info['description'])
            description.setObjectName("modal_message")
            description.setWordWrap(True)
            layout.addWidget(description)

        layout.addStretch()

        note_label = QLabel(
            "Note: After downloading, quit BitSota and install the new version from your Downloads folder."
        )
        note_label.setObjectName("modal_note")
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #74c0fc; font-size: 12px;")
        layout.addWidget(note_label)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(16)

        skip_btn = QPushButton("Skip This Version")
        skip_btn.setObjectName("secondary_button")
        skip_btn.setFixedHeight(48)
        skip_btn.clicked.connect(self._on_skip)
        buttons_layout.addWidget(skip_btn, 1)

        later_btn = QPushButton("Remind Me Later")
        later_btn.setObjectName("secondary_button")
        later_btn.setFixedHeight(48)
        later_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(later_btn, 1)

        download_btn = QPushButton("Download Update")
        download_btn.setObjectName("primary_button")
        download_btn.setFixedHeight(48)
        download_btn.clicked.connect(self._on_download)
        buttons_layout.addWidget(download_btn, 1)

        layout.addLayout(buttons_layout)

    def _on_download(self):
        self.download_clicked.emit()
        self.accept()

    def _on_skip(self):
        self.skip_clicked.emit()
        self.accept()

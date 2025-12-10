from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtSvgWidgets import QSvgWidget

from gui.components import PrimaryButton
from gui.resource_path import resource_path


class StartScreen(QWidget):
    start_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("start_screen")
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addStretch()

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(90, 0, 90, 0)
        content_layout.setSpacing(0)

        left_box = QWidget()
        left_box_layout = QVBoxLayout(left_box)
        left_box_layout.setContentsMargins(0, 0, 0, 0)
        left_box_layout.setSpacing(32)
        left_box_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self.logo = QSvgWidget(resource_path("gui/images/logo.svg"))
        self.logo.setFixedSize(226, 57)
        left_box_layout.addWidget(self.logo)

        self.tagline = QLabel("Fueling the future of decentralized research")
        self.tagline.setObjectName("start_tagline")
        left_box_layout.addWidget(self.tagline)

        self.start_button = PrimaryButton("Start")
        self.start_button.clicked.connect(self.start_clicked.emit)
        left_box_layout.addWidget(self.start_button)

        content_layout.addWidget(left_box, 0, Qt.AlignmentFlag.AlignVCenter)
        content_layout.addStretch()

        self.right_image = QSvgWidget(resource_path("gui/images/start_logo.svg"))
        self.right_image.setFixedSize(634, 264)
        content_layout.addWidget(self.right_image, 0, Qt.AlignmentFlag.AlignVCenter)

        main_layout.addLayout(content_layout)
        main_layout.addStretch()

        self.frame_icon = QSvgWidget(resource_path("gui/images/frame.svg"), self)
        self.frame_icon.setFixedSize(40, 40)
        self.frame_icon.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'frame_icon'):
            self.frame_icon.move(self.width() - 160, self.height() - 100)

from PySide6.QtWidgets import QPushButton, QHBoxLayout, QLabel
from PySide6.QtCore import QSize, Qt
from PySide6.QtSvgWidgets import QSvgWidget


class PrimaryButton(QPushButton):
    def __init__(self, text: str, width: int = 200, height: int = 48, icon_path: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("primary_button")
        self.setFixedSize(QSize(width, height))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.icon_path = icon_path
        self.icon_widget = None
        self.text_label_widget = None

        if icon_path:
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.icon_widget = QSvgWidget(icon_path)
            self.icon_widget.setFixedSize(16, 16)
            layout.addWidget(self.icon_widget)

            self.text_label_widget = QLabel(text)
            self.text_label_widget.setStyleSheet("background: transparent; border: none; color: #8EFBFF;")
            layout.addWidget(self.text_label_widget)
        else:
            self.setText(text)

    def update_icon(self, new_icon_path: str):
        if self.icon_widget:
            self.icon_widget.load(new_icon_path)

    def update_text(self, new_text: str):
        if self.text_label_widget:
            self.text_label_widget.setText(new_text)
        else:
            self.setText(new_text)


class SecondaryButton(QPushButton):
    def __init__(self, text: str, width: int = 200, height: int = 48, parent=None):
        super().__init__(text, parent)
        self.setObjectName("secondary_button")
        self.setFixedSize(QSize(width, height))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

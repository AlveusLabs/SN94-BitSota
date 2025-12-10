from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QWidget
)
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path


class ErrorModal(QDialog):
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

        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("primary_button")
        ok_btn.setFixedHeight(48)
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)



class TermsAcceptanceModal(QDialog):
    confirmed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(600, 500)
        self.terms_checkbox = None
        self.info_checkbox = None
        self.confirm_button = None
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

        title_label = QLabel("Import Existing Wallet")
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        close_btn_widget = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn_widget.setFixedSize(24, 24)
        close_btn_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn_widget.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn_widget)

        layout.addLayout(header_layout)

        important_container = QWidget()
        important_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        important_container.setStyleSheet("""
            QWidget {
                background-color: rgba(21, 0, 73, 0.04);
                border-radius: 8px;
            }
        """)
        important_container_layout = QVBoxLayout(important_container)
        important_container_layout.setContentsMargins(16, 16, 16, 16)
        important_container_layout.setSpacing(16)

        important_header = QHBoxLayout()
        important_header.setSpacing(8)

        important_icon = QSvgWidget(resource_path("gui/images/Info Circle.svg"))
        important_icon.setFixedSize(20, 20)
        important_header.addWidget(important_icon)

        important_label = QLabel("Important")
        important_label.setStyleSheet("""
            background-color: transparent;
            color: #150049;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
        """)
        important_header.addWidget(important_label)
        important_header.addStretch()

        important_container_layout.addLayout(important_header)

        info_items = [
            "Please make sure you're importing the correct wallet. For the best experience, we recommend avoiding wallets that hold large balances.",
            "Your seed phrase always stays on your device — BitSota never sees or stores it.",
            "If you ever lose access to your keys, we won't be able to recover them for you.",
            "Not sure which wallet to use? Creating a new one is always the safest option."
        ]

        for item in info_items:
            item_widget = QWidget()
            item_widget.setStyleSheet("QWidget { background-color: transparent; }")
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(8)
            item_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

            bullet = QLabel("•")
            bullet.setStyleSheet("""
                background-color: transparent;
                color: rgba(21, 0, 73, 0.60);
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
            """)
            bullet.setFixedWidth(20)
            item_layout.addWidget(bullet)

            item_label = QLabel(item)
            item_label.setStyleSheet("""
                background-color: transparent;
                color: rgba(21, 0, 73, 0.60);
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
                line-height: 150%;
            """)
            item_label.setWordWrap(True)
            item_layout.addWidget(item_label, 1)

            important_container_layout.addWidget(item_widget)

        layout.addWidget(important_container)

        layout.addStretch()

        self.terms_checkbox = QCheckBox("I agree to the Terms of Service & Privacy Policy")
        self.terms_checkbox.stateChanged.connect(self._update_confirm_button)
        layout.addWidget(self.terms_checkbox)

        self.info_checkbox = QCheckBox("I understand and accept the information above")
        self.info_checkbox.stateChanged.connect(self._update_confirm_button)
        layout.addWidget(self.info_checkbox)

        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.setObjectName("confirm_button_disabled")
        self.confirm_button.setFixedSize(207, 48)
        self.confirm_button.clicked.connect(self._on_confirm)
        self.confirm_button.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.confirm_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

    def _update_confirm_button(self):
        if self.terms_checkbox.isChecked() and self.info_checkbox.isChecked():
            self.confirm_button.setObjectName("confirm_button_enabled")
            self.confirm_button.setEnabled(True)
        else:
            self.confirm_button.setObjectName("confirm_button_disabled")
            self.confirm_button.setEnabled(False)

        self.confirm_button.style().unpolish(self.confirm_button)
        self.confirm_button.style().polish(self.confirm_button)

    def _on_confirm(self):
        if self.terms_checkbox.isChecked() and self.info_checkbox.isChecked():
            self.confirmed.emit()
            self.accept()


class WalletImportedSuccessModal(QDialog):
    start_mining = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(600, 400)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(32)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)

        title_label = QLabel("Wallet Imported Successfully")
        title_label.setObjectName("success_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        close_btn_widget = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn_widget.setFixedSize(24, 24)
        close_btn_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn_widget.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn_widget)

        layout.addLayout(header_layout)

        layout.addStretch()

        success_content = QVBoxLayout()
        success_content.setSpacing(24)
        success_content.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_layout = QHBoxLayout()
        icon_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        wallet_check_icon = QSvgWidget(resource_path("gui/images/wallet_check.svg"))
        wallet_check_icon.setFixedSize(120, 120)
        icon_layout.addWidget(wallet_check_icon)

        success_content.addLayout(icon_layout)

        message_widget = QWidget()
        message_widget.setStyleSheet("QWidget { background-color: transparent; }")
        message_layout = QVBoxLayout(message_widget)
        message_layout.setContentsMargins(0, 0, 0, 0)
        message_layout.setSpacing(8)

        message1 = QLabel("Your wallet has been successfully imported!")
        message1.setStyleSheet("""
            background-color: transparent;
            color: rgba(21, 0, 73, 0.80);
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
        """)
        message1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message1.setWordWrap(True)
        message_layout.addWidget(message1)

        message2 = QLabel("You can now securely manage your assets and start using all features.")
        message2.setStyleSheet("""
            background-color: transparent;
            color: rgba(21, 0, 73, 0.80);
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
        """)
        message2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message2.setWordWrap(True)
        message_layout.addWidget(message2)

        success_content.addWidget(message_widget)

        layout.addLayout(success_content)

        layout.addStretch()

        start_mining_button = QPushButton("Start Mining")
        start_mining_button.setObjectName("primary_button")
        start_mining_button.setFixedSize(207, 48)
        start_mining_button.clicked.connect(self._on_start_mining)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(start_mining_button)

        layout.addLayout(button_layout)

    def _on_start_mining(self):
        self.start_mining.emit()
        self.accept()

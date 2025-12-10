from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QWidget
)
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path


class ColdkeyAddressModal(QDialog):
    address_submitted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(600, 400)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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

        title_label = QLabel("Coldkey Address Required")
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        close_btn_widget = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn_widget.setFixedSize(24, 24)
        close_btn_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn_widget.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn_widget)

        layout.addLayout(header_layout)

        info_text = QLabel(
            "To receive mining rewards, please provide your coldkey address. "
            "Your coldkey is used for transactions and reward payments."
        )
        info_text.setObjectName("modal_message")
        info_text.setWordWrap(True)
        layout.addWidget(info_text)

        address_label = QLabel("Coldkey Address")
        address_label.setObjectName("form_label")
        layout.addWidget(address_label)

        self.address_input = QLineEdit()
        self.address_input.setObjectName("form_input")
        self.address_input.setPlaceholderText("Enter your coldkey address (starts with 5)")
        layout.addWidget(self.address_input)

        layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.setSpacing(16)

        submit_btn = QPushButton("Submit")
        submit_btn.setObjectName("primary_button")
        submit_btn.setFixedSize(207, 48)
        submit_btn.clicked.connect(self._on_submit)
        button_layout.addWidget(submit_btn)

        skip_btn = QPushButton("Skip for Now")
        skip_btn.setObjectName("secondary_button")
        skip_btn.setFixedSize(207, 48)
        skip_btn.clicked.connect(self.reject)
        button_layout.addWidget(skip_btn)

        button_layout.addStretch()

        layout.addLayout(button_layout)

    def _on_submit(self):
        address = self.address_input.text().strip()

        if not address:
            from gui.components.import_confirmation_modals import ErrorModal
            error_modal = ErrorModal(
                "Empty Address",
                "Please enter a coldkey address.",
                parent=self
            )
            error_modal.exec()
            return

        from gui.wallet_utils_gui import validate_coldkey_address
        is_valid, error_message = validate_coldkey_address(address)

        if not is_valid:
            from gui.components.import_confirmation_modals import ErrorModal
            error_modal = ErrorModal(
                "Invalid Address",
                error_message,
                parent=self
            )
            error_modal.exec()
            return

        self.address_submitted.emit(address)
        self.accept()

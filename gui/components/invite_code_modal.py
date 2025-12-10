from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QWidget
)
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path
import requests
import time
import logging

logger = logging.getLogger(__name__)


class InviteCodeWorker(QThread):
    success = Signal()
    error = Signal(str)

    def __init__(self, code: str, relay_url: str, wallet, coldkey_address: str = None):
        super().__init__()
        self.code = code
        self.relay_url = relay_url.rstrip('/')
        self.wallet = wallet
        self.coldkey_address = coldkey_address

    def run(self):
        try:
            msg = f"auth:{int(time.time())}"
            sig = self.wallet.hotkey.sign(msg).hex()

            payload = {"code": self.code}
            if self.coldkey_address:
                payload["coldkey_address"] = self.coldkey_address

            response = requests.post(
                f"{self.relay_url}/invitation_code/link",
                json=payload,
                headers={
                    "X-Key": self.wallet.hotkey.ss58_address,
                    "X-Signature": sig,
                    "X-Timestamp": msg
                },
                timeout=10
            )

            response.raise_for_status()
            result = response.json()

            if result.get("data") == 1:
                self.success.emit()
            else:
                self.error.emit("Verification failed, please try again.")
        except requests.exceptions.Timeout:
            self.error.emit("Request timed out. Please check your connection.")
        except requests.exceptions.ConnectionError:
            self.error.emit("Cannot connect to server. Please try again later.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                self.error.emit("Invalid or already used code.")
            else:
                self.error.emit(f"Server error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Invite code verification error: {e}")
            self.error.emit("Verification failed, please try again.")


class InviteCodeModal(QDialog):
    code_verified = Signal()

    STATE_ENTERING = "entering"
    STATE_ENTERED = "entered"
    STATE_ERROR = "error"
    STATE_SUCCESS = "success"

    def __init__(self, relay_url: str, wallet=None, coldkey_address: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(440, 480)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.relay_url = relay_url
        self.wallet = wallet
        self.coldkey_address = coldkey_address
        self.current_state = self.STATE_ENTERING
        self.code_inputs = []
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 20, 20, 0)
        header_layout.setSpacing(8)

        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)

        self.title_label = QLabel("Invite Code")
        self.title_label.setObjectName("modal_title")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        close_btn_widget = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn_widget.setFixedSize(24, 24)
        close_btn_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn_widget.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn_widget)

        self.main_layout.addLayout(header_layout)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(32, 32, 32, 32)
        self.content_layout.setSpacing(24)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.main_layout.addWidget(self.content_widget, 1)

        self._setup_entering_state()

    def _setup_entering_state(self):
        self._clear_content()

        self.icon_widget = QSvgWidget(resource_path("gui/images/invite_code_modal.svg"))
        self.icon_widget.setFixedSize(130, 130)
        self.content_layout.addWidget(self.icon_widget, 0, Qt.AlignmentFlag.AlignCenter)

        self.message_label = QLabel("Your Invite Code")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message_label.setStyleSheet("""
            color: #150049;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
        """)
        self.content_layout.addWidget(self.message_label)

        self.subtitle_label = QLabel("You're early. Unlock mining access with an invite code dropped during our Twitter events and giveaways")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("""
            color: rgba(21, 0, 73, 0.60);
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 12px;
            font-weight: 400;
            line-height: 150%;
        """)
        self.content_layout.addWidget(self.subtitle_label)

        code_input_layout = QHBoxLayout()
        code_input_layout.setSpacing(12)
        self.code_inputs = []

        for i in range(8):
            code_input = QLineEdit()
            code_input.setMaxLength(1)
            code_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
            code_input.setFixedSize(40, 48)
            code_input.setStyleSheet("""
                QLineEdit {
                    background-color: #FFFFFF;
                    color: #150049;
                    border: 1px solid rgba(21, 0, 73, 0.12);
                    border-radius: 4px;
                    font-family: "JetBrains Mono", monospace;
                    font-size: 18px;
                    font-weight: 500;
                    padding: 0;
                }
                QLineEdit:focus {
                    border: 2px solid #150049;
                }
            """)
            code_input.textChanged.connect(lambda text, idx=i: self._on_code_input_changed(text, idx))
            self.code_inputs.append(code_input)
            code_input_layout.addWidget(code_input)

        self.content_layout.addLayout(code_input_layout)

        self.unlock_btn = QPushButton("Unlock Now")
        self.unlock_btn.setObjectName("primary_button")
        self.unlock_btn.setFixedHeight(48)
        self.unlock_btn.setEnabled(False)
        self.unlock_btn.clicked.connect(self._on_unlock_clicked)
        self.content_layout.addWidget(self.unlock_btn)

    def _setup_entered_state(self):
        self.unlock_btn.setEnabled(False)
        self.unlock_btn.setText("Verifying...")
        for input_field in self.code_inputs:
            input_field.setEnabled(False)

    def _setup_error_state(self, error_message: str):
        self._clear_content()

        icon_widget = QSvgWidget(resource_path("gui/images/invite_code_modal.svg"))
        icon_widget.setFixedSize(130, 130)
        self.content_layout.addWidget(icon_widget, 0, Qt.AlignmentFlag.AlignCenter)

        message_label = QLabel("Your Invite Code")
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setStyleSheet("""
            color: #150049;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
        """)
        self.content_layout.addWidget(message_label)

        subtitle_label = QLabel("You're early. Unlock mining access with an invite code dropped during our Twitter events and giveaways")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("""
            color: rgba(21, 0, 73, 0.60);
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 12px;
            font-weight: 400;
            line-height: 150%;
        """)
        self.content_layout.addWidget(subtitle_label)

        code_input_layout = QHBoxLayout()
        code_input_layout.setSpacing(8)

        code_value = self._get_code_value()
        for i, char in enumerate(code_value):
            code_display = QLabel(char)
            code_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
            code_display.setFixedSize(40, 48)
            code_display.setStyleSheet("""
                background-color: #FFFFFF;
                color: #150049;
                border: 1px solid rgba(21, 0, 73, 0.12);
                border-radius: 4px;
                font-family: "JetBrains Mono", monospace;
                font-size: 18px;
                font-weight: 500;
            """)
            code_input_layout.addWidget(code_display)

        self.content_layout.addLayout(code_input_layout)

        error_label = QLabel(f"⚠ {error_message}")
        error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        error_label.setWordWrap(True)
        error_label.setStyleSheet("""
            color: #FF0000;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 12px;
            font-weight: 400;
        """)
        self.content_layout.addWidget(error_label)

        retry_btn = QPushButton("Try Again")
        retry_btn.setObjectName("primary_button")
        retry_btn.setFixedHeight(48)
        retry_btn.clicked.connect(self._reset_to_entering_state)
        self.content_layout.addWidget(retry_btn)

    def _setup_success_state(self):
        self._clear_content()

        self.content_layout.setSpacing(16)

        check_icon = QSvgWidget(resource_path("gui/images/check_square.svg"))
        check_icon.setFixedSize(64, 64)
        self.content_layout.addWidget(check_icon, 0, Qt.AlignmentFlag.AlignCenter)

        success_title = QLabel("Unlocked Successfully")
        success_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        success_title.setStyleSheet("""
            color: #150049;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 20px;
            font-weight: 500;
        """)
        self.content_layout.addWidget(success_title)

        success_message = QLabel("Welcome to the money printer. It's already running")
        success_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        success_message.setWordWrap(True)
        success_message.setStyleSheet("""
            color: rgba(21, 0, 73, 0.60);
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 400;
            line-height: 150%;
        """)
        self.content_layout.addWidget(success_message)

        start_mining_btn = QPushButton("Start Mining")
        start_mining_btn.setObjectName("primary_button")
        start_mining_btn.setFixedHeight(48)
        start_mining_btn.clicked.connect(self._on_start_mining)
        self.content_layout.addWidget(start_mining_btn)

    def _clear_content(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_code_input_changed(self, text: str, index: int):
        if text and index < 7:
            self.code_inputs[index + 1].setFocus()

        all_filled = all(input_field.text() for input_field in self.code_inputs)
        self.unlock_btn.setEnabled(all_filled)

    def _get_code_value(self) -> str:
        return ''.join(input_field.text().upper() for input_field in self.code_inputs)

    def _on_unlock_clicked(self):
        if not self.wallet:
            self._setup_error_state("No wallet connected. Please connect a wallet first.")
            return

        code = self._get_code_value()
        self.current_state = self.STATE_ENTERED
        self._setup_entered_state()

        self.worker = InviteCodeWorker(code, self.relay_url, self.wallet, self.coldkey_address)
        self.worker.success.connect(self._on_verification_success)
        self.worker.error.connect(self._on_verification_error)
        self.worker.start()

    def _on_verification_success(self):
        self.current_state = self.STATE_SUCCESS
        self._setup_success_state()

    def _on_verification_error(self, error_message: str):
        self.current_state = self.STATE_ERROR
        self._setup_error_state(error_message)

    def _reset_to_entering_state(self):
        self.current_state = self.STATE_ENTERING
        self._setup_entering_state()

    def _on_start_mining(self):
        self.code_verified.emit()
        self.accept()

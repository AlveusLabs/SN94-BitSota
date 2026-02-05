from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QWidget
)
from PySide6.QtSvgWidgets import QSvgWidget

from gui.resource_path import resource_path
from gui.theme import BitSOTATheme
from gui.components.common.button import PrimaryButton, SecondaryButton
from gui.components.common.overlay import show_modal_with_overlay


class ColdkeyAddressModal(QDialog):
    """Coldkey Address input modal dialog component"""
    
    # Constants
    MODAL_WIDTH = 600
    MODAL_HEIGHT = 400
    MODAL_PADDING = 32
    HEADER_SPACING = 8
    CONTENT_SPACING = 24
    BUTTON_HEIGHT = 48
    BUTTON_WIDTH = 207
    
    address_submitted = Signal(str)
    
    @staticmethod
    def get_stylesheet():
        """Get the stylesheet for ColdkeyAddressModal component"""
        return f"""
            QDialog#modal_dialog {{
                background-color: {BitSOTATheme.CONTENT_BOX_BG};
                border: none;
                border-radius: 4px;
            }}
            
            QLabel#modal_title {{
                color: {BitSOTATheme.BLACK100};
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 24px;
                font-weight: 600;
            }}
            
            QLabel#modal_message {{
                color: {BitSOTATheme.BLACK60};
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
                line-height: 150%;
            }}
            
            QLabel#form_label {{
                color: {BitSOTATheme.BLACK60};
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 500;
            }}
        """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup dialog properties
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(self.MODAL_WIDTH, self.MODAL_HEIGHT)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        # Apply styles and setup UI
        self.setStyleSheet(self.get_stylesheet())
        self.setup_ui()

    def setup_ui(self):
        """Initialize the modal UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.MODAL_PADDING, 24, self.MODAL_PADDING, 24)
        layout.setSpacing(self.CONTENT_SPACING)
        
        # Header
        header_layout = self._create_header()
        layout.addLayout(header_layout)
        
        # Info message
        info_text = QLabel(
            "To receive mining rewards, please provide your coldkey address. "
            "Your coldkey is used for transactions and reward payments."
        )
        info_text.setObjectName("modal_message")
        info_text.setWordWrap(True)
        layout.addWidget(info_text)
        
        # Form label
        address_label = QLabel("Coldkey Address")
        address_label.setObjectName("form_label")
        layout.addWidget(address_label)
        
        # Input field
        self.address_input = QLineEdit()
        self.address_input.setObjectName("form_input")
        self.address_input.setPlaceholderText("Enter your coldkey address (starts with 5)")
        layout.addWidget(self.address_input)
        
        layout.addStretch()
        
        # Action buttons
        button_layout = self._create_buttons()
        layout.addLayout(button_layout)
    
    def _create_header(self) -> QHBoxLayout:
        """Create modal header with title and close button"""
        header_layout = QHBoxLayout()
        header_layout.setSpacing(self.HEADER_SPACING)
        
        # Title icon
        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)
        
        # Title
        title_label = QLabel("Coldkey Address Required")
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Close button
        close_btn = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn)
        
        return header_layout
    
    def _create_buttons(self) -> QHBoxLayout:
        """Create action buttons"""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(16)
        
        # Submit button
        self.submit_btn = PrimaryButton(
            "Submit",
            width=self.BUTTON_WIDTH,
            height=self.BUTTON_HEIGHT
        )
        self.submit_btn.clicked.connect(self._on_submit)
        button_layout.addWidget(self.submit_btn)
        
        # Skip button
        self.skip_btn = SecondaryButton(
            "Skip for Now",
            width=self.BUTTON_WIDTH,
            height=self.BUTTON_HEIGHT
        )
        self.skip_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.skip_btn)
        
        button_layout.addStretch()
        return button_layout

    def _on_submit(self):
        address = self.address_input.text().strip()

        if not address:
            from gui.components.modals.import_confirmation import ErrorModal
            error_modal = ErrorModal(
                "Empty Address",
                "Please enter a coldkey address.",
                parent=self
            )
            show_modal_with_overlay(error_modal, self)
            return

        from gui.wallet_utils_gui import validate_coldkey_address
        is_valid, error_message = validate_coldkey_address(address)

        if not is_valid:
            from gui.components.modals.import_confirmation import ErrorModal
            error_modal = ErrorModal(
                "Invalid Address",
                error_message,
                parent=self
            )
            show_modal_with_overlay(error_modal, self)
            return

        self.address_submitted.emit(address)
        self.accept()

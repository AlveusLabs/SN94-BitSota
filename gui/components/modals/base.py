from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget
from PySide6.QtSvgWidgets import QSvgWidget

from gui.resource_path import resource_path
from gui.theme import BitSOTATheme
from gui.components.common.button import PrimaryButton, SecondaryButton


class ConfirmationModal(QDialog):
    """Confirmation modal dialog component"""
    
    # Constants
    MODAL_WIDTH = 600
    MODAL_HEIGHT = 300
    MODAL_PADDING = 32
    HEADER_SPACING = 8
    CONTENT_SPACING = 24
    BUTTON_HEIGHT = 48
    
    confirmed = Signal()
    cancelled = Signal()
    
    @staticmethod
    def get_stylesheet():
        """Get the stylesheet for ConfirmationModal component"""
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
        """

    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self.title_text = title
        self.message_text = message
        
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
        
        # Message
        message_label = QLabel(self.message_text)
        message_label.setObjectName("modal_message")
        message_label.setWordWrap(True)
        layout.addWidget(message_label)
        
        layout.addStretch()
        
        # Action buttons
        buttons_layout = self._create_buttons()
        layout.addLayout(buttons_layout)
    
    def _create_header(self) -> QHBoxLayout:
        """Create modal header with title and close button"""
        header_layout = QHBoxLayout()
        header_layout.setSpacing(self.HEADER_SPACING)
        
        # Title icon
        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)
        
        # Title
        title_label = QLabel(self.title_text)
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
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(16)
        
        # Calculate button width for 2 buttons
        button_width = (self.MODAL_WIDTH - self.MODAL_PADDING * 2 - 16) // 2
        
        # Yes button
        yes_btn = SecondaryButton(
            "Yes",
            width=button_width,
            height=self.BUTTON_HEIGHT
        )
        yes_btn.clicked.connect(self._on_yes)
        buttons_layout.addWidget(yes_btn)
        
        # No button
        no_btn = PrimaryButton(
            "No",
            width=button_width,
            height=self.BUTTON_HEIGHT
        )
        no_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(no_btn)
        
        return buttons_layout
    
    def _on_yes(self):
        """Handle Yes button click"""
        self.confirmed.emit()
        self.accept()

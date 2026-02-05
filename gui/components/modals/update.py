from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget
from PySide6.QtSvgWidgets import QSvgWidget

from gui.resource_path import resource_path
from gui.theme import BitSOTATheme
from gui.components.common.button import PrimaryButton, SecondaryButton


class UpdateAvailableModal(QDialog):
    """Update Available modal dialog component"""
    
    # Constants
    MODAL_WIDTH = 600
    MODAL_HEIGHT = 400
    MODAL_PADDING = 32
    HEADER_SPACING = 8
    CONTENT_SPACING = 24
    BUTTON_HEIGHT = 48
    
    download_clicked = Signal()
    skip_clicked = Signal()
    
    @staticmethod
    def get_stylesheet():
        """Get the stylesheet for UpdateAvailableModal component"""
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
            
            QLabel#modal_subtitle {{
                color: {BitSOTATheme.BLACK100};
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 16px;
                font-weight: 500;
            }}
            
            QLabel#modal_note {{
                color: {BitSOTATheme.COLOR1_60};
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 12px;
                font-weight: 400;
                font-style: italic;
            }}
        """

    def __init__(self, update_info: dict, parent=None):
        super().__init__(parent)
        self.update_info = update_info
        
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
        
        # Version info
        version_info = QLabel(
            f"Current version: {self.update_info['current_version']}\n"
            f"New version: {self.update_info['new_version']}"
        )
        version_info.setObjectName("modal_message")
        layout.addWidget(version_info)
        
        # What's new section
        if self.update_info.get('description'):
            whats_new_label = QLabel("What's new:")
            whats_new_label.setObjectName("modal_subtitle")
            layout.addWidget(whats_new_label)
            
            description = QLabel(self.update_info['description'])
            description.setObjectName("modal_message")
            description.setWordWrap(True)
            layout.addWidget(description)
        
        layout.addStretch()
        
        # Note
        note_label = QLabel(
            "Note: After downloading, quit BitSota and install the new version from your Downloads folder."
        )
        note_label.setObjectName("modal_note")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)
        
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
        title_label = QLabel("New Update Available")
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
        
        # Calculate button width for 3 buttons
        button_width = (self.MODAL_WIDTH - self.MODAL_PADDING * 2 - 16 * 2) // 3
        
        # Skip button
        skip_btn = SecondaryButton(
            "Skip This Version",
            width=button_width,
            height=self.BUTTON_HEIGHT
        )
        skip_btn.clicked.connect(self._on_skip)
        buttons_layout.addWidget(skip_btn)
        
        # Later button
        later_btn = SecondaryButton(
            "Remind Me Later",
            width=button_width,
            height=self.BUTTON_HEIGHT
        )
        later_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(later_btn)
        
        # Download button
        download_btn = PrimaryButton(
            "Download Update",
            width=button_width,
            height=self.BUTTON_HEIGHT
        )
        download_btn.clicked.connect(self._on_download)
        buttons_layout.addWidget(download_btn)
        
        return buttons_layout
    
    def _on_download(self):
        """Handle download button click"""
        self.download_clicked.emit()
        self.accept()
    
    def _on_skip(self):
        """Handle skip button click"""
        self.skip_clicked.emit()
        self.accept()

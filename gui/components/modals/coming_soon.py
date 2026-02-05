from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget
)
from PySide6.QtSvgWidgets import QSvgWidget

from gui.resource_path import resource_path
from gui.theme import BitSOTATheme
from gui.components.common.button import PrimaryButton


class ComingSoonModal(QDialog):
    """Coming Soon modal dialog component"""
    
    # Constants
    MODAL_WIDTH = 600
    MODAL_HEIGHT = 400
    MODAL_PADDING = 32
    HEADER_SPACING = 8
    CONTENT_SPACING = 24
    BUTTON_HEIGHT = 48
    
    @staticmethod
    def get_stylesheet():
        """Get the stylesheet for ComingSoonModal component"""
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
            
            QWidget#coming_soon_container {{
                background-color: {BitSOTATheme.COLOR1_04};
                border: 1px solid {BitSOTATheme.COLOR1_12};
                border-radius: 8px;
            }}
            
            QLabel#feature_name {{
                background-color: transparent;
                color: {BitSOTATheme.COLOR1};
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 20px;
                font-weight: 500;
                line-height: 150%;
            }}
            
            QLabel#feature_description {{
                background-color: transparent;
                color: {BitSOTATheme.COLOR1_60};
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
                line-height: 150%;
            }}
        """
    
    def __init__(self, screen_name: str, description: str, parent=None):
        super().__init__(parent)
        self.screen_name = screen_name
        self.description = description
        
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
        
        layout.addStretch()
        
        # Coming soon content container
        coming_soon_container = self._create_content_container()
        layout.addWidget(coming_soon_container)
        
        layout.addStretch()
        
        # Action button
        self.got_it_btn = PrimaryButton(
            "Got it",
            width=self.MODAL_WIDTH - (self.MODAL_PADDING * 2),
            height=self.BUTTON_HEIGHT
        )
        self.got_it_btn.clicked.connect(self.accept)
        layout.addWidget(self.got_it_btn)
    
    def _create_header(self) -> QHBoxLayout:
        """Create modal header with title and close button"""
        header_layout = QHBoxLayout()
        header_layout.setSpacing(self.HEADER_SPACING)
        
        # Title icon
        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)
        
        # Title
        title_label = QLabel("Coming Soon")
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
    
    def _create_content_container(self) -> QWidget:
        """Create coming soon content container with feature info"""
        container = QWidget()
        container.setObjectName("coming_soon_container")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(24, 32, 24, 32)
        container_layout.setSpacing(16)
        
        # Feature name
        feature_name = QLabel(self.screen_name)
        feature_name.setObjectName("feature_name")
        feature_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(feature_name)
        
        # Feature description
        feature_desc = QLabel(self.description)
        feature_desc.setObjectName("feature_description")
        feature_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feature_desc.setWordWrap(True)
        container_layout.addWidget(feature_desc)
        
        return container

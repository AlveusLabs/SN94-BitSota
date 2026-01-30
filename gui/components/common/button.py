from PySide6.QtWidgets import QPushButton, QHBoxLayout, QLabel, QWidget
from PySide6.QtCore import QSize, Qt
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QPixmap, QPainter
from gui.theme import BitSOTATheme


class PrimaryButton(QPushButton):
    """Primary button component with icon support based on Figma design."""
    
    @staticmethod
    def get_stylesheet():
        """Get the stylesheet for PrimaryButton component."""
        return f"""
            QPushButton#primary_button {{
                background-color: {BitSOTATheme.COLOR1};
                color: {BitSOTATheme.COLOR2_VARIANT};
                border: none;
                border-radius: 4px;
                font-size: 16px;
                font-weight: 600;
                line-height: 1.2;
            }}
            
            QPushButton#primary_button:hover {{
                background-color: rgba(21, 0, 73, 0.9);
            }}
            
            QPushButton#primary_button:pressed {{
                background-color: rgba(21, 0, 73, 0.8);
            }}
            
            QPushButton#primary_button QWidget#icon_container {{
                background-color: transparent;
                border: none;
            }}
            
            QPushButton#primary_button QLabel#button_text_label {{
                background: transparent;
                border: none;
                color: {BitSOTATheme.COLOR2};
            }}
        """
    
    def __init__(
        self, 
        text: str, 
        width: int = 200, 
        height: int = 48, 
        icon_path: str = None,
        icon_size: int = 20,
        icon_rotation: int = 0,
        parent=None
    ):
        """
        Primary button component based on Figma design.
        
        Args:
            text: Button text
            width: Button width (default: 200px)
            height: Button height (default: 48px)
            icon_path: Path to SVG icon file (optional)
            icon_size: Icon size in pixels (default: 20px)
            icon_rotation: Icon rotation in degrees (default: 0, use 180 for flipped icon)
            parent: Parent widget
        """
        super().__init__(parent)
        self.setObjectName("primary_button")
        self.setFixedSize(QSize(width, height))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Apply component stylesheet
        self.setStyleSheet(self.get_stylesheet())
        
        self.icon_path = icon_path
        self.icon_size = icon_size
        self.icon_rotation = icon_rotation
        self.icon_widget = None
        self.icon_container = None
        self.text_label_widget = None

        if icon_path:
            # Create layout with 10px gap (matching Figma)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(4, 4, 4, 4)  # Leave space for border-radius
            layout.setSpacing(10)  # Updated to match Figma design
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Icon container
            self.icon_container = QWidget()
            self.icon_container.setObjectName("icon_container")
            icon_container_layout = QHBoxLayout(self.icon_container)
            icon_container_layout.setContentsMargins(0, 0, 0, 0)
            icon_container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Create icon label with rotated SVG
            self.icon_widget = QLabel()
            self.icon_widget.setFixedSize(icon_size, icon_size)
            self._load_and_rotate_icon(icon_path, icon_size, icon_rotation)
            
            icon_container_layout.addWidget(self.icon_widget)
            layout.addWidget(self.icon_container)

            # Text label
            self.text_label_widget = QLabel(text)
            self.text_label_widget.setObjectName("button_text_label")
            layout.addWidget(self.text_label_widget)
        else:
            self.setText(text)

    def _load_and_rotate_icon(self, icon_path: str, icon_size: int, rotation: int):
        """Load SVG and render it with rotation to a pixmap."""
        # Render SVG to pixmap
        renderer = QSvgRenderer(icon_path)
        pixmap = QPixmap(icon_size, icon_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Apply rotation if needed
        if rotation != 0:
            painter.translate(icon_size / 2, icon_size / 2)
            painter.rotate(rotation)
            painter.translate(-icon_size / 2, -icon_size / 2)
        
        renderer.render(painter)
        painter.end()
        
        self.icon_widget.setPixmap(pixmap)

    def update_icon(self, new_icon_path: str):
        """Update the icon image."""
        if self.icon_widget and self.icon_path:
            self.icon_path = new_icon_path
            self._load_and_rotate_icon(new_icon_path, self.icon_size, self.icon_rotation)

    def update_text(self, new_text: str):
        """Update the button text."""
        if self.text_label_widget:
            self.text_label_widget.setText(new_text)
        else:
            self.setText(new_text)
    
    def set_icon_rotation(self, rotation: int):
        """Set the icon rotation in degrees."""
        if self.icon_widget and self.icon_path:
            self.icon_rotation = rotation
            self._load_and_rotate_icon(self.icon_path, self.icon_size, rotation)


class SecondaryButton(QPushButton):
    """Secondary button component based on Figma design."""
    
    @staticmethod
    def get_stylesheet():
        """Get the stylesheet for SecondaryButton component."""
        return f"""
            QPushButton#secondary_button {{
                background-color: {BitSOTATheme.SECONDARY_BUTTON_BG};
                color: {BitSOTATheme.BLACK100};
                border: none;
                border-radius: 4px;
                font-size: 16px;
                font-weight: 600;
                line-height: 1.2;
                min-height: 48px;
            }}
            
            QPushButton#secondary_button:hover {{
                background-color: #C0BCCB;
            }}
            
            QPushButton#secondary_button:pressed {{
                background-color: #B0ACBB;
            }}
        """
    
    def __init__(self, text: str, width: int = 200, height: int = 48, parent=None):
        super().__init__(text, parent)
        self.setObjectName("secondary_button")
        self.setFixedSize(QSize(width, height))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Apply component stylesheet
        self.setStyleSheet(self.get_stylesheet())

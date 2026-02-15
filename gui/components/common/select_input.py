# PySide6 imports
from PySide6.QtWidgets import QComboBox, QStyledItemDelegate, QStyle
from PySide6.QtCore import Qt, QSize, QRect
from PySide6.QtGui import QColor, QPainter, QPixmap, QFont
from PySide6.QtSvg import QSvgRenderer

# Local application imports
from gui.theme import BitSOTATheme
from gui.resource_path import resource_path


class SelectItemDelegate(QStyledItemDelegate):
    """Custom delegate for dropdown items with checkmark indicator on selected item."""

    def __init__(self, combo_box: QComboBox, parent=None):
        super().__init__(parent)
        self.combo_box = combo_box
        self._check_pixmap = self._load_check_icon()

    def _load_check_icon(self) -> QPixmap:
        """Load checkmark SVG icon as pixmap."""
        renderer = QSvgRenderer(resource_path("gui/images/select_check.svg"))
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        return pixmap

    def paint(self, painter: QPainter, option, index):
        """Paint dropdown item with highlight and checkmark for selected item."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        is_current = index.row() == self.combo_box.currentIndex()
        is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        # Background
        if is_current:
            # Selected item: light purple highlight
            painter.fillRect(option.rect, QColor(109, 96, 142, 41))
        elif is_hovered:
            # Hovered item: subtle tint
            painter.fillRect(option.rect, QColor(21, 0, 73, 10))
        else:
            # Normal item: white
            painter.fillRect(option.rect, QColor(255, 255, 255))

        # Text
        text = index.data(Qt.ItemDataRole.DisplayRole)
        text_rect = option.rect.adjusted(14, 0, -40, 0)
        painter.setPen(QColor("#0C0029"))
        font = painter.font()
        font.setPixelSize(14)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            text,
        )

        # Checkmark for selected item
        if is_current:
            icon_x = option.rect.right() - 30
            icon_y = option.rect.center().y() - 8
            painter.drawPixmap(icon_x, icon_y, self._check_pixmap)

        painter.restore()

    def sizeHint(self, option, index):
        """Return size hint for each dropdown item."""
        return QSize(option.rect.width(), 40)


class SelectInput(QComboBox):
    """
    Reusable select/dropdown component based on Figma design.

    Features:
    - Styled input with border, rounded corners, and chevron icon
    - Dropdown popup with checkmark indicator on selected item
    - Hover highlight on dropdown items
    - Self-contained stylesheet (no dependency on theme.py for QComboBox styles)

    Usage:
        select = SelectInput()
        select.addItems(["Option 1", "Option 2", "Option 3"])
    """

    @staticmethod
    def get_stylesheet() -> str:
        """Get the stylesheet for SelectInput component."""
        return f"""
            QComboBox#select_input {{
                background-color: {BitSOTATheme.CONTENT_BOX_BG};
                color: {BitSOTATheme.BLACK100};
                border: 1px solid {BitSOTATheme.BORDER_12};
                border-radius: 4px;
                padding: 8px 14px;
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 500;
            }}

            QComboBox#select_input::drop-down {{
                border: none;
                width: 30px;
            }}

            QComboBox#select_input::down-arrow {{
                image: url({resource_path("gui/images/chevron-down-2.svg")});
                width: 16px;
                height: 16px;
            }}

            QComboBox#select_input QAbstractItemView {{
                background-color: {BitSOTATheme.CONTENT_BOX_BG};
                border: 1px solid {BitSOTATheme.BORDER_12};
                border-radius: 4px;
                padding: 4px 0px;
                outline: none;
            }}

            QComboBox#select_input QAbstractItemView::item {{
                min-height: 40px;
                padding: 0px;
            }}
        """

    def __init__(self, height: int = 48, parent=None):
        """
        Create a styled select input.

        Args:
            height: Fixed height of the input (default: 48px)
            parent: Parent widget
        """
        super().__init__(parent)
        self.setObjectName("select_input")
        self.setFixedHeight(height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self.get_stylesheet())

        # Custom delegate for styled dropdown items with checkmark
        delegate = SelectItemDelegate(self, self)
        self.setItemDelegate(delegate)

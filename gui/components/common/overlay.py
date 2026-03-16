from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor


def show_modal_with_overlay(dialog, widget=None):
    """
    Show dialog with overlay
    
    .. deprecated::
        This function is deprecated and will be removed in a future version.
        Use modal_manager methods instead:
        - main_window.modal_manager.show_error(title, message)
        - main_window.modal_manager.show_coming_soon(title, message)
        - main_window.modal_manager.show_wallet_selection()
        - main_window.modal_manager.show_invite_code(relay_url, wallet, coldkey_address)
        - main_window.modal_manager.show_terms_acceptance()
        - main_window.modal_manager.show_wallet_import_success()
    
    Args:
        dialog: The dialog to display
        widget: The widget calling the dialog (used to find the main window)
    
    Returns:
        Dialog result (QDialog.Accepted or QDialog.Rejected)
    """
    import warnings
    warnings.warn(
        "show_modal_with_overlay() is deprecated. Use modal_manager methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Find main window
    main_window = None
    if widget:
        current = widget
        while current:
            parent = current.parent()
            if parent is None:
                # Found top-level window
                from PySide6.QtWidgets import QMainWindow
                if isinstance(current, QMainWindow):
                    main_window = current
                break
            current = parent
    
    # If main window found and has show_modal_with_overlay method, use it
    if main_window and hasattr(main_window, 'show_modal_with_overlay'):
        return main_window.show_modal_with_overlay(dialog)
    else:
        # Otherwise show dialog directly (without overlay)
        return dialog.exec()


class ModalOverlay(QWidget):
    """Semi-transparent overlay for modal dialog background"""
    
    clicked = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Ensure overlay displays within parent window, no need for WindowStaysOnTopHint
        self.setAutoFillBackground(False)
        self.hide()
    
    def showEvent(self, event):
        """Raise to top when shown"""
        super().showEvent(event)
        self.raise_()
    
    def mousePressEvent(self, event):
        """Emit signal when overlay is clicked (can be used to close dialog)"""
        # Don't propagate mouse events to prevent click-through
        self.clicked.emit()
        event.accept()
    
    def paintEvent(self, event):
        """Custom paint semi-transparent background"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Fill with semi-transparent dark gray - #181C1F with 60% opacity
        # #181C1F = RGB(24, 28, 31)
        # opacity: 0.6 = alpha: 153 (255 * 0.6)
        painter.fillRect(self.rect(), QColor(24, 28, 31, 153))

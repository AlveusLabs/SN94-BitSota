from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QWidget, QCheckBox
)
from PySide6.QtSvgWidgets import QSvgWidget

from gui.components.common.overlay import show_modal_with_overlay
from gui.components.common.button import PrimaryButton, SecondaryButton
from gui.wallet_utils_gui import discover_wallets, get_coldkey_address_from_wallet
from gui.resource_path import resource_path
from gui.theme import BitSOTATheme


class WalletListItem(QPushButton):
    """Wallet list item component.
    
    Button styles are defined in theme.py (QPushButton#wallet_list_item / 
    QPushButton#wallet_list_item_selected). Child QLabel colors are set
    via inline styles in set_selected() to avoid stylesheet cascading issues.
    """
    
    def __init__(self, wallet_name: str, hotkey_name: str, source: str = "bitsota", parent=None):
        super().__init__(parent)
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.source = source
        self.is_selected = False
        
        self.setObjectName("wallet_list_item")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setup_ui(wallet_name, hotkey_name)

    def setup_ui(self, wallet_name: str, hotkey_name: str):
        """Setup the wallet item UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Checkmark
        self.checkmark = QLabel("")
        self.checkmark.setFixedWidth(20)
        layout.addWidget(self.checkmark)

        # Wallet info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        self.wallet_label = QLabel(f"{wallet_name}...{hotkey_name[-4:]}")
        self.wallet_label.setStyleSheet("background: transparent; border: none;")
        info_layout.addWidget(self.wallet_label)

        layout.addLayout(info_layout)
        layout.addStretch()

    def set_selected(self, selected: bool):
        """Set the selected state of the wallet item."""
        self.is_selected = selected
        
        if selected:
            self.setObjectName("wallet_list_item_selected")
            self.checkmark.setText("✓")
            self.wallet_label.setStyleSheet(
                f"background: transparent; border: none; color: {BitSOTATheme.COLOR2}; font-weight: 500;"
            )
            self.checkmark.setStyleSheet(
                f"background: transparent; border: none; color: {BitSOTATheme.COLOR2}; font-weight: 500;"
            )
        else:
            self.setObjectName("wallet_list_item")
            self.checkmark.setText("")
            self.wallet_label.setStyleSheet(
                f"background: transparent; border: none; color: {BitSOTATheme.COLOR1}; font-weight: 500;"
            )
            self.checkmark.setStyleSheet(
                f"background: transparent; border: none; color: {BitSOTATheme.COLOR1}; font-weight: 500;"
            )
        
        # Refresh button styles
        self.style().unpolish(self)
        self.style().polish(self)


class WalletSelectionModal(QDialog):
    """Wallet selection modal dialog component."""
    
    # Constants
    MODAL_WIDTH = 800
    MODAL_HEIGHT = 650
    MODAL_PADDING = 32
    BUTTON_SPACING = 16
    
    wallet_selected = Signal(str, str, bool, str)
    
    @staticmethod
    def get_stylesheet():
        """Get the stylesheet for WalletSelectionModal component."""
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
                color: {BitSOTATheme.COLOR1_60};
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 16px;
                font-weight: 400;
            }}
            
            QScrollArea {{
                border: none;
                background-color: {BitSOTATheme.CONTENT_BOX_BG};
            }}
            
            QWidget#wallet_scroll_content {{
                background-color: {BitSOTATheme.CONTENT_BOX_BG};
            }}
            
            QCheckBox {{
                color: {BitSOTATheme.COLOR1};
                font-family: "PingFang SC", "Microsoft YaHei", "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
            }}
            
            QCheckBox:disabled {{
                color: {BitSOTATheme.COLOR1_20};
            }}
        """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(self.MODAL_WIDTH, self.MODAL_HEIGHT)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(self.get_stylesheet())
        
        # State
        self.selected_item = None
        self.wallet_items = []
        self.selected_wallet_source = None
        self.current_coldkey_address = None
        
        self.setup_ui()
        self.load_wallets()

    def setup_ui(self):
        """Setup the main UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.MODAL_PADDING, 24, self.MODAL_PADDING, 24)
        layout.setSpacing(24)

        # Header
        layout.addLayout(self._create_header())
        
        # Wallet list scroll area
        layout.addWidget(self._create_wallet_list_area(), 1)
        
        # Coldkey checkbox
        layout.addWidget(self._create_coldkey_checkbox())
        
        # Action buttons
        layout.addLayout(self._create_action_buttons())

    def _create_header(self):
        """Create the modal header with title, count, and close button."""
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title_icon = QSvgWidget(resource_path("gui/images/Wallet.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)

        title_label = QLabel("My Wallets")
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)

        self.wallet_count_label = QLabel("[0]")
        self.wallet_count_label.setObjectName("modal_message")
        header_layout.addWidget(self.wallet_count_label)

        header_layout.addStretch()

        close_btn = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn)

        return header_layout

    def _create_wallet_list_area(self):
        """Create the scrollable wallet list area."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_content = QWidget()
        scroll_content.setObjectName("wallet_scroll_content")
        scroll_content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        self.wallet_list_layout = QVBoxLayout(scroll_content)
        self.wallet_list_layout.setContentsMargins(0, 0, 0, 0)
        self.wallet_list_layout.setSpacing(12)
        self.wallet_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll_area.setWidget(scroll_content)
        return scroll_area

    def _create_coldkey_checkbox(self):
        """Create the coldkey address checkbox."""
        checkbox = QCheckBox("Use coldkey address already associated with this wallet for rewards")
        checkbox.setEnabled(False)
        checkbox.stateChanged.connect(self._on_checkbox_state_changed)
        self.use_coldkey_checkbox = checkbox
        return checkbox

    def _create_action_buttons(self):
        """Create the Select and Cancel buttons."""
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(self.BUTTON_SPACING)

        # Calculate button width: (modal_width - padding*2 - spacing) / 2
        button_width = (self.MODAL_WIDTH - self.MODAL_PADDING * 2 - self.BUTTON_SPACING) // 2

        select_btn = PrimaryButton("Select", width=button_width)
        select_btn.clicked.connect(self._on_select)
        buttons_layout.addWidget(select_btn)

        cancel_btn = SecondaryButton("Cancel", width=button_width)
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn)

        return buttons_layout

    def load_wallets(self):
        """Load and display available wallets."""
        # Clear existing items
        self._clear_wallet_list()
        
        wallets = discover_wallets()

        if not wallets:
            self._show_no_wallets_message()
            return

        # Add wallet items
        wallet_count = 0
        for wallet_name, hotkeys, source in wallets:
            for hotkey_name in hotkeys:
                wallet_count += 1
                item = self._create_wallet_item(wallet_name, hotkey_name, source)
                self.wallet_list_layout.addWidget(item)
                self.wallet_items.append(item)

        self.wallet_count_label.setText(f"[{wallet_count}]")

        # Auto-select first wallet
        if self.wallet_items:
            self._on_wallet_item_clicked(self.wallet_items[0])

    def _clear_wallet_list(self):
        """Clear all items from the wallet list."""
        for i in reversed(range(self.wallet_list_layout.count())):
            widget = self.wallet_list_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.wallet_items.clear()

    def _show_no_wallets_message(self):
        """Show a message when no wallets are found."""
        no_wallets_label = QLabel("No wallets found. Please create or import a wallet.")
        no_wallets_label.setObjectName("modal_message")
        no_wallets_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wallet_list_layout.addWidget(no_wallets_label)
        self.wallet_count_label.setText("[0]")

    def _create_wallet_item(self, wallet_name: str, hotkey_name: str, source: str):
        """Create a wallet list item.
        
        Args:
            wallet_name: Name of the wallet
            hotkey_name: Name of the hotkey
            source: Source of the wallet
        """
        item = WalletListItem(
            wallet_name=wallet_name,
            hotkey_name=hotkey_name,
            source=source
        )
        item.clicked.connect(lambda checked=False, w=item: self._on_wallet_item_clicked(w))
        return item

    def _on_wallet_item_clicked(self, item: WalletListItem):
        """Handle wallet item selection.
        
        Args:
            item: The wallet item that was clicked
        """
        # Deselect previous item
        if self.selected_item:
            self.selected_item.set_selected(False)

        # Select new item
        item.set_selected(True)
        self.selected_item = item
        self.selected_wallet_source = item.source

        # Get coldkey address for selected wallet
        self.current_coldkey_address = get_coldkey_address_from_wallet(
            item.wallet_name, 
            item.source
        )

        # Update checkbox state
        self._update_coldkey_checkbox()

    def _update_coldkey_checkbox(self):
        """Update the coldkey checkbox based on current coldkey address."""
        if self.current_coldkey_address:
            self.use_coldkey_checkbox.setEnabled(True)
            self.use_coldkey_checkbox.setStyleSheet("")
        else:
            self.use_coldkey_checkbox.setEnabled(False)
            self.use_coldkey_checkbox.setChecked(False)
            self.use_coldkey_checkbox.setStyleSheet(f"color: {BitSOTATheme.COLOR1_20};")

    def _on_checkbox_state_changed(self, state):
        """Handle coldkey checkbox state change.
        
        Args:
            state: The new checkbox state
        """
        if state == Qt.CheckState.Checked.value and not self.current_coldkey_address:
            self._show_no_coldkey_error()
            self.use_coldkey_checkbox.setChecked(False)

    def _show_no_coldkey_error(self):
        """Show error when trying to use coldkey but none exists."""
        from gui.components.modals.import_confirmation import ErrorModal
        error_modal = ErrorModal(
            "No Coldkey Found",
            "This wallet does not have an associated coldkey address. "
            "Please provide a coldkey address on the next screen.",
            parent=self
        )
        show_modal_with_overlay(error_modal, self)

    def _on_select(self):
        """Handle Select button click."""
        if not self.selected_item:
            return
        
        use_existing_coldkey = self.use_coldkey_checkbox.isChecked()
        coldkey_address = self.current_coldkey_address if use_existing_coldkey else ""
        
        self.wallet_selected.emit(
            self.selected_item.wallet_name,
            self.selected_item.hotkey_name,
            use_existing_coldkey,
            coldkey_address
        )
        self.accept()

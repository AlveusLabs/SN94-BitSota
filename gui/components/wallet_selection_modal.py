from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QWidget, QCheckBox
)
from PySide6.QtSvgWidgets import QSvgWidget
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from gui.wallet_utils_gui import discover_wallets
from gui.resource_path import resource_path


class WalletListItem(QPushButton):
    def __init__(self, wallet_name: str, hotkey_name: str, source: str = "bitsota", parent=None):
        super().__init__(parent)
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.source = source
        self.setObjectName("wallet_list_item")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.is_selected = False
        self.setup_ui(wallet_name, hotkey_name)

    def setup_ui(self, wallet_name: str, hotkey_name: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.checkmark = QLabel("")
        self.checkmark.setFixedWidth(20)
        layout.addWidget(self.checkmark)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        self.wallet_label = QLabel(f"{wallet_name}...{hotkey_name[-4:]}")
        self.wallet_label.setStyleSheet("background: transparent; border: none;")
        info_layout.addWidget(self.wallet_label)

        layout.addLayout(info_layout)
        layout.addStretch()

    def set_selected(self, selected: bool):
        self.is_selected = selected
        if selected:
            self.setObjectName("wallet_list_item_selected")
            self.checkmark.setText("✓")
            self.wallet_label.setStyleSheet("background: transparent; border: none; color: #8EFBFF; font-weight: 500;")
        else:
            self.setObjectName("wallet_list_item")
            self.checkmark.setText("")
            self.wallet_label.setStyleSheet("background: transparent; border: none; color: #150049; font-weight: 500;")
        self.style().unpolish(self)
        self.style().polish(self)


class WalletSelectionModal(QDialog):
    wallet_selected = Signal(str, str, bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(800, 650)
        self.selected_item = None
        self.wallet_items = []
        self.selected_wallet_source = None
        self.current_coldkey_address = None
        self.setup_ui()
        self.load_wallets()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(24)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title_icon = QSvgWidget(resource_path("gui/images/Wallet.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)

        title_label = QLabel("My Wallets")
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)

        wallet_count_label = QLabel("[0]")
        wallet_count_label.setObjectName("modal_message")
        header_layout.addWidget(wallet_count_label)
        self.wallet_count_label = wallet_count_label

        header_layout.addStretch()

        close_btn_widget = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn_widget.setFixedSize(24, 24)
        close_btn_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn_widget.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn_widget)

        layout.addLayout(header_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #FFFFFF; }")

        scroll_content = QWidget()
        scroll_content.setObjectName("wallet_scroll_content")
        scroll_content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll_content.setStyleSheet("QWidget#wallet_scroll_content { background-color: #FFFFFF; }")
        self.wallet_list_layout = QVBoxLayout(scroll_content)
        self.wallet_list_layout.setContentsMargins(0, 0, 0, 0)
        self.wallet_list_layout.setSpacing(12)
        self.wallet_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)

        self.use_coldkey_checkbox = QCheckBox("Use coldkey address already associated with this wallet for rewards")
        self.use_coldkey_checkbox.setEnabled(False)
        self.use_coldkey_checkbox.stateChanged.connect(self._on_checkbox_state_changed)
        layout.addWidget(self.use_coldkey_checkbox)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(16)

        select_btn = QPushButton("Select")
        select_btn.setObjectName("primary_button")
        select_btn.setFixedHeight(48)
        select_btn.clicked.connect(self._on_select)
        buttons_layout.addWidget(select_btn, 1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondary_button")
        cancel_btn.setFixedHeight(48)
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_btn, 1)

        layout.addLayout(buttons_layout)

    def load_wallets(self):
        for i in reversed(range(self.wallet_list_layout.count())):
            widget = self.wallet_list_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self.wallet_items.clear()

        wallets = discover_wallets()

        if not wallets:
            no_wallets_label = QLabel("No wallets found. Please create or import a wallet.")
            no_wallets_label.setObjectName("modal_message")
            no_wallets_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.wallet_list_layout.addWidget(no_wallets_label)
            self.wallet_count_label.setText("[0]")
            return

        wallet_count = 0
        for wallet_name, hotkeys, source in wallets:
            for hotkey_name in hotkeys:
                wallet_count += 1
                item = WalletListItem(
                    wallet_name=wallet_name,
                    hotkey_name=hotkey_name,
                    source=source
                )
                item.clicked.connect(lambda checked=False, w=item: self._on_wallet_item_clicked(w))
                self.wallet_list_layout.addWidget(item)
                self.wallet_items.append(item)

        self.wallet_count_label.setText(f"[{wallet_count}]")

        if self.wallet_items:
            self._on_wallet_item_clicked(self.wallet_items[0])

    def _on_wallet_item_clicked(self, item: WalletListItem):
        if self.selected_item:
            self.selected_item.set_selected(False)

        item.set_selected(True)
        self.selected_item = item
        self.selected_wallet_source = item.source

        from gui.wallet_utils_gui import get_coldkey_address_from_wallet
        self.current_coldkey_address = get_coldkey_address_from_wallet(item.wallet_name, item.source)

        if self.current_coldkey_address:
            self.use_coldkey_checkbox.setEnabled(True)
            self.use_coldkey_checkbox.setStyleSheet("")
        else:
            self.use_coldkey_checkbox.setEnabled(False)
            self.use_coldkey_checkbox.setChecked(False)
            self.use_coldkey_checkbox.setStyleSheet("color: rgba(21, 0, 73, 0.3);")

    def _on_checkbox_state_changed(self, state):
        if state == Qt.CheckState.Checked.value and not self.current_coldkey_address:
            from gui.components.import_confirmation_modals import ErrorModal
            error_modal = ErrorModal(
                "No Coldkey Found",
                "This wallet does not have an associated coldkey address. Please provide a coldkey address on the next screen.",
                parent=self
            )
            error_modal.exec()
            self.use_coldkey_checkbox.setChecked(False)

    def _on_select(self):
        if self.selected_item:
            use_existing_coldkey = self.use_coldkey_checkbox.isChecked()
            coldkey_address = self.current_coldkey_address if use_existing_coldkey else ""
            self.wallet_selected.emit(
                self.selected_item.wallet_name,
                self.selected_item.hotkey_name,
                use_existing_coldkey,
                coldkey_address
            )
            self.accept()

from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QGridLayout, QStackedWidget
)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import QKeyEvent, QEnterEvent
from gui.resource_path import resource_path


class WalletOptionContainer(QWidget):
    clicked = Signal()

    def __init__(
        self,
        title: str,
        description: str,
        button_text: str,
        icon_path: str = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("wallet_option_container")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setup_ui(title, description, button_text, icon_path)

    def setup_ui(self, title: str, description: str, button_text: str, icon_path: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 32, 24, 32)
        layout.setSpacing(24)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        if icon_path:
            header_layout = QHBoxLayout()
            header_layout.setSpacing(12)

            icon_widget = QSvgWidget(icon_path)
            icon_widget.setFixedSize(24, 24)
            header_layout.addWidget(icon_widget)

            title_label = QLabel(title)
            title_label.setObjectName("wallet_option_title")
            header_layout.addWidget(title_label)

            layout.addLayout(header_layout)
        else:
            title_label = QLabel(title)
            title_label.setObjectName("wallet_option_title")
            layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setObjectName("wallet_option_desc")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        self.button = QPushButton(button_text)
        self.button.setObjectName("primary_button")
        self.button.setFixedSize(207, 48)
        self.button.clicked.connect(self.clicked.emit)

        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignLeft)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class ImportHotkeyScreen(QWidget):
    imported = Signal(str, str, str)
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mnemonic_boxes = []
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        content_box = QWidget()
        content_box.setObjectName("content_box")
        content_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(content_box)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)

        title = QLabel("Hotkey Credentials")
        title.setObjectName("hotkey_credentials_title")
        layout.addWidget(title)

        hotkey_name_label = QLabel("Hotkey name")
        hotkey_name_label.setObjectName("form_label")
        layout.addWidget(hotkey_name_label)

        self.hotkey_name_input = QLineEdit()
        self.hotkey_name_input.setObjectName("form_input")
        self.hotkey_name_input.setPlaceholderText("Enter hotkey name")
        layout.addWidget(self.hotkey_name_input)

        mnemonic_label = QLabel("Mnemonic")
        mnemonic_label.setObjectName("form_label")
        layout.addWidget(mnemonic_label)

        mnemonic_grid = QGridLayout()
        mnemonic_grid.setSpacing(12)

        for i in range(12):
            word_box = QLineEdit()
            word_box.setObjectName("mnemonic_word_box")
            word_box.setPlaceholderText(f"{i+1}.----")
            word_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
            word_box.setEchoMode(QLineEdit.EchoMode.Password)
            word_box.installEventFilter(self)
            self.mnemonic_boxes.append(word_box)

            row = i // 4
            col = i % 4
            mnemonic_grid.addWidget(word_box, row, col)

        layout.addLayout(mnemonic_grid)

        coldkey_label = QLabel("Coldkey Address (for rewards)")
        coldkey_label.setObjectName("form_label")
        layout.addWidget(coldkey_label)

        self.coldkey_input = QLineEdit()
        self.coldkey_input.setObjectName("form_input")
        self.coldkey_input.setPlaceholderText("Enter your coldkey address (starts with 5)")
        layout.addWidget(self.coldkey_input)

        layout.addStretch()

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(16)

        import_btn = QPushButton("Import")
        import_btn.setObjectName("primary_button")
        import_btn.setFixedSize(207, 48)
        import_btn.clicked.connect(self._on_import)
        buttons_layout.addWidget(import_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("secondary_button")
        cancel_btn.setFixedSize(207, 48)
        cancel_btn.clicked.connect(self.cancelled.emit)
        buttons_layout.addWidget(cancel_btn)

        buttons_layout.addStretch()

        layout.addLayout(buttons_layout)

        main_layout.addWidget(content_box)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key_event = QKeyEvent(event)
            if key_event.key() == Qt.Key.Key_Space:
                if obj in self.mnemonic_boxes:
                    current_index = self.mnemonic_boxes.index(obj)
                    if current_index < len(self.mnemonic_boxes) - 1:
                        self.mnemonic_boxes[current_index + 1].setFocus()
                        return True
        elif event.type() == QEvent.Type.Enter:
            if obj in self.mnemonic_boxes:
                obj.setEchoMode(QLineEdit.EchoMode.Normal)
        elif event.type() == QEvent.Type.Leave:
            if obj in self.mnemonic_boxes:
                obj.setEchoMode(QLineEdit.EchoMode.Password)
        return super().eventFilter(obj, event)

    def _on_import(self):
        hotkey_name = self.hotkey_name_input.text().strip()
        mnemonic_words = [box.text().strip() for box in self.mnemonic_boxes]
        mnemonic = " ".join(mnemonic_words)
        coldkey_address = self.coldkey_input.text().strip()

        from gui.components.import_confirmation_modals import ErrorModal, TermsAcceptanceModal
        from gui.wallet_utils_gui import validate_coldkey_address

        if not hotkey_name:
            error_modal = ErrorModal(
                "Invalid Hotkey Name",
                "Please enter a hotkey name.",
                parent=self
            )
            error_modal.exec()
            return

        if not all(mnemonic_words):
            error_modal = ErrorModal(
                "Incomplete Mnemonic",
                "Please fill in all 12 mnemonic words.",
                parent=self
            )
            error_modal.exec()
            return

        if len(mnemonic_words) != 12:
            error_modal = ErrorModal(
                "Invalid Mnemonic",
                f"Expected 12 words, but got {len(mnemonic_words)}.",
                parent=self
            )
            error_modal.exec()
            return

        if coldkey_address:
            is_valid, error_message = validate_coldkey_address(coldkey_address)
            if not is_valid:
                error_modal = ErrorModal(
                    "Invalid Coldkey Address",
                    error_message,
                    parent=self
                )
                error_modal.exec()
                return

        try:
            from substrateinterface import Keypair
            Keypair.create_from_mnemonic(mnemonic)
        except Exception as e:
            error_modal = ErrorModal(
                "Invalid Mnemonic",
                f"The mnemonic phrase is invalid. Please check your words and try again.\n\nError: {str(e)}",
                parent=self
            )
            error_modal.exec()
            return

        terms_modal = TermsAcceptanceModal(parent=self)
        terms_modal.confirmed.connect(lambda: self.imported.emit(hotkey_name, mnemonic, coldkey_address))
        terms_modal.exec()

    def clear_form(self):
        self.hotkey_name_input.clear()
        self.coldkey_input.clear()
        for box in self.mnemonic_boxes:
            box.clear()


class WalletScreen(QWidget):
    wallet_loaded = Signal(str, str, bool, str)
    hotkey_imported = Signal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.stacked_widget = QStackedWidget()

        self.initial_screen = self._create_initial_screen()
        self.import_screen = ImportHotkeyScreen()
        self.import_screen.imported.connect(self._on_hotkey_imported)
        self.import_screen.cancelled.connect(self._on_import_cancelled)

        self.stacked_widget.addWidget(self.initial_screen)
        self.stacked_widget.addWidget(self.import_screen)

        main_layout.addWidget(self.stacked_widget)

    def _create_initial_screen(self):
        screen = QWidget()
        layout = QVBoxLayout(screen)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(24)

        layout.addStretch(1)

        containers_layout = QHBoxLayout()
        containers_layout.setSpacing(24)

        load_container = WalletOptionContainer(
            title="Load Wallet",
            description="Load a hotkey from computer and use that for mining",
            button_text="Load",
            icon_path=resource_path("gui/images/wallet_add.svg"),
        )
        load_container.clicked.connect(self._on_load_wallet)
        containers_layout.addWidget(load_container, 1)

        import_container = WalletOptionContainer(
            title="Import Wallet",
            description="Import a mining wallet by inputting your hotkey mnemonic",
            button_text="Import",
            icon_path=resource_path("gui/images/wallet_add.svg"),
        )
        import_container.clicked.connect(self._on_import_wallet)
        containers_layout.addWidget(import_container, 1)

        layout.addLayout(containers_layout)
        layout.addStretch(2)

        return screen

    def _on_load_wallet(self):
        from gui.components.wallet_selection_modal import WalletSelectionModal

        modal = WalletSelectionModal(parent=self)
        modal.wallet_selected.connect(self._on_wallet_selected)
        modal.exec()

    def _on_wallet_selected(self, wallet_name: str, hotkey_name: str, use_existing_coldkey: bool, coldkey_address: str):
        self.wallet_loaded.emit(wallet_name, hotkey_name, use_existing_coldkey, coldkey_address)

    def _on_import_wallet(self):
        self.stacked_widget.setCurrentWidget(self.import_screen)

    def _on_hotkey_imported(self, hotkey_name: str, mnemonic: str, coldkey_address: str):
        from gui.components.import_confirmation_modals import WalletImportedSuccessModal

        success_modal = WalletImportedSuccessModal(parent=self)
        success_modal.start_mining.connect(lambda: self._finalize_import(hotkey_name, mnemonic, coldkey_address))
        success_modal.rejected.connect(lambda: self._finalize_import(hotkey_name, mnemonic, coldkey_address))
        success_modal.exec()

    def _finalize_import(self, hotkey_name: str, mnemonic: str, coldkey_address: str):
        self.hotkey_imported.emit(hotkey_name, mnemonic, coldkey_address)
        self.import_screen.clear_form()
        self.stacked_widget.setCurrentWidget(self.initial_screen)

    def _on_import_cancelled(self):
        self.import_screen.clear_form()
        self.stacked_widget.setCurrentWidget(self.initial_screen)

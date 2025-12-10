from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QWidget
)
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path


class AccordionItem(QWidget):
    def __init__(self, title: str, content_items: list, parent=None):
        super().__init__(parent)
        self.setStyleSheet("AccordionItem { background-color: transparent; }")
        self.is_expanded = False
        self.content_items = content_items
        self.setup_ui(title)

    def setup_ui(self, title: str):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.header = QWidget()
        self.header.setStyleSheet("QWidget { background-color: transparent; }")
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(16, 16, 16, 16)
        header_layout.setSpacing(12)

        rectangle_icon = QSvgWidget(resource_path("gui/images/rectangle.svg"))
        rectangle_icon.setFixedSize(8, 8)
        header_layout.addWidget(rectangle_icon)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("accordion_title")
        self.title_label.setStyleSheet("""
            background: transparent;
            color: #150049;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            font-weight: 500;
        """)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.arrow_icon = QSvgWidget(resource_path("gui/images/arrowt01.svg"))
        self.arrow_icon.setFixedSize(16, 16)
        header_layout.addWidget(self.arrow_icon)

        self.header.mousePressEvent = lambda e: self.toggle()
        self.main_layout.addWidget(self.header)

        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("QWidget { background-color: transparent; }")
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(16, 0, 16, 16)
        content_layout.setSpacing(8)

        for item in self.content_items:
            if isinstance(item, dict):
                item_label = QLabel(f"• {item['title']}")
                item_label.setWordWrap(True)
                item_label.setStyleSheet("""
                    background-color: transparent;
                    color: #150049;
                    font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                    font-size: 14px;
                    font-weight: 500;
                    line-height: 150%;
                """)
                content_layout.addWidget(item_label)

                desc_label = QLabel(item['description'])
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("""
                    background-color: transparent;
                    color: rgba(21, 0, 73, 0.60);
                    font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                    font-size: 14px;
                    font-weight: 400;
                    line-height: 150%;
                    margin-left: 16px;
                """)
                content_layout.addWidget(desc_label)
            else:
                item_label = QLabel(item)
                item_label.setWordWrap(True)
                item_label.setStyleSheet("""
                    background-color: transparent;
                    color: #150049;
                    font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                    font-size: 14px;
                    font-weight: 400;
                    line-height: 150%;
                """)
                content_layout.addWidget(item_label)

        self.content_widget.hide()
        self.main_layout.addWidget(self.content_widget)

    def toggle(self):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.content_widget.show()
        else:
            self.content_widget.hide()


class UserGuideModal(QDialog):
    proceed_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modal_dialog")
        self.setModal(True)
        self.setFixedSize(560, 675)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(24)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)

        title_label = QLabel("User Guide")
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        close_btn_widget = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn_widget.setFixedSize(24, 24)
        close_btn_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn_widget.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn_widget)

        layout.addLayout(header_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        scroll_content = QWidget()
        scroll_content.setStyleSheet("QWidget { background-color: #FFFFFF; }")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)

        getting_started_label = QLabel("Getting Started")
        getting_started_label.setStyleSheet("""
            background-color: transparent;
            color: #150049;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 8px;
        """)
        scroll_layout.addWidget(getting_started_label)

        content_container = QWidget()
        content_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content_container.setStyleSheet("""
            background-color: rgba(21, 0, 73, 0.04);
            border-radius: 8px;
        """)
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        bitsota_item = AccordionItem(
            "What is BitSota",
            ["Platform to let one easily participate in AutoML experiments. You can mine on any machine regardless of compute limitation"]
        )
        content_layout.addWidget(bitsota_item)

        scroll_layout.addWidget(content_container)

        scroll_layout.addSpacing(16)

        mining_label = QLabel("Mining")
        mining_label.setStyleSheet("""
            background-color: transparent;
            color: #150049;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 8px;
        """)
        scroll_layout.addWidget(mining_label)

        mining_container = QWidget()
        mining_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        mining_container.setStyleSheet("""
            background-color: rgba(21, 0, 73, 0.04);
            border-radius: 8px;
        """)
        mining_layout = QVBoxLayout(mining_container)
        mining_layout.setContentsMargins(0, 0, 0, 0)
        mining_layout.setSpacing(0)

        mining_modes_item = AccordionItem(
            "Understanding mining modes",
            [
                {
                    "title": "Direct Mining",
                    "description": "Connect directly to validators in the Bittensor network. Best for experienced users who want full control over their mining operations. Results are sent to a relay server from which validators can retrieve them"
                },
                {
                    "title": "Pool Mining",
                    "description": "Join a mining pool for simplified setup and shared resources. Ideal for beginners. Tasks are retrieved from a centralised pool, mining operation runs on the user's machine and results are sent to the pool server which also handles rewards for miners"
                }
            ]
        )
        mining_layout.addWidget(mining_modes_item)

        scroll_layout.addWidget(mining_container)

        scroll_layout.addSpacing(16)

        wallet_setup_label = QLabel("Wallet Setup")
        wallet_setup_label.setStyleSheet("""
            background-color: transparent;
            color: #150049;
            font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 8px;
        """)
        scroll_layout.addWidget(wallet_setup_label)

        wallet_container = QWidget()
        wallet_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        wallet_container.setStyleSheet("""
            background-color: rgba(21, 0, 73, 0.04);
            border-radius: 8px;
        """)
        wallet_layout = QVBoxLayout(wallet_container)
        wallet_layout.setContentsMargins(0, 0, 0, 0)
        wallet_layout.setSpacing(0)

        load_wallet_item = AccordionItem(
            "Load existing Bittensor wallet from laptop",
            ["People with wallets already on their machine can load their hotkeys to mine the subnet"]
        )
        wallet_layout.addWidget(load_wallet_item)

        separator2 = QWidget()
        separator2.setFixedHeight(1)
        separator2.setStyleSheet("background-color: rgba(21, 0, 73, 0.12);")
        wallet_layout.addWidget(separator2)

        import_hotkey_item = AccordionItem(
            "Import hotkey",
            ["Those with hotkeys but not in the folder we'd load from can enter the hotkey's secret phrase and import it"]
        )
        wallet_layout.addWidget(import_hotkey_item)

        separator3 = QWidget()
        separator3.setFixedHeight(1)
        separator3.setStyleSheet("background-color: rgba(21, 0, 73, 0.12);")
        wallet_layout.addWidget(separator3)

        create_hotkey_content = QWidget()
        create_hotkey_content.setStyleSheet("QWidget { background-color: transparent; }")
        create_hotkey_layout = QVBoxLayout(create_hotkey_content)
        create_hotkey_layout.setContentsMargins(16, 16, 16, 16)
        create_hotkey_layout.setSpacing(12)

        create_steps = [
            "Create your TAO wallet",
            "Secure your mnemonic and hotkey",
            "Connect your wallet"
        ]

        for i, step in enumerate(create_steps, 1):
            step_widget = QWidget()
            step_widget.setStyleSheet("QWidget { background-color: transparent; }")
            step_layout = QHBoxLayout(step_widget)
            step_layout.setContentsMargins(0, 0, 0, 0)
            step_layout.setSpacing(12)

            number_label = QLabel(str(i))
            number_label.setFixedSize(24, 24)
            number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            number_label.setStyleSheet("""
                background-color: rgba(109, 96, 142, 0.16);
                color: #150049;
                border-radius: 4px;
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 500;
            """)
            step_layout.addWidget(number_label)

            step_label = QLabel(step)
            step_label.setStyleSheet("""
                background-color: transparent;
                color: #150049;
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
            """)
            if i == 1:
                step_label.setText('<a href="https://docs.learnbittensor.org/keys/working-with-keys?create-wallet=cold-hot#creating-a-wallet-with-btcli" style="color: #0F6FFF; text-decoration: underline;">Create your TAO wallet</a>')
                step_label.setOpenExternalLinks(True)
                step_label.setStyleSheet("""
                    background-color: transparent;
                    font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                    font-size: 14px;
                    font-weight: 400;
                """)
            step_layout.addWidget(step_label)
            step_layout.addStretch()

            create_hotkey_layout.addWidget(step_widget)

        create_hotkey_item = AccordionItem(
            "How to create a hotkey",
            []
        )
        create_hotkey_item.content_widget.layout().addWidget(create_hotkey_content)
        wallet_layout.addWidget(create_hotkey_item)

        separator4 = QWidget()
        separator4.setFixedHeight(1)
        separator4.setStyleSheet("background-color: rgba(21, 0, 73, 0.12);")
        wallet_layout.addWidget(separator4)

        coldkey_address_item = AccordionItem(
            "Providing coldkey address",
            ["We need you to provide a coldkey address where we pay out to you your earnings/rewards as paying cannot be done to hotkeys"]
        )
        wallet_layout.addWidget(coldkey_address_item)

        scroll_layout.addWidget(wallet_container)
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)

        proceed_btn = QPushButton("Proceed")
        proceed_btn.setObjectName("primary_button")
        proceed_btn.setFixedHeight(48)
        proceed_btn.clicked.connect(self._on_proceed)
        layout.addWidget(proceed_btn)

    def _on_proceed(self):
        self.proceed_clicked.emit()
        self.accept()

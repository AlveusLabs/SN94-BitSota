from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QWidget
)
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path
from gui.theme import BitSOTATheme
from gui.components.common.button import PrimaryButton


class AccordionItem(QWidget):
    """Accordion item component for collapsible content sections."""
    
    @staticmethod
    def get_stylesheet():
        """Get the stylesheet for AccordionItem component."""
        return f"""
            AccordionItem {{
                background-color: transparent;
            }}
            
            QWidget {{
                background-color: transparent;
            }}
            
            QLabel#accordion_title {{
                background: transparent;
                color: {BitSOTATheme.COLOR1};
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 500;
            }}
            
            QLabel#accordion_item_title {{
                background-color: transparent;
                color: {BitSOTATheme.COLOR1};
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 500;
                line-height: 150%;
            }}
            
            QLabel#accordion_item_desc {{
                background-color: transparent;
                color: {BitSOTATheme.COLOR1_60};
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
                line-height: 150%;
                margin-left: 16px;
            }}
            
            QLabel#accordion_item_text {{
                background-color: transparent;
                color: {BitSOTATheme.COLOR1};
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
                line-height: 150%;
            }}
        """
    
    def __init__(self, title: str, content_items: list, parent=None):
        super().__init__(parent)
        self.is_expanded = False
        self.content_items = content_items
        
        # Apply component stylesheet
        self.setStyleSheet(self.get_stylesheet())
        
        self.setup_ui(title)

    def setup_ui(self, title: str):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.header = QWidget()
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(16, 16, 16, 16)
        header_layout.setSpacing(12)

        rectangle_icon = QSvgWidget(resource_path("gui/images/rectangle.svg"))
        rectangle_icon.setFixedSize(8, 8)
        header_layout.addWidget(rectangle_icon)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("accordion_title")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.arrow_icon = QSvgWidget(resource_path("gui/images/arrowt01.svg"))
        self.arrow_icon.setFixedSize(16, 16)
        header_layout.addWidget(self.arrow_icon)

        self.header.mousePressEvent = lambda e: self.toggle()
        self.main_layout.addWidget(self.header)

        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(16, 0, 16, 16)
        content_layout.setSpacing(8)

        for item in self.content_items:
            if isinstance(item, dict):
                item_label = QLabel(f"• {item['title']}")
                item_label.setObjectName("accordion_item_title")
                item_label.setWordWrap(True)
                content_layout.addWidget(item_label)

                desc_label = QLabel(item['description'])
                desc_label.setObjectName("accordion_item_desc")
                desc_label.setWordWrap(True)
                content_layout.addWidget(desc_label)
            else:
                item_label = QLabel(item)
                item_label.setObjectName("accordion_item_text")
                item_label.setWordWrap(True)
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
    """User Guide modal dialog component."""
    
    # Constants
    MODAL_WIDTH = 560
    MODAL_HEIGHT = 675
    MODAL_PADDING = 32
    SECTION_SPACING = 16
    BUTTON_WIDTH = MODAL_WIDTH - (MODAL_PADDING * 2)  # 496
    
    proceed_clicked = Signal()
    
    @staticmethod
    def get_stylesheet():
        """Get the stylesheet for UserGuideModal component."""
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
            
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            
            QWidget#scroll_content {{
                background-color: {BitSOTATheme.CONTENT_BOX_BG};
            }}
            
            QLabel#section_title {{
                background-color: transparent;
                color: {BitSOTATheme.COLOR1};
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 8px;
            }}
            
            QWidget#content_container {{
                background-color: {BitSOTATheme.COLOR1_04};
                border-radius: 8px;
            }}
            
            QWidget#separator {{
                background-color: {BitSOTATheme.COLOR1_12};
            }}
            
            QLabel#command_label {{
                background-color: {BitSOTATheme.COLOR1_08};
                color: {BitSOTATheme.COLOR1};
                font-family: "SF Mono", "Monaco", "Consolas", "Courier New", monospace;
                font-size: 13px;
                font-weight: 500;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid {BitSOTATheme.COLOR1_12};
            }}
            
            QLabel#step_number {{
                background-color: rgba(109, 96, 142, 0.16);
                color: {BitSOTATheme.COLOR1};
                border-radius: 4px;
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 500;
            }}
            
            QLabel#step_text {{
                background-color: transparent;
                color: {BitSOTATheme.COLOR1};
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
            }}
            
            QLabel#info_text {{
                background-color: transparent;
                color: {BitSOTATheme.COLOR1};
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 14px;
                font-weight: 400;
                line-height: 150%;
            }}
            
            QLabel#note_text {{
                background-color: transparent;
                color: {BitSOTATheme.COLOR1_60};
                font-family: "Geist", -apple-system, BlinkMacSystemFont, sans-serif;
                font-size: 13px;
                font-weight: 400;
                font-style: italic;
                line-height: 150%;
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
        self.setup_ui()

    def setup_ui(self):
        """Setup the main UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.MODAL_PADDING, 24, self.MODAL_PADDING, 24)
        layout.setSpacing(24)

        # Header
        layout.addLayout(self._create_header())
        
        # Scrollable content
        scroll_area = self._create_scroll_area()
        layout.addWidget(scroll_area, 1)
        
        # Proceed button
        layout.addWidget(self._create_proceed_button())

    def _create_header(self):
        """Create the modal header with title and close button."""
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        title_icon = QSvgWidget(resource_path("gui/images/frame.svg"))
        title_icon.setFixedSize(24, 24)
        header_layout.addWidget(title_icon)

        title_label = QLabel("User Guide")
        title_label.setObjectName("modal_title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        close_btn = QSvgWidget(resource_path("gui/images/cancel.svg"))
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.mousePressEvent = lambda event: self.reject()
        header_layout.addWidget(close_btn)

        return header_layout

    def _create_scroll_area(self):
        """Create the scrollable content area."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_content.setObjectName("scroll_content")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)

        # Getting Started section
        self._add_section(scroll_layout, "Getting Started", [
            AccordionItem(
                "What is BitSota",
                ["Platform to let one easily participate in AutoML experiments. You can mine on any machine regardless of compute limitation"]
            )
        ])

        scroll_layout.addSpacing(self.SECTION_SPACING)

        # Mining section
        self._add_section(scroll_layout, "Mining", [
            AccordionItem(
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
        ])

        scroll_layout.addSpacing(self.SECTION_SPACING)

        # Wallet Setup section
        wallet_items = [
            AccordionItem(
                "Load existing Bittensor wallet from laptop",
                ["People with wallets already on their machine can load their hotkeys to mine the subnet"]
            ),
            self._create_separator(),
            AccordionItem(
                "Import hotkey",
                ["Those with hotkeys but not in the folder we'd load from can enter the hotkey's secret phrase and import it"]
            ),
            self._create_separator(),

            self._create_hotkey_accordion_item(),
            self._create_separator(),

            self._create_registration_accordion_item(),
            self._create_separator(),
            AccordionItem(
                "Providing coldkey address",
                ["We need you to provide a coldkey address where we pay out to you your earnings/rewards as paying cannot be done to hotkeys"]
            )
        ]
        
        self._add_section(scroll_layout, "Wallet Setup", wallet_items)
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_content)
        return scroll_area

    def _create_proceed_button(self):
        """Create the proceed button."""
        proceed_btn = PrimaryButton("Proceed", width=self.BUTTON_WIDTH)
        proceed_btn.clicked.connect(self._on_proceed)
        return proceed_btn

    def _add_section(self, layout, title, items):
        """Add a section with title and content items.
        
        Args:
            layout: The layout to add to
            title: Section title text
            items: List of AccordionItems or widgets to add
        """
        # Section title
        title_label = QLabel(title)
        title_label.setObjectName("section_title")
        layout.addWidget(title_label)

        # Content container
        container = QWidget()
        container.setObjectName("content_container")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Add items
        for item in items:
            container_layout.addWidget(item)

        layout.addWidget(container)

    def _create_separator(self):
        """Create a horizontal separator line."""
        separator = QWidget()
        separator.setObjectName("separator")
        separator.setFixedHeight(1)
        return separator

    def _create_hotkey_accordion_item(self):
        """Create the 'How to create a hotkey' accordion item."""
        create_hotkey_content = QWidget()
        layout = QVBoxLayout(create_hotkey_content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        steps = [
            "Create your TAO wallet",
            "Secure your mnemonic and hotkey",
            "Connect your wallet"
        ]

        for i, step in enumerate(steps, 1):
            step_widget = self._create_step_widget(i, step)
            layout.addWidget(step_widget)

        accordion = AccordionItem("How to create a hotkey", [])
        accordion.content_widget.layout().addWidget(create_hotkey_content)
        return accordion

    def _create_step_widget(self, number, text):
        """Create a numbered step widget.
        
        Args:
            number: Step number
            text: Step text
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Number label
        number_label = QLabel(str(number))
        number_label.setObjectName("step_number")
        number_label.setFixedSize(24, 24)
        number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(number_label)

        # Text label
        text_label = QLabel(text)
        text_label.setObjectName("step_text")
        
        # Add link for first step
        if number == 1:
            text_label.setText(
                '<a href="https://docs.learnbittensor.org/keys/working-with-keys?create-wallet=cold-hot#creating-a-wallet-with-btcli" '
                'style="color: #0F6FFF; text-decoration: underline;">Create your TAO wallet</a>'
            )
            text_label.setOpenExternalLinks(True)
        
        layout.addWidget(text_label)
        layout.addStretch()

        return widget

    def _create_registration_accordion_item(self):
        """Create the 'Wallet Registration for Direct Mining' accordion item."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Info text
        info_label = QLabel(
            "Before you can start direct mining, you must register your wallet on the subnet using btcli:"
        )
        info_label.setObjectName("info_text")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Command
        command_label = QLabel(
            "btcli subnet register --netuid 94 --wallet.name your_wallet --wallet.hotkey your_hotkey"
        )
        command_label.setObjectName("command_label")
        command_label.setWordWrap(True)
        layout.addWidget(command_label)

        # Note
        note_label = QLabel("Note: Pool mining does not require subnet registration")
        note_label.setObjectName("note_text")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

        accordion = AccordionItem("Wallet Registration for Direct Mining", [])
        accordion.content_widget.layout().addWidget(content)
        return accordion

    def _on_proceed(self):
        """Handle proceed button click."""
        self.proceed_clicked.emit()
        self.accept()

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtSvgWidgets import QSvgWidget
from gui.resource_path import resource_path


class NavTab:
    """Navigation tab identifiers and labels"""
    
    # Tab IDs
    SETUP_WALLET = "setup_wallet"
    MINING = "mining"
    PROFILE = "profile"
    LEADERBOARD = "settings"  # Note: ID is "settings" for backward compatibility
    
    # Tab labels
    LABELS = {
        SETUP_WALLET: "Setup Wallet",
        MINING: "Mining",
        PROFILE: "Profile",
        LEADERBOARD: "Leaderboard"
    }
    
    # All tabs in display order
    ALL_TABS = [
        (SETUP_WALLET, LABELS[SETUP_WALLET]),
        (MINING, LABELS[MINING]),
        (PROFILE, LABELS[PROFILE]),
        (LEADERBOARD, LABELS[LEADERBOARD])
    ]


class NavTabButton(QWidget):
    """Top navigation tab button"""
    clicked = Signal()

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("nav_tab")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(24)  # Fixed height 24px to match design
        self.is_active = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)  # Spacing between text and indicator

        # Text label
        self.label = QLabel(text)
        self.label.setObjectName("nav_tab_label")
        layout.addWidget(self.label, 0, Qt.AlignmentFlag.AlignCenter)

        # Bottom active indicator - fixed width 24px, height 2px, centered
        indicator_container = QWidget()
        indicator_container.setFixedHeight(2)
        indicator_layout = QHBoxLayout(indicator_container)
        indicator_layout.setContentsMargins(0, 0, 0, 0)
        indicator_layout.setSpacing(0)
        
        self.indicator = QWidget()
        self.indicator.setObjectName("nav_tab_indicator")
        self.indicator.setFixedSize(24, 2)
        self.indicator.hide()
        indicator_layout.addWidget(self.indicator, 0, Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(indicator_container)

    def set_active(self, active: bool):
        self.is_active = active
        if active:
            self.label.setStyleSheet("color: #FFFFFF;")
            self.indicator.show()
        else:
            self.label.setStyleSheet("color: rgba(255, 255, 255, 0.6);")
            self.indicator.hide()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class IconButton(QWidget):
    """Icon button"""
    clicked = Signal()

    def __init__(self, icon_path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("icon_button")
        self.setFixedSize(40, 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.icon = QSvgWidget(icon_path)
        self.icon.setFixedSize(24, 24)
        layout.addWidget(self.icon)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class WalletDropdown(QWidget):
    """Wallet dropdown component - shows wallet name when connected"""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("wallet_dropdown")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)  # 8px left, 8px right
        layout.setSpacing(6)  # 12px gap between elements

        # Bitsota logo icon (left side)
        self.wallet_icon = QSvgWidget(resource_path("gui/images/logo/bitsota-logo.svg"))
        self.wallet_icon.setFixedSize(24, 24)
        layout.addWidget(self.wallet_icon)

        # Wallet name label
        self.wallet_label = QLabel("Wallet1")
        self.wallet_label.setObjectName("wallet_name_label")
        layout.addWidget(self.wallet_label)

        # Chevron dropdown icon (right side)
        self.dropdown_icon = QSvgWidget(resource_path("gui/images/chevron-down.svg"))
        self.dropdown_icon.setFixedSize(12, 12)
        layout.addWidget(self.dropdown_icon)

    def set_wallet_name(self, name: str):
        """Set wallet name"""
        self.wallet_label.setText(name)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class WalletNotConnectedButton(QWidget):
    """Wallet not connected button"""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("wallet_not_connected_button")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 10, 24, 10)  # 24px horizontal, 10px vertical
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Wallet Not Connected")
        self.label.setObjectName("wallet_not_connected_label")
        layout.addWidget(self.label)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class TopBar(QWidget):
    """Top navigation bar"""
    tab_changed = Signal(str)
    user_guide_clicked = Signal()
    wallet_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topbar")
        self.setFixedHeight(48)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.tabs = {}
        self.current_tab = None
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(0)

        # Left side: Logo
        logo_container = QWidget()
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(8, 0, 8, 0)
        logo_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.logo = QSvgWidget(resource_path("gui/images/logo-new.svg"))
        self.logo.setFixedSize(95, 24)
        logo_layout.addWidget(self.logo)

        layout.addWidget(logo_container)

        # Add flexible space, push nav bar to center
        layout.addSpacing(147 - 24 - 8 - 95)  # Calculated left spacing

        # Center: Navigation menu
        self.nav_container = QHBoxLayout()
        self.nav_container.setSpacing(32)
        self.nav_container.setContentsMargins(0, 6, 0, 0)
        layout.addLayout(self.nav_container)

        # Add navigation tabs
        for tab_id, tab_label in NavTab.ALL_TABS:
            self.add_nav_tab(tab_id, tab_label)

        layout.addStretch()

        # Right side: Icons and wallet
        right_container = QHBoxLayout()
        right_container.setSpacing(8)

        # Chat icon
        chat_btn = IconButton(resource_path("gui/images/logo/chat.svg"))
        right_container.addWidget(chat_btn)

        # Help/User guide icon
        self.user_guide_btn = IconButton(resource_path("gui/images/logo/guide-info.svg"))
        self.user_guide_btn.clicked.connect(self.user_guide_clicked.emit)
        right_container.addWidget(self.user_guide_btn)

        # Wallet dropdown (shown when wallet is connected)
        self.wallet_dropdown = WalletDropdown()
        self.wallet_dropdown.clicked.connect(self.wallet_clicked.emit)
        self.wallet_dropdown.hide()  # Hidden by default
        right_container.addWidget(self.wallet_dropdown)

        # Wallet not connected button (shown when wallet is not connected)
        self.wallet_not_connected_btn = WalletNotConnectedButton()
        self.wallet_not_connected_btn.clicked.connect(self.wallet_clicked.emit)
        self.wallet_not_connected_btn.show()  # Shown by default
        right_container.addWidget(self.wallet_not_connected_btn)

        layout.addLayout(right_container)

    def add_nav_tab(self, tab_id: str, label: str):
        """Add navigation tab"""
        tab_btn = NavTabButton(label)
        tab_btn.clicked.connect(lambda: self._on_tab_clicked(tab_id))
        self.tabs[tab_id] = tab_btn
        self.nav_container.addWidget(tab_btn)

        if not self.current_tab:
            self.set_active_tab(tab_id)

    def _on_tab_clicked(self, tab_id: str):
        """Handle tab click"""
        self.set_active_tab(tab_id)
        self.tab_changed.emit(tab_id)

    def set_active_tab(self, tab_id: str):
        """Set active tab"""
        if tab_id not in self.tabs:
            return

        self.current_tab = tab_id
        for tid, tab in self.tabs.items():
            tab.set_active(tid == tab_id)

    def set_wallet_info(self, wallet_name: str, wallet_address: str = None):
        """Set wallet info and show connected state"""
        self.wallet_dropdown.set_wallet_name(wallet_address)
        self.wallet_dropdown.show()
        self.wallet_not_connected_btn.hide()

    def hide_wallet_info(self):
        """Hide wallet info and show not connected state"""
        self.wallet_dropdown.hide()
        self.wallet_not_connected_btn.show()

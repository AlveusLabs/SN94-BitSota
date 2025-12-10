from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtGui import QDesktopServices
from gui.resource_path import resource_path


class TabButton(QWidget):
    clicked = Signal()

    def __init__(self, text: str, icon_path: str = None, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar_tab")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.icon_path = icon_path
        self.icon_widget = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        if icon_path:
            self.icon_widget = QSvgWidget()
            self.icon_widget.setFixedSize(16, 16)
            layout.addWidget(self.icon_widget)
            self._load_icon_with_color("#150049")

        self.label = QLabel(text)
        self.label.setStyleSheet("background: transparent; border: none; padding: 0;")
        layout.addWidget(self.label)
        layout.addStretch()

    def _load_icon_with_color(self, color: str):
        if not self.icon_path or not self.icon_widget:
            return

        try:
            with open(self.icon_path, 'r') as f:
                svg_content = f.read()

            svg_content = svg_content.replace('#150049', color)
            svg_content = svg_content.replace('fill="currentColor"', f'fill="{color}"')

            svg_bytes = svg_content.encode('utf-8')
            self.icon_widget.load(svg_bytes)
        except Exception:
            self.icon_widget.load(self.icon_path)

    def set_active(self, active: bool):
        if active:
            self.setObjectName("sidebar_tab_active")
            if self.icon_widget:
                self._load_icon_with_color("#FFFFFF")
            if hasattr(self, 'label'):
                self.label.setStyleSheet("background: transparent; border: none; padding: 0; color: #FFFFFF;")
        else:
            self.setObjectName("sidebar_tab")
            if self.icon_widget:
                self._load_icon_with_color("#150049")
            if hasattr(self, 'label'):
                self.label.setStyleSheet("background: transparent; border: none; padding: 0; color: rgba(21, 0, 73, 0.60);")

        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class SocialIconButton(QWidget):
    clicked = Signal()

    def __init__(self, icon_path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("social_icon")
        self.setFixedSize(32, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.icon = QSvgWidget(icon_path)
        self.icon.setFixedSize(16, 16)
        layout.addWidget(self.icon)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class Sidebar(QWidget):
    tab_changed = Signal(str)
    connect_wallet_clicked = Signal()
    user_guide_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(280)
        self.tabs = {}
        self.current_tab = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        logo_container = QWidget()
        logo_container.setObjectName("sidebar_logo_container")
        logo_layout = QHBoxLayout(logo_container)
        logo_layout.setContentsMargins(20, 16, 20, 16)
        logo_layout.setSpacing(0)

        self.logo = QSvgWidget(resource_path("gui/images/logo.svg"))
        self.logo.setFixedSize(95, 24)
        logo_layout.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(logo_container)

        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(20, 20, 20, 16)
        content_layout.setSpacing(20)

        self.connect_wallet_btn = QPushButton("Connect Wallet")
        self.connect_wallet_btn.setObjectName("primary_button")
        self.connect_wallet_btn.setFixedSize(232, 40)
        self.connect_wallet_btn.clicked.connect(self.connect_wallet_clicked.emit)
        content_layout.addWidget(self.connect_wallet_btn)

        self.wallet_info_box = self._create_wallet_info_box()
        self.wallet_info_box.hide()
        content_layout.addWidget(self.wallet_info_box)

        self.tabs_container = QVBoxLayout()
        self.tabs_container.setSpacing(4)
        content_layout.addLayout(self.tabs_container)

        content_layout.addStretch()

        user_guide_separator = QWidget()
        user_guide_separator.setFixedHeight(1)
        user_guide_separator.setStyleSheet("background-color: rgba(21, 0, 73, 0.12);")
        content_layout.addWidget(user_guide_separator)

        self.user_guide_btn = TabButton("User Guide", resource_path("gui/images/Info Circle.svg"))
        self.user_guide_btn.clicked.connect(self.user_guide_clicked.emit)
        content_layout.addWidget(self.user_guide_btn)

        follow_separator = QWidget()
        follow_separator.setFixedHeight(1)
        follow_separator.setStyleSheet("background-color: rgba(21, 0, 73, 0.12);")
        content_layout.addWidget(follow_separator)

        follow_section = QWidget()
        follow_layout = QVBoxLayout(follow_section)
        follow_layout.setContentsMargins(0, 0, 0, 0)
        follow_layout.setSpacing(6)

        follow_label = QLabel("Follow us")
        follow_label.setObjectName("sidebar_follow_label")
        follow_layout.addWidget(follow_label)

        social_icons_container = QHBoxLayout()
        social_icons_container.setSpacing(8)

        social_platforms = [
            (resource_path("gui/images/Discord.svg"), None),
            (resource_path("gui/images/Website.svg"), "https://bitsota.ai/"),
            (resource_path("gui/images/X.svg"), "https://x.com/bitsota"),
            (resource_path("gui/images/GitHub.svg"), "https://github.com/AlveusLabs/SN94-BitSota"),
        ]

        for icon_path, url in social_platforms:
            icon_btn = SocialIconButton(icon_path)
            if url:
                icon_btn.clicked.connect(lambda checked=False, u=url: self._open_url(u))
            social_icons_container.addWidget(icon_btn)

        social_icons_container.addStretch()
        follow_layout.addLayout(social_icons_container)

        content_layout.addWidget(follow_section)

        layout.addWidget(content_container)

    def add_tab(self, tab_id: str, label: str, icon_path: str = None):
        tab_btn = TabButton(label, icon_path)
        tab_btn.clicked.connect(lambda: self._on_tab_clicked(tab_id))
        self.tabs[tab_id] = tab_btn
        self.tabs_container.addWidget(tab_btn)

        if self.current_tab is None:
            self.set_active_tab(tab_id)

    def _on_tab_clicked(self, tab_id: str):
        self.set_active_tab(tab_id)
        self.tab_changed.emit(tab_id)

    def set_active_tab(self, tab_id: str):
        if tab_id not in self.tabs:
            return

        for tid, btn in self.tabs.items():
            btn.set_active(tid == tab_id)

        self.current_tab = tab_id

    def _create_wallet_info_box(self):
        wallet_box = QWidget()
        wallet_box.setObjectName("sidebar_wallet_info")
        wallet_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        wallet_layout = QVBoxLayout(wallet_box)
        wallet_layout.setContentsMargins(16, 12, 16, 12)
        wallet_layout.setSpacing(8)

        wallet_header = QHBoxLayout()
        wallet_header.setSpacing(8)

        self.wallet_name_label = QLabel("Wallet1")
        self.wallet_name_label.setObjectName("sidebar_wallet_name")
        wallet_header.addWidget(self.wallet_name_label)

        arrow_icon = QSvgWidget(resource_path("gui/images/arrow_2.svg"))
        arrow_icon.setFixedSize(12, 12)
        wallet_header.addWidget(arrow_icon)

        wallet_header.addStretch()
        wallet_layout.addLayout(wallet_header)

        address_container = QHBoxLayout()
        address_container.setSpacing(8)

        self.wallet_address_label = QLabel("0x4892...81ae")
        self.wallet_address_label.setObjectName("sidebar_wallet_address")
        address_container.addWidget(self.wallet_address_label)

        copy_icon = QSvgWidget(resource_path("gui/images/copy-06.svg"))
        copy_icon.setFixedSize(16, 16)
        copy_icon.setCursor(Qt.CursorShape.PointingHandCursor)
        address_container.addWidget(copy_icon)

        address_container.addStretch()
        wallet_layout.addLayout(address_container)

        return wallet_box

    def set_wallet_info(self, wallet_name: str, wallet_address: str):
        self.wallet_name_label.setText(wallet_name)
        self.wallet_address_label.setText(wallet_address)
        self.connect_wallet_btn.hide()
        self.wallet_info_box.show()

    def hide_connect_wallet_button(self):
        self.connect_wallet_btn.hide()

    def show_connect_wallet_button(self):
        self.wallet_info_box.hide()
        self.connect_wallet_btn.show()

    def _open_url(self, url: str):
        QDesktopServices.openUrl(QUrl(url))

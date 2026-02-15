# PySide6 imports
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

# Local application imports
from gui.components.common.normal_tab_switcher import NormalTabSwitcher
from .direct_mining_screen import DirectMiningScreen
from .pool_mining_screen import PoolMiningScreen

# Description text per tab
DIRECT_MINING_DESCRIPTION = (
    "Connect straight to Bittensor validators, ideal for users who want "
    "complete control over their mining operations."
)
POOL_MINING_DESCRIPTION = (
    "Join a Mining Pool for simplified setup and shared resources. "
    "Ideal for beginners."
)


class MiningScreen(QWidget):
    """Main mining screen — thin coordinator for Direct and Pool Mining tabs.

    Layout:
        content_box (white background)
        ├── [Shared] Tab switcher (Direct Mining / Pool Mining)
        ├── [Shared] Description label (text changes per tab)
        ├── [Tab: Direct] DirectMiningScreen   ← shown by default
        └── [Tab: Pool]   PoolMiningScreen     ← hidden by default
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setup_ui()

    # ========== UI Setup ==========

    def setup_ui(self):
        """Initialize the user interface layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        content_box = QWidget()
        content_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content_layout = QVBoxLayout(content_box)
        content_layout.setSpacing(0)

        # Shared header: Tab switcher + Description
        self._setup_shared_header(content_layout)

        # Direct Mining content (shown by default)
        self.direct_content = DirectMiningScreen(main_window=self.main_window)
        content_layout.addWidget(self.direct_content)

        # Pool Mining content (hidden by default)
        self.pool_content = PoolMiningScreen(main_window=self.main_window)
        self.pool_content.hide()
        content_layout.addWidget(self.pool_content, 1)

        main_layout.addWidget(content_box)

    def _setup_shared_header(self, parent_layout: QVBoxLayout):
        """Create the shared tab switcher and description label."""
        mining_tabs = [
            ("direct", "Direct Mining"),
            ("pool", "Pool Mining"),
        ]
        self.mining_tab_switcher = NormalTabSwitcher(
            tabs=mining_tabs,
            on_tab_changed=self._on_mining_tab_changed,
        )
        parent_layout.addWidget(self.mining_tab_switcher)
        parent_layout.addSpacing(12)

        self.description = QLabel(DIRECT_MINING_DESCRIPTION)
        self.description.setObjectName("mining_description")
        self.description.setWordWrap(True)
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        parent_layout.addWidget(self.description)
        parent_layout.addSpacing(24)

    # ========== Tab Switching ==========

    def _on_mining_tab_changed(self, tab_id: str):
        """Handle mining mode tab changes (direct/pool)."""
        if tab_id == "pool":
            self._switch_to_pool()
        else:
            self._switch_to_direct()

    def _switch_to_pool(self):
        """Switch to pool mining mode."""
        self.direct_content.hide()
        self.pool_content.show()
        self.description.setText(POOL_MINING_DESCRIPTION)

    def _switch_to_direct(self):
        """Switch to direct mining mode."""
        self.pool_content.hide()
        self.direct_content.show()
        self.description.setText(DIRECT_MINING_DESCRIPTION)

    # ========== Public Methods (forwarded to sub-screens) ==========

    def update_wallet_status(self, wallet_name: str):
        """Update wallet status display on both sub-screens."""
        self.direct_content.update_wallet_status(wallet_name)
        self.pool_content.update_wallet_status(wallet_name)

    def update_connection_status(self, connected: bool):
        """Update connection and mining status indicators."""
        self.direct_content.update_connection_status(connected)

    def update_global_sota(self):
        """Fetch and update the global SOTA score."""
        self.direct_content.update_global_sota()

    def _on_invite_code_verified(self):
        """Forward invite code verification to direct mining screen."""
        self.direct_content._on_invite_code_verified()

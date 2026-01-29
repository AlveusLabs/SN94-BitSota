"""Navigation Manager - handles screen switching and navigation routing"""

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QStackedWidget


class NavigationManager(QObject):
    """Manages application screen navigation and tab switching"""

    # Signals
    screen_changed = Signal(str)  # screen_name
    show_coming_soon = Signal(str, str)  # title, message
    tab_changed = Signal(str)  # tab_id

    def __init__(
        self,
        content_stack: QStackedWidget,
        screen_stack: QStackedWidget,
        topbar,
        parent=None
    ):
        """
        Initialize navigation manager
        
        Args:
            content_stack: Main content stack (contains start screen and app container)
            screen_stack: Screen stack (contains wallet, mining, profile screens, etc.)
            topbar: Top navigation bar
            parent: Parent object
        """
        super().__init__(parent)
        self.content_stack = content_stack
        self.screen_stack = screen_stack
        self.topbar = topbar
        
        # Screen references
        self.wallet_screen = None
        self.mining_screen = None
        self.profile_screen = None

    def set_screens(self, wallet_screen, mining_screen, profile_screen):
        """
        Set screen references
        
        Args:
            wallet_screen: Wallet screen
            mining_screen: Mining screen
            profile_screen: Profile screen
        """
        self.wallet_screen = wallet_screen
        self.mining_screen = mining_screen
        self.profile_screen = profile_screen

    def handle_start_click(self):
        """Handle start button click on start screen"""
        # Show user guide after start screen click
        # Actual user guide display is handled by ModalManager
        pass

    def show_main_app(self):
        """Show main app interface (switch from start screen to main app)"""
        self.content_stack.setCurrentIndex(1)
        self.screen_changed.emit("main_app")

    def handle_tab_change(self, tab_id: str):
        """
        Handle navigation tab switching
        
        Args:
            tab_id: Tab ID (setup_wallet, mining, settings, profile)
        """
        if tab_id == "setup_wallet":
            self.navigate_to_wallet()
        elif tab_id == "mining":
            self.navigate_to_mining()
        elif tab_id == "profile":
            # Profile feature not yet implemented, show coming soon prompt
            # self.show_coming_soon.emit(
            #     "Profile Screen",
            #     "The Profile screen is coming soon! This screen will show your mining "
            #     "history, rewards, and balances from both Direct Mining and Pool Mining. "
            #     "You'll be able to view detailed statistics and claim your rewards."
            # )
            # Switch back to mining tab
            # self.topbar.set_active_tab("mining")
            self.navigate_to_profile()
        elif tab_id == "settings":
            # Settings feature can be implemented here
            pass

    def navigate_to_wallet(self):
        """Navigate to wallet screen"""
        if self.wallet_screen:
            self.screen_stack.setCurrentWidget(self.wallet_screen)
            self.screen_changed.emit("wallet")

    def navigate_to_mining(self):
        """Navigate to mining screen"""
        if self.mining_screen:
            self.screen_stack.setCurrentWidget(self.mining_screen)
            self.screen_changed.emit("mining")

    def navigate_to_profile(self):
        """Navigate to profile screen"""
        if self.profile_screen:
            self.screen_stack.setCurrentWidget(self.profile_screen)
            self.screen_changed.emit("profile")

    def handle_wallet_connect(self):
        """Handle wallet connection request"""
        self.topbar.set_active_tab("setup_wallet")
        self.navigate_to_wallet()

    def handle_stack_change(self, index: int):
        """
        Handle content stack change
        
        Args:
            index: Stack index
        """
        # Additional operations can be performed when stack page changes
        # For example, update overlay geometry, etc.
        pass

    def auto_navigate_to_mining(self):
        """Auto-navigate to mining screen (used after wallet auto-load)"""
        self.show_main_app()
        self.topbar.set_active_tab("mining")
        self.navigate_to_mining()

    def auto_navigate_to_profile(self):
        """Auto-navigate to profile screen (used after wallet auto-load)"""
        self.show_main_app()
        self.topbar.set_active_tab("profile")
        self.navigate_to_profile()

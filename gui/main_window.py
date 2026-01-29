from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QStackedWidget,
)

from gui.theme import BitSOTATheme
from gui.screens import StartScreen, WalletScreen, MiningScreen, ProfileScreen
from gui.components.navigation.topbar import TopBar
from gui.components.common.overlay import ModalOverlay
from gui.managers import (
    WalletManager,
    ClientManager,
    NavigationManager,
    ModalManager,
    UpdateManager,
    WindowStyleManager,
)

class MiningWindow(QMainWindow):
    """Main window - coordinates managers and screens"""

    def __init__(self):
        super().__init__()
        
        # Initialize window style manager
        self.style_manager = WindowStyleManager(self)
        self.style_manager.setup_window()
        
        # Create UI
        self._create_ui()
        
        # Apply theme
        self._apply_theme()
        
        # Initialize managers
        self._initialize_managers()
        
        # Connect signals
        self._connect_signals()
        
        # Try to auto-load wallet
        self.wallet_manager.auto_load_wallet()
        
        # Setup update checker
        self.update_manager.setup()

    def showEvent(self, event):
        """Window show event - used for setting titlebar color"""
        super().showEvent(event)
        self.style_manager.handle_show_event(event)

    def _create_ui(self):
        """Create UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Content stack (start screen + main app)
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack)

        # Start screen
        self.start_screen = StartScreen()
        self.content_stack.addWidget(self.start_screen)

        # Main app container
        self.app_container = QWidget()
        self.app_container.setObjectName("app_container")
        app_layout = QVBoxLayout(self.app_container)
        app_layout.setContentsMargins(0, 0, 0, 0)
        app_layout.setSpacing(0)

        # Top navigation bar
        self.topbar = TopBar()
        app_layout.addWidget(self.topbar)

        # Content area wrapper (with padding)
        self.content_wrapper = QWidget()
        self.content_wrapper.setObjectName("content_wrapper")
        self.content_wrapper.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        content_wrapper_layout = QVBoxLayout(self.content_wrapper)
        content_wrapper_layout.setContentsMargins(24, 24, 24, 24)

        # Screen stack
        self.screen_stack = QStackedWidget()
        self.screen_stack.setObjectName("screen_stack")
        self.screen_stack.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        self.wallet_screen = WalletScreen()
        self.mining_screen = MiningScreen(main_window=self)
        self.profile_screen = ProfileScreen()
        
        self.screen_stack.addWidget(self.wallet_screen)
        self.screen_stack.addWidget(self.mining_screen)
        self.screen_stack.addWidget(self.profile_screen)
        
        content_wrapper_layout.addWidget(self.screen_stack)
        app_layout.addWidget(self.content_wrapper, 1)

        self.content_stack.addWidget(self.app_container)

        # Create modal overlay
        self.modal_overlay = ModalOverlay(central_widget)
        self.modal_overlay.hide()

        # Keep old sidebar reference for compatibility (but not added to layout)
        self.sidebar = None

    def _apply_theme(self):
        """Apply theme styles"""
        self.setStyleSheet(BitSOTATheme.get_main_stylesheet())
        fonts = BitSOTATheme.get_font_system()
        self.setFont(fonts["primary"])

    def _initialize_managers(self):
        """Initialize all managers"""
        # Wallet manager
        self.wallet_manager = WalletManager(self)
        
        # Client manager
        self.client_manager = ClientManager(self)
        
        # Navigation manager
        self.navigation_manager = NavigationManager(
            self.content_stack,
            self.screen_stack,
            self.topbar,
            self
        )
        self.navigation_manager.set_screens(
            self.wallet_screen,
            self.mining_screen,
            self.profile_screen
        )
        
        # Modal manager
        self.modal_manager = ModalManager(
            self,
            self.modal_overlay,
            self.content_stack,
            self.screen_stack,
            self.app_container,
            self
        )
        
        # Update manager
        self.update_manager = UpdateManager(self, self)

    def _connect_signals(self):
        """Connect signals and slots"""
        # Start screen
        self.start_screen.start_clicked.connect(self._on_start_clicked)
        
        # Top navigation bar
        self.topbar.tab_changed.connect(self.navigation_manager.handle_tab_change)
        self.topbar.user_guide_clicked.connect(self.modal_manager.show_user_guide)
        self.topbar.wallet_clicked.connect(self._on_wallet_dropdown_clicked)
        
        # Wallet screen
        self.wallet_screen.wallet_loaded.connect(self._on_wallet_loaded)
        self.wallet_screen.hotkey_imported.connect(self._on_hotkey_imported)
        
        # Content stack change
        self.content_stack.currentChanged.connect(self._on_stack_changed)
        
        # Wallet manager
        self.wallet_manager.wallet_loaded.connect(self._on_wallet_manager_loaded)
        self.wallet_manager.hotkey_imported.connect(self._on_wallet_manager_imported)
        self.wallet_manager.wallet_status_updated.connect(self._update_mining_screen_status)
        
        # Modal manager
        self.modal_manager.user_guide_completed.connect(self.navigation_manager.show_main_app)
        self.modal_manager.coldkey_address_submitted.connect(self.wallet_manager.save_coldkey_address)
        self.modal_manager.invite_code_verified.connect(self._on_invite_code_verified)
        self.modal_manager.wallet_selected.connect(self._on_wallet_selected_from_modal)
        self.modal_manager.terms_accepted.connect(self._on_terms_accepted)
        self.modal_manager.wallet_import_success_confirmed.connect(self._on_wallet_import_success_confirmed)
        
        # Navigation manager
        self.navigation_manager.show_coming_soon.connect(self.modal_manager.show_coming_soon)

    def resizeEvent(self, event):
        """Window resize event"""
        super().resizeEvent(event)
        self.modal_manager.handle_resize_event()

    def _on_stack_changed(self, index):
        """Content stack change event"""
        self.modal_manager.handle_stack_change()

    def _on_start_clicked(self):
        """Start button click"""
        self.modal_manager.show_user_guide()

    def _on_wallet_dropdown_clicked(self):
        """Wallet dropdown click"""
        # Can show wallet details or switch wallet modal here
        pass

    def _on_wallet_loaded(
        self,
        wallet_name: str,
        hotkey_name: str,
        use_existing_coldkey: bool,
        coldkey_address: str
    ):
        """Handle wallet loaded from wallet screen"""
        success, error = self.wallet_manager.load_wallet(
            wallet_name,
            hotkey_name,
            use_existing_coldkey,
            coldkey_address
        )
        
        if not success:
            self.modal_manager.show_error("Wallet Load Error", error)
            return
        
        # Initialize client
        wallet = self.wallet_manager.get_wallet()
        if wallet:
            self.client_manager.initialize_client(wallet)
        
        # Check if coldkey prompt is needed
        if self.wallet_manager.needs_coldkey_prompt(use_existing_coldkey, coldkey_address):
            self.modal_manager.show_coldkey_prompt()

    def _on_hotkey_imported(self, hotkey_name: str, mnemonic: str, coldkey_address: str):
        """Handle hotkey imported from wallet screen"""
        success, error = self.wallet_manager.import_hotkey(
            hotkey_name,
            mnemonic,
            coldkey_address
        )
        
        if not success:
            self.modal_manager.show_error("Import Failed", error)
            return
        
        # Initialize client
        wallet = self.wallet_manager.get_wallet()
        if wallet:
            self.client_manager.initialize_client(wallet)
        
        # Check if coldkey prompt is needed
        if not coldkey_address:
            self.modal_manager.show_coldkey_prompt()

    def _on_wallet_manager_loaded(self, wallet, wallet_name: str, display_address: str):
        """Handle wallet loaded by wallet manager"""
        # Update topbar
        self.topbar.set_wallet_info(wallet_name, display_address)
        
        # Initialize client
        if wallet:
            self.client_manager.initialize_client(wallet)
        
        # If auto-loaded, navigate to mining screen
        # Check by whether currently on start screen
        if self.content_stack.currentIndex() == 0:
            self.navigation_manager.auto_navigate_to_mining()

    def _on_wallet_manager_imported(self, wallet, wallet_name: str, display_address: str):
        """Handle hotkey imported by wallet manager"""
        # Update topbar
        self.topbar.set_wallet_info(wallet_name, display_address)
        
        # Initialize client
        if wallet:
            self.client_manager.initialize_client(wallet)

    def _update_mining_screen_status(self, wallet_name: str):
        """Update mining screen status"""
        if hasattr(self.mining_screen, 'update_wallet_status'):
            self.mining_screen.update_wallet_status(wallet_name)
            self.mining_screen.update_global_sota()

    def _on_invite_code_verified(self):
        """Handle invite code verification from modal_manager"""
        # Route to mining screen
        if hasattr(self.mining_screen, '_on_invite_code_verified'):
            self.mining_screen._on_invite_code_verified()

    def _on_wallet_selected_from_modal(self, wallet_name: str, hotkey_name: str, use_existing_coldkey: bool, coldkey_address: str):
        """Handle wallet selection from modal_manager"""
        # Route to wallet screen's signal
        self.wallet_screen.wallet_loaded.emit(wallet_name, hotkey_name, use_existing_coldkey, coldkey_address)

    def _on_terms_accepted(self):
        """Handle terms acceptance from modal_manager"""
        # Get pending import data from import screen
        if hasattr(self.wallet_screen.import_screen, '_pending_import_data') and self.wallet_screen.import_screen._pending_import_data:
            hotkey_name, mnemonic, coldkey_address = self.wallet_screen.import_screen._pending_import_data
            self.wallet_screen.import_screen.imported.emit(hotkey_name, mnemonic, coldkey_address)
            self.wallet_screen.import_screen._pending_import_data = None

    def _on_wallet_import_success_confirmed(self):
        """Handle wallet import success confirmation from modal_manager"""
        # Finalize import process
        if hasattr(self.wallet_screen, '_pending_finalize_data') and self.wallet_screen._pending_finalize_data:
            hotkey_name, mnemonic, coldkey_address = self.wallet_screen._pending_finalize_data
            self.wallet_screen._finalize_import(hotkey_name, mnemonic, coldkey_address)
            self.wallet_screen._pending_finalize_data = None

    def get_current_sota(self):
        """Get current SOTA threshold (kept for backward compatibility)"""
        return self.client_manager.fetch_current_sota()

    def _get_relay_endpoint_from_config(self):
        """Get relay endpoint (kept for backward compatibility)"""
        return self.client_manager.get_relay_endpoint()

    def _prompt_for_coldkey_address(self):
        """Prompt for coldkey address (kept for backward compatibility)"""
        self.modal_manager.show_coldkey_prompt()

    # Keep these properties for backward compatibility
    @property
    def wallet(self):
        """Get current wallet"""
        return self.wallet_manager.get_wallet()

    @property
    def client(self):
        """Get current client"""
        return self.client_manager.get_client()

    @property
    def coldkey_address(self):
        """Get coldkey address"""
        return self.wallet_manager.get_coldkey_address()
    
    def show_modal_with_overlay(self, modal):
        """Show modal with overlay (kept for backward compatibility)"""
        return self.modal_manager.show_modal(modal)

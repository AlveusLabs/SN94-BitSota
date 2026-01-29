"""Modal Manager - handles modal display and overlay management"""

from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtWidgets import QStackedWidget, QWidget

from gui.components.modals.user_guide import UserGuideModal
from gui.components.modals.coldkey_address import ColdkeyAddressModal
from gui.components.modals.coming_soon import ComingSoonModal
from gui.components.common.overlay import ModalOverlay
from gui.components.modals.import_confirmation import (
    ErrorModal,
    TermsAcceptanceModal,
    WalletImportedSuccessModal
)
from gui.components.modals.invite_code import InviteCodeModal
from gui.components.modals.wallet_selection import WalletSelectionModal


class ModalManager(QObject):
    """Manages modal dialogs and overlay"""

    # Signals
    user_guide_completed = Signal()
    coldkey_address_submitted = Signal(str)  # address
    invite_code_verified = Signal()
    wallet_selected = Signal(str, str, bool, str)  # wallet_name, hotkey_name, use_existing_coldkey, coldkey_address
    terms_accepted = Signal()
    wallet_import_success_confirmed = Signal()

    def __init__(
        self,
        main_window,
        modal_overlay: ModalOverlay,
        content_stack: QStackedWidget,
        screen_stack: QWidget,
        app_container: QWidget,
        parent=None
    ):
        """
        Initialize modal manager
        
        Args:
            main_window: Main window object
            modal_overlay: Modal overlay component
            content_stack: Content stack
            screen_stack: Screen stack
            app_container: App container
            parent: Parent object
        """
        super().__init__(parent)
        self.main_window = main_window
        self.modal_overlay = modal_overlay
        self.content_stack = content_stack
        self.screen_stack = screen_stack
        self.app_container = app_container

    def show_modal(self, modal):
        """
        Show modal dialog and manage overlay
        
        Args:
            modal: Modal object to display
            
        Returns:
            Return value of the modal
        """
        # Update overlay geometry to match screen_stack
        self.update_overlay_geometry()
        self.modal_overlay.raise_()
        self.modal_overlay.show()

        # Show dialog
        result = modal.exec()

        # Hide overlay after dialog is closed
        self.modal_overlay.hide()

        return result

    def update_overlay_geometry(self):
        """Update overlay position and size to match content area"""
        # Only update when app_container is visible
        if self.content_stack.currentWidget() == self.app_container:
            # Get screen_stack position relative to central widget
            central_widget = self.main_window.centralWidget()
            pos = self.screen_stack.mapTo(central_widget, self.screen_stack.rect().topLeft())
            # Set overlay geometry to match screen_stack
            self.modal_overlay.setGeometry(
                pos.x(),
                pos.y(),
                self.screen_stack.width(),
                self.screen_stack.height()
            )

    def handle_resize_event(self):
        """Handle window resize event"""
        self.update_overlay_geometry()

    def handle_stack_change(self):
        """Handle stack change event"""
        # Use QTimer to ensure layout is complete before updating
        QTimer.singleShot(0, self.update_overlay_geometry)

    def show_user_guide(self):
        """Show user guide modal"""
        guide_modal = UserGuideModal(parent=self.main_window)
        guide_modal.proceed_clicked.connect(self._on_user_guide_proceed)
        self.show_modal(guide_modal)

    def _on_user_guide_proceed(self):
        """Handle user guide completion"""
        self.user_guide_completed.emit()

    def show_coldkey_prompt(self):
        """Show coldkey address input prompt"""
        coldkey_modal = ColdkeyAddressModal(parent=self.main_window)
        coldkey_modal.address_submitted.connect(self._on_coldkey_submitted)
        self.show_modal(coldkey_modal)

    def _on_coldkey_submitted(self, address: str):
        """Handle coldkey address submission"""
        self.coldkey_address_submitted.emit(address)

    def show_coming_soon(self, title: str, message: str):
        """
        Show 'coming soon' modal
        
        Args:
            title: Title
            message: Message content
        """
        modal = ComingSoonModal(title, message, parent=self.main_window)
        self.show_modal(modal)

    def show_error(self, title: str, message: str):
        """
        Show error modal
        
        Args:
            title: Error title
            message: Error message
        """
        error_modal = ErrorModal(title, message, parent=self.main_window)
        self.show_modal(error_modal)

    def show_invite_code(self, relay_url: str, wallet, coldkey_address: str = None):
        """
        Show invite code modal
        
        Args:
            relay_url: Relay URL
            wallet: Wallet object
            coldkey_address: Optional coldkey address
        """
        invite_modal = InviteCodeModal(
            relay_url=relay_url,
            wallet=wallet,
            coldkey_address=coldkey_address,
            parent=self.main_window
        )
        invite_modal.code_verified.connect(self._on_invite_code_verified)
        self.show_modal(invite_modal)

    def _on_invite_code_verified(self):
        """Handle invite code verification"""
        self.invite_code_verified.emit()

    def show_wallet_selection(self):
        """Show wallet selection modal"""
        wallet_modal = WalletSelectionModal(parent=self.main_window)
        wallet_modal.wallet_selected.connect(self._on_wallet_selected)
        self.show_modal(wallet_modal)

    def _on_wallet_selected(self, wallet_name: str, hotkey_name: str, use_existing_coldkey: bool, coldkey_address: str):
        """Handle wallet selection"""
        self.wallet_selected.emit(wallet_name, hotkey_name, use_existing_coldkey, coldkey_address)

    def show_terms_acceptance(self):
        """Show terms acceptance modal"""
        terms_modal = TermsAcceptanceModal(parent=self.main_window)
        terms_modal.confirmed.connect(self._on_terms_accepted)
        self.show_modal(terms_modal)

    def _on_terms_accepted(self):
        """Handle terms acceptance"""
        self.terms_accepted.emit()

    def show_wallet_import_success(self):
        """Show wallet imported successfully modal"""
        success_modal = WalletImportedSuccessModal(parent=self.main_window)
        success_modal.start_mining.connect(self._on_wallet_import_success_confirmed)
        success_modal.rejected.connect(self._on_wallet_import_success_confirmed)
        self.show_modal(success_modal)

    def _on_wallet_import_success_confirmed(self):
        """Handle wallet import success confirmation"""
        self.wallet_import_success_confirmed.emit()

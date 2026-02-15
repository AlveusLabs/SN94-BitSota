"""GUI components package - reorganized into categorized structure

Components are now categorized into sub-packages:
- common: Common UI components (buttons, overlay, tab switcher)
- modals: All modal dialog components
- navigation: Navigation components (topbar, etc.)

For backward compatibility, all components can still be imported from here.
"""

# Import common components from sub-packages
from .common.button import PrimaryButton, SecondaryButton
from .common.overlay import ModalOverlay, show_modal_with_overlay
from .common.tab_switcher import TabSwitcher
from .common.select_input import SelectInput

# Import navigation components from sub-packages
from .navigation.topbar import TopBar

# Import modal components from sub-packages
from .modals.base import ConfirmationModal
from .modals.coldkey_address import ColdkeyAddressModal
from .modals.coming_soon import ComingSoonModal
from .modals.import_confirmation import (
    ErrorModal,
    TermsAcceptanceModal,
    WalletImportedSuccessModal,
)
from .modals.invite_code import InviteCodeModal
from .modals.update import UpdateAvailableModal
from .modals.user_guide import UserGuideModal
from .modals.wallet_selection import WalletSelectionModal

# Export all components (maintain backward compatibility)
__all__ = [
    # Common components
    "PrimaryButton",
    "SecondaryButton",
    "ModalOverlay",
    "show_modal_with_overlay",
    "TabSwitcher",
    "SelectInput",
    # Navigation components
    "TopBar",
    # Modal components
    "ConfirmationModal",
    "ColdkeyAddressModal",
    "ComingSoonModal",
    "ErrorModal",
    "TermsAcceptanceModal",
    "WalletImportedSuccessModal",
    "InviteCodeModal",
    "UpdateAvailableModal",
    "UserGuideModal",
    "WalletSelectionModal",
]

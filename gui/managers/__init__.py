"""GUI Managers module

Splits various responsibilities of the main window into independent manager classes.
"""

from .wallet_manager import WalletManager
from .client_manager import ClientManager
from .navigation_manager import NavigationManager
from .modal_manager import ModalManager
from .update_manager import UpdateManager
from .window_style_manager import WindowStyleManager

__all__ = [
    "WalletManager",
    "ClientManager",
    "NavigationManager",
    "ModalManager",
    "UpdateManager",
    "WindowStyleManager",
]

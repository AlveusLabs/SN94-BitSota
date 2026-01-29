"""GUI Managers - 管理器模块

负责将主窗口的各种职责拆分成独立的管理器类。
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

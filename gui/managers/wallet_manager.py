"""钱包管理器 - 负责钱包加载、导入和 coldkey 管理"""

import uuid
from typing import Optional
from PySide6.QtCore import QObject, Signal

from bittensor_network.wallet import Wallet
from gui.wallet_utils_gui import (
    get_wallet_dir,
    get_bittensor_wallet_dir,
    discover_wallets,
    get_coldkey_address,
    save_coldkey_address,
    save_wallet_settings,
    get_last_wallet,
)


class WalletManager(QObject):
    """管理钱包的加载、导入和 coldkey 地址"""

    # 信号
    wallet_loaded = Signal(object, str, str)  # wallet, wallet_name, display_address
    hotkey_imported = Signal(object, str, str)  # wallet, wallet_name, display_address
    coldkey_submitted = Signal(str)  # coldkey_address
    wallet_status_updated = Signal(str)  # wallet_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.wallet: Optional[Wallet] = None
        self.coldkey_address: Optional[str] = None

    def load_wallet(
        self,
        wallet_name: str,
        hotkey_name: str,
        use_existing_coldkey: bool,
        coldkey_address: str
    ) -> tuple[bool, Optional[str]]:
        """
        加载钱包
        
        Args:
            wallet_name: 钱包名称
            hotkey_name: 热钥名称
            use_existing_coldkey: 是否使用现有 coldkey
            coldkey_address: Coldkey 地址
            
        Returns:
            (success, error_message)
        """
        # 查找钱包目录
        wallet_dir = None
        wallets = discover_wallets()
        for w_name, hotkeys, source in wallets:
            if w_name == wallet_name and hotkey_name in hotkeys:
                if source == "bittensor":
                    wallet_dir = str(get_bittensor_wallet_dir())
                else:
                    wallet_dir = str(get_wallet_dir())
                break

        if not wallet_dir:
            wallet_dir = str(get_wallet_dir())

        # 创建钱包对象
        self.wallet = Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_dir)

        # 获取热钥地址
        try:
            hotkey = self.wallet.get_hotkey()
            address = hotkey.ss58_address
            short_address = f"{address[:6]}...{address[-4:]}" if address else "Unknown"
        except Exception as e:
            print(f"Error loading hotkey: {e}")
            return False, f"Error loading hotkey: {e}"

        # 处理 coldkey 地址
        if use_existing_coldkey and coldkey_address:
            self.coldkey_address = coldkey_address
            save_coldkey_address(coldkey_address)
            save_wallet_settings(wallet_name, hotkey_name, coldkey_address)
            print(f"Using existing coldkey address: {coldkey_address}")
            short_coldkey = (
                f"{coldkey_address[:6]}...{coldkey_address[-4:]}"
                if coldkey_address and len(coldkey_address) > 10
                else coldkey_address
            )
            display_address = short_coldkey
        else:
            display_address = short_address

        # 发送信号
        self.wallet_loaded.emit(self.wallet, wallet_name, display_address)
        self.wallet_status_updated.emit(wallet_name)

        return True, None

    def import_hotkey(
        self,
        hotkey_name: str,
        mnemonic: str,
        coldkey_address: str
    ) -> tuple[bool, Optional[str]]:
        """
        导入热钥
        
        Args:
            hotkey_name: 热钥名称
            mnemonic: 助记词
            coldkey_address: Coldkey 地址
            
        Returns:
            (success, error_message)
        """
        wallet_name = f"imported_{str(uuid.uuid4())[:8]}"
        wallet_dir = str(get_wallet_dir())

        try:
            # 创建钱包并导入热钥
            self.wallet = Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_dir)
            self.wallet.import_hotkey_from_mnemonic(mnemonic, overwrite=True)

            # 保存 coldkey 地址
            self.coldkey_address = coldkey_address if coldkey_address else None
            save_wallet_settings(wallet_name, hotkey_name, coldkey_address)

            # 获取热钥地址
            hotkey = self.wallet.get_hotkey()
            address = hotkey.ss58_address
            short_address = f"{address[:6]}...{address[-4:]}" if address else "Unknown"

            # 发送信号
            self.hotkey_imported.emit(self.wallet, wallet_name, short_address)
            self.wallet_status_updated.emit(wallet_name)

            return True, None
        except Exception as e:
            error_msg = f"Failed to import hotkey. Please try again.\n\nError: {str(e)}"
            print(f"Error importing hotkey: {e}")
            return False, error_msg

    def auto_load_wallet(self) -> bool:
        """
        尝试自动加载上次使用的钱包
        
        Returns:
            是否成功加载
        """
        last_wallet_name, last_hotkey_name = get_last_wallet()

        if not last_wallet_name or not last_hotkey_name:
            return False

        wallets = discover_wallets()
        wallet_dir = None
        wallet_found = False

        # 查找钱包
        for w_name, hotkeys, source in wallets:
            if w_name == last_wallet_name and last_hotkey_name in hotkeys:
                wallet_found = True
                if source == "bittensor":
                    wallet_dir = str(get_bittensor_wallet_dir())
                else:
                    wallet_dir = str(get_wallet_dir())
                break

        if not wallet_found:
            print(f"Last wallet {last_wallet_name}/{last_hotkey_name} not found")
            return False

        if not wallet_dir:
            wallet_dir = str(get_wallet_dir())

        try:
            # 加载钱包
            self.wallet = Wallet(name=last_wallet_name, hotkey=last_hotkey_name, path=wallet_dir)
            hotkey = self.wallet.get_hotkey()
            address = hotkey.ss58_address

            # 加载 coldkey 地址
            self.coldkey_address = get_coldkey_address()

            # 确定显示地址
            if self.coldkey_address:
                short_address = (
                    f"{self.coldkey_address[:6]}...{self.coldkey_address[-4:]}"
                    if self.coldkey_address and len(self.coldkey_address) > 10
                    else self.coldkey_address
                )
            else:
                short_address = f"{address[:6]}...{address[-4:]}" if address else "Unknown"

            # 发送信号
            self.wallet_loaded.emit(self.wallet, last_wallet_name, short_address)
            self.wallet_status_updated.emit(last_wallet_name)

            print(f"Auto-loaded wallet: {last_wallet_name}/{last_hotkey_name}")
            return True
        except Exception as e:
            print(f"Failed to auto-load wallet: {e}")
            return False

    def save_coldkey_address(self, address: str):
        """
        保存 coldkey 地址
        
        Args:
            address: Coldkey 地址
        """
        self.coldkey_address = address
        save_coldkey_address(address)
        print(f"Coldkey address saved: {address}")

        # 发送信号
        self.coldkey_submitted.emit(address)

        # 更新钱包显示
        short_address = (
            f"{address[:6]}...{address[-4:]}"
            if address and len(address) > 10
            else address
        )
        if self.wallet:
            self.wallet_loaded.emit(self.wallet, self.wallet.name, short_address)

    def get_wallet(self) -> Optional[Wallet]:
        """获取当前钱包对象"""
        return self.wallet

    def get_coldkey_address(self) -> Optional[str]:
        """获取 coldkey 地址"""
        return self.coldkey_address

    def needs_coldkey_prompt(self, use_existing_coldkey: bool, coldkey_address: str) -> bool:
        """
        判断是否需要提示输入 coldkey 地址
        
        Args:
            use_existing_coldkey: 是否使用现有 coldkey
            coldkey_address: Coldkey 地址
            
        Returns:
            是否需要提示
        """
        return not use_existing_coldkey or not coldkey_address

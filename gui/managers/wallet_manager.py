"""Wallet Manager - handles wallet loading, import and coldkey management"""

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
    """Manages wallet loading, import and coldkey address"""

    # Signals
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
        Load wallet
        
        Args:
            wallet_name: Wallet name
            hotkey_name: Hotkey name
            use_existing_coldkey: Whether to use existing coldkey
            coldkey_address: Coldkey address
            
        Returns:
            (success, error_message)
        """
        # Find wallet directory
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

        # Create wallet object
        self.wallet = Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_dir)

        # Get hotkey address
        try:
            hotkey = self.wallet.get_hotkey()
            address = hotkey.ss58_address
            short_address = f"{address[:6]}...{address[-4:]}" if address else "Unknown"
        except Exception as e:
            print(f"Error loading hotkey: {e}")
            return False, f"Error loading hotkey: {e}"

        # Handle coldkey address
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

        # Emit signals
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
        Import hotkey
        
        Args:
            hotkey_name: Hotkey name
            mnemonic: Mnemonic phrase
            coldkey_address: Coldkey address
            
        Returns:
            (success, error_message)
        """
        wallet_name = f"imported_{str(uuid.uuid4())[:8]}"
        wallet_dir = str(get_wallet_dir())

        try:
            # Create wallet and import hotkey
            self.wallet = Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_dir)
            self.wallet.import_hotkey_from_mnemonic(mnemonic, overwrite=True)

            # Save coldkey address
            self.coldkey_address = coldkey_address if coldkey_address else None
            save_wallet_settings(wallet_name, hotkey_name, coldkey_address)

            # Get hotkey address
            hotkey = self.wallet.get_hotkey()
            address = hotkey.ss58_address
            short_address = f"{address[:6]}...{address[-4:]}" if address else "Unknown"

            # Emit signals
            self.hotkey_imported.emit(self.wallet, wallet_name, short_address)
            self.wallet_status_updated.emit(wallet_name)

            return True, None
        except Exception as e:
            error_msg = f"Failed to import hotkey. Please try again.\n\nError: {str(e)}"
            print(f"Error importing hotkey: {e}")
            return False, error_msg

    def auto_load_wallet(self) -> bool:
        """
        Try to auto-load the last used wallet
        
        Returns:
            Whether successfully loaded
        """
        last_wallet_name, last_hotkey_name = get_last_wallet()

        if not last_wallet_name or not last_hotkey_name:
            return False

        wallets = discover_wallets()
        wallet_dir = None
        wallet_found = False

        # Find wallet
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
            # Load wallet
            self.wallet = Wallet(name=last_wallet_name, hotkey=last_hotkey_name, path=wallet_dir)
            hotkey = self.wallet.get_hotkey()
            address = hotkey.ss58_address

            # Load coldkey address
            self.coldkey_address = get_coldkey_address()

            # Determine display address
            if self.coldkey_address:
                short_address = (
                    f"{self.coldkey_address[:6]}...{self.coldkey_address[-4:]}"
                    if self.coldkey_address and len(self.coldkey_address) > 10
                    else self.coldkey_address
                )
            else:
                short_address = f"{address[:6]}...{address[-4:]}" if address else "Unknown"

            # Emit signals
            self.wallet_loaded.emit(self.wallet, last_wallet_name, short_address)
            self.wallet_status_updated.emit(last_wallet_name)

            print(f"Auto-loaded wallet: {last_wallet_name}/{last_hotkey_name}")
            return True
        except Exception as e:
            print(f"Failed to auto-load wallet: {e}")
            return False

    def save_coldkey_address(self, address: str):
        """
        Save coldkey address
        
        Args:
            address: Coldkey address
        """
        self.coldkey_address = address
        save_coldkey_address(address)
        print(f"Coldkey address saved: {address}")

        # Emit signals
        self.coldkey_submitted.emit(address)

        # Update wallet display
        short_address = (
            f"{address[:6]}...{address[-4:]}"
            if address and len(address) > 10
            else address
        )
        if self.wallet:
            self.wallet_loaded.emit(self.wallet, self.wallet.name, short_address)

    def get_wallet(self) -> Optional[Wallet]:
        """Get current wallet object"""
        return self.wallet

    def get_coldkey_address(self) -> Optional[str]:
        """Get coldkey address"""
        return self.coldkey_address

    def needs_coldkey_prompt(self, use_existing_coldkey: bool, coldkey_address: str) -> bool:
        """
        Check if coldkey address prompt is needed
        
        Args:
            use_existing_coldkey: Whether using existing coldkey
            coldkey_address: Coldkey address
            
        Returns:
            Whether prompt is needed
        """
        return not use_existing_coldkey or not coldkey_address

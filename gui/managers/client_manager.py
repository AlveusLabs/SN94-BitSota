"""Client Manager - handles BittensorDirectClient initialization and SOTA fetching"""

from typing import Optional
from PySide6.QtCore import QObject, Signal

from bittensor_network.wallet import Wallet
from miner.client import BittensorDirectClient
from gui.app_config import get_app_config


class ClientManager(QObject):
    """Manages Bittensor client initialization and configuration"""

    # Signals
    client_initialized = Signal(object)  # client
    sota_fetched = Signal(float)  # sota_threshold

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client: Optional[BittensorDirectClient] = None
        self.contract_manager = None

    def initialize_client(self, wallet: Wallet) -> bool:
        """
        Initialize Bittensor client
        
        Args:
            wallet: Wallet object
            
        Returns:
            Whether successfully initialized
        """
        if not wallet:
            print("Cannot initialize client: wallet is None")
            return False

        try:
            relay_endpoint = self.get_relay_endpoint()
            cfg = get_app_config()
            self.client = BittensorDirectClient(
                wallet=wallet,
                relay_endpoint=relay_endpoint,
                verbose=True,
                contract_manager=self.contract_manager,
                miner_task_count=cfg.miner_task_count,
            )
            print(f"Direct client created successfully with relay: {relay_endpoint}")
            
            # Emit signal
            self.client_initialized.emit(self.client)
            return True
        except Exception as e:
            print(f"Failed to create direct client: {e}")
            self.client = None
            return False

    @staticmethod
    def get_relay_endpoint() -> str:
        """
        Get relay endpoint from configuration
        
        Returns:
            relay endpoint URL
        """
        return get_app_config().relay_endpoint

    def fetch_current_sota(self) -> Optional[float]:
        """
        Fetch current SOTA threshold from relay
        
        Returns:
            SOTA threshold, or None if fetch fails
        """
        try:
            relay_endpoint = self.get_relay_endpoint()
            import requests
            response = requests.get(f"{relay_endpoint}/sota_threshold", timeout=10)
            response.raise_for_status()
            result = response.json()
            sota_threshold = result.get("sota_threshold")
            
            if sota_threshold is not None:
                # Emit signal
                self.sota_fetched.emit(sota_threshold)
            
            return sota_threshold
        except Exception as e:
            print(f"Failed to fetch SOTA from relay: {e}")
            return None

    def get_client(self) -> Optional[BittensorDirectClient]:
        """Get current client object"""
        return self.client

    def is_initialized(self) -> bool:
        """Check if client is initialized"""
        return self.client is not None

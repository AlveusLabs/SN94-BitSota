"""客户端管理器 - 负责 BittensorDirectClient 初始化和 SOTA 获取"""

from typing import Optional
from PySide6.QtCore import QObject, Signal

from bittensor_network.wallet import Wallet
from miner.client import BittensorDirectClient
from gui.app_config import get_app_config


class ClientManager(QObject):
    """管理 Bittensor 客户端的初始化和配置"""

    # 信号
    client_initialized = Signal(object)  # client
    sota_fetched = Signal(float)  # sota_threshold

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client: Optional[BittensorDirectClient] = None
        self.contract_manager = None

    def initialize_client(self, wallet: Wallet) -> bool:
        """
        初始化 Bittensor 客户端
        
        Args:
            wallet: 钱包对象
            
        Returns:
            是否成功初始化
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
            
            # 发送信号
            self.client_initialized.emit(self.client)
            return True
        except Exception as e:
            print(f"Failed to create direct client: {e}")
            self.client = None
            return False

    @staticmethod
    def get_relay_endpoint() -> str:
        """
        从配置获取 relay endpoint
        
        Returns:
            relay endpoint URL
        """
        return get_app_config().relay_endpoint

    def fetch_current_sota(self) -> Optional[float]:
        """
        从 relay 获取当前 SOTA 阈值
        
        Returns:
            SOTA 阈值，如果获取失败则返回 None
        """
        try:
            relay_endpoint = self.get_relay_endpoint()
            import requests
            response = requests.get(f"{relay_endpoint}/sota_threshold", timeout=10)
            response.raise_for_status()
            result = response.json()
            sota_threshold = result.get("sota_threshold")
            
            if sota_threshold is not None:
                # 发送信号
                self.sota_fetched.emit(sota_threshold)
            
            return sota_threshold
        except Exception as e:
            print(f"Failed to fetch SOTA from relay: {e}")
            return None

    def get_client(self) -> Optional[BittensorDirectClient]:
        """获取当前客户端对象"""
        return self.client

    def is_initialized(self) -> bool:
        """检查客户端是否已初始化"""
        return self.client is not None

import requests
import logging
import time
import bittensor as bt
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class RelayClient:
    """
    Client for interacting with the BitSota Relay Server.
    """

    def __init__(self, relay_url: str, wallet: "bt.wallet"):
        if not relay_url:
            raise ValueError("Relay URL cannot be empty")
        self.relay_url = relay_url
        self.wallet = wallet

    def _get_auth_headers(self) -> Dict[str, str]:
        """Generates authentication headers for a request."""
        timestamp = str(int(time.time()))
        try:
            signature = self.wallet.hotkey.sign(timestamp).hex()
            return {
                "X-Key": self.wallet.hotkey.ss58_address,
                "X-Timestamp": timestamp,
                "X-Signature": signature,
                "Content-Type": "application/json",
            }
        except Exception as e:
            logger.error(f"Failed to sign message: {e}")
            return {}

    def get_all_results(
        self, limit: int = 100, task_id: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        Fetches all recent results from the relay server.
        """
        try:
            endpoint = f"{self.relay_url}/results"
            params: Dict[str, Any] = {"limit": limit}
            if task_id:
                params["task_id"] = task_id

            headers = self._get_auth_headers()
            if not headers:
                return None

            response = requests.get(
                endpoint, headers=headers, params=params, timeout=10
            )
            response.raise_for_status()
            results = response.json()
            logger.debug(
                f"Relay query successful. Status: {response.status_code}. "
                f"Solutions returned: {len(results) if results else 0}."
            )
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get results from relay: {e}")
            return None

    def verify_result(self, result_id: str) -> bool:
        """
        Marks a result as verified on the relay server.
        """
        try:
            endpoint = f"{self.relay_url}/verify/{result_id}"
            headers = self._get_auth_headers()
            if not headers:
                return False
            response = requests.post(endpoint, headers=headers, timeout=10)
            response.raise_for_status()
            is_verified = response.json().get("status") == "verified"
            if is_verified:
                logger.debug(
                    f"Successfully verified result {result_id}. Status: {response.status_code}"
                )
            return is_verified
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to verify result {result_id} on relay: {e}")
            return False

    def blacklist_miner(self, miner_hotkey: str) -> bool:
        """
        Sends a request to blacklist a miner.
        """
        try:
            endpoint = f"{self.relay_url}/blacklist/{miner_hotkey}"
            headers = self._get_auth_headers()
            if not headers:
                return False
            response = requests.post(endpoint, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Blacklist vote for miner {miner_hotkey[:8]} recorded.")
            return response.json().get("status") == "blacklist vote recorded"
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send blacklist vote for miner {miner_hotkey}: {e}")
            return False

    def get_sota_threshold(self) -> Optional[float]:
        """
        Fetches the current SOTA threshold from the relay server.
        """
        try:
            endpoint = f"{self.relay_url}/sota_threshold"
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"SOTA threshold from relay: {data.get('sota_threshold')}")
            return data.get("sota_threshold")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to get SOTA from relay: {e}")
            return None

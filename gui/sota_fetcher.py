import time
import threading
from typing import Optional
from web3 import Web3


class ReadOnlySOTAFetcher:
    """Lightweight read-only contract interface for fetching SOTA threshold without EVM wallet."""

    def __init__(self, rpc_url: str, contract_address: str, abi: list, cache_duration: int = 60):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=abi
        )
        self.sota_cache_duration = cache_duration
        self._sota_cache = {
            "value": None,
            "timestamp": 0,
            "lock": threading.Lock(),
            "fetch_in_progress": False,
            "last_error": None
        }

    def get_current_sota_threshold(self, force_refresh: bool = False) -> float:
        """Cached latestScore (scaled 1e18 → float)."""
        with self._sota_cache["lock"]:
            current_time = time.time()
            if (
                not force_refresh
                and self._sota_cache["value"] is not None
                and current_time - self._sota_cache["timestamp"] < self.sota_cache_duration
            ):
                return self._sota_cache["value"]

            if self._sota_cache["fetch_in_progress"]:
                time.sleep(0.1)
                if self._sota_cache["value"] is not None:
                    return self._sota_cache["value"]
                raise Exception("SOTA fetch in progress, no cached value available")

            self._sota_cache["fetch_in_progress"] = True

        try:
            scaled_value = self.contract.functions.latestScore().call()
            new_value = scaled_value / 10**18
            with self._sota_cache["lock"]:
                self._sota_cache["value"] = new_value
                self._sota_cache["timestamp"] = current_time
                self._sota_cache["last_error"] = None
                self._sota_cache["fetch_in_progress"] = False
            return new_value
        except Exception as e:
            with self._sota_cache["lock"]:
                self._sota_cache["last_error"] = str(e)
                self._sota_cache["fetch_in_progress"] = False
                if self._sota_cache["value"] is not None:
                    return self._sota_cache["value"]
            raise e

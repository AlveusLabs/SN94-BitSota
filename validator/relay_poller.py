import threading
import time
import logging
from typing import Callable, List, Dict, Any

from validator.relay_client import RelayClient

logger = logging.getLogger(__name__)


class RelayPoller:
    """
    Manages background polling of the relay server for new results.
    """

    def __init__(
        self,
        relay_client: RelayClient,
        interval: int,
        on_new_results: Callable[[List[Dict[str, Any]]], None],
    ):
        self.relay_client = relay_client
        self.interval = interval
        self.on_new_results = on_new_results
        self.is_running = False
        self.background_thread = None
        self.lock = threading.Lock()
        self.last_error = None
        self.consecutive_failures = 0

    def start(self):
        """Starts the background polling thread."""
        with self.lock:
            if self.is_running:
                logger.info("Relay poller is already running.")
                return
            self.is_running = True
            self.background_thread = threading.Thread(
                target=self._poll_loop, daemon=True
            )
            self.background_thread.start()
            logger.info("Relay poller started.")

    def stop(self):
        """Stops the background polling thread."""
        with self.lock:
            self.is_running = False
        if self.background_thread:
            self.background_thread.join()
        logger.info("Relay poller stopped.")

    def _poll_loop(self):
        """The main loop for polling the relay server."""
        while self.is_running:
            try:
                results = self.relay_client.get_all_results()
                if results:
                    self.on_new_results(results)
                self.consecutive_failures = 0
            except Exception as e:
                self.consecutive_failures += 1
                self.last_error = str(e)
                logger.error(f"Relay poll failed ({self.consecutive_failures}x): {e}")
                # Exponential backoff
                sleep_time = min(self.interval * (2**self.consecutive_failures), 3600)
                time.sleep(sleep_time)
                continue

            time.sleep(self.interval)

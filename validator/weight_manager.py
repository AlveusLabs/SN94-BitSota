import threading
import time
import logging
import sys
from datetime import datetime

logger = logging.getLogger(__name__)


class WeightManager:
    def __init__(self, bittensor_network):
        self.network = bittensor_network
        self.lock = threading.Lock()
        self.last_weight_check = 0
        self.check_interval = 300
        self.background_thread = None
        self.is_running = False
        self.last_error = None
        self.consecutive_failures = 0

    def start_background_worker(self):
        """Start the background weight checking thread"""
        with self.lock:
            if self.is_running:
                logger.info("Weight checker already running")
                return

            self.is_running = True
            self.background_thread = threading.Thread(
                target=self._background_weight_loop, daemon=True
            )
            self.background_thread.start()
            logger.info("Started weight checking background thread")

    def _background_weight_loop(self):
        """Main background loop for weight checking"""
        while self.is_running:
            try:
                self._check_and_set_weights()
                self.consecutive_failures = 0
            except Exception as e:
                self.consecutive_failures += 1
                self.last_error = str(e)
                logger.error(f"Weight check failed ({self.consecutive_failures}x): {e}")

                # Exponential backoff on failures
                sleep_time = min(
                    300 * (2**self.consecutive_failures), 3600
                )  # max 1 hour
                time.sleep(sleep_time)
                continue

            time.sleep(self.check_interval)

    def _check_and_set_weights(self):
        """Actually check and set weights"""
        with self.lock:
            self.last_weight_check = time.time()

        # Do the actual work outside the lock to avoid blocking
        if self.network.should_set_weights():
            logger.info("Setting weights...")
            contract_bots = self.network.discover_contract_bots()
            if contract_bots:
                self.network.set_weights(contract_bots)
                logger.info(f"Weights set for {len(contract_bots)} contract bots")
            else:
                logger.warning("No contract bots found to weight")

    def trigger_immediate_check(self):
        """Trigger an immediate weight check if needed"""
        with self.lock:
            # Check if background thread is healthy
            if not self.is_running or not self.background_thread.is_alive():
                logger.warning("Weight checker thread died, restarting...")
                self.is_running = False
                self.start_background_worker()
                return

            # If last check was recent, skip
            if time.time() - self.last_weight_check < 60:  # 1 minute cooldown
                return

        # Do immediate check in current thread if it's been a while
        try:
            self._check_and_set_weights()
        except Exception as e:
            logger.error(f"Immediate weight check failed: {e}")

    def get_miner_score(self, miner_hotkey: str) -> float | None:
        """Get the current score for a specific miner."""
        try:
            # The metagraph is the source of truth for scores
            metagraph = self.network.get_metagraph()
            if metagraph is None:
                logger.warning("Metagraph not available, cannot get miner score.")
                return None

            if miner_hotkey in metagraph.hotkeys:
                index = metagraph.hotkeys.index(miner_hotkey)
                score = metagraph.weights[index].item()
                logger.info(f"Score for miner {miner_hotkey[:8]} is {score}")
                return score
            else:
                # Miner not in metagraph, maybe a new miner, score is 0
                logger.warning(
                    f"Miner {miner_hotkey[:8]} not found in metagraph. Returning score of 0."
                )
                return 0.0
        except Exception as e:
            logger.error(f"Error getting score for miner {miner_hotkey}: {e}")
            return None

    def get_status(self):
        """Get health status of weight manager"""
        with self.lock:
            return {
                "is_running": self.is_running,
                "thread_alive": (
                    self.background_thread.is_alive()
                    if self.background_thread
                    else False
                ),
                "last_check": (
                    datetime.fromtimestamp(self.last_weight_check).isoformat()
                    if self.last_weight_check
                    else None
                ),
                "last_error": self.last_error,
                "consecutive_failures": self.consecutive_failures,
            }

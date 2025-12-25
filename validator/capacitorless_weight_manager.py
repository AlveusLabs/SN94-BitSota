import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional

from validator.sota_schedule import SOTAEvent, active_event

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from validator.relay_client import RelayClient


class CapacitorlessWeightManager:
    """
    Weight manager for capacitorless mode.

    - Default: point all weight to a configured burn hotkey.
    - When a SOTA event is accepted: point full weight to the winning miner during
      [start_block, end_block), cutting off previous rewards at the next event's start.
    - Applies at most once per `alignment_mod` interval, without requiring the loop
      to run on the exact boundary block.
    """

    def __init__(
        self,
        bittensor_network,
        relay_client: "RelayClient",
        burn_hotkey: str,
        alignment_mod: int,
        *,
        events_limit: int = 50,
        event_refresh_interval_s: int = 60,
        metagraph_refresh_interval_s: int = 600,
        poll_interval_s: float = 6.0,
        retry_interval_s: float = 5.0,
        event_fetch_failure_grace_s: float = 120.0,
    ):
        self.network = bittensor_network
        self.relay_client = relay_client
        self.burn_hotkey = burn_hotkey
        self.alignment_mod = int(alignment_mod)

        self.events_limit = int(events_limit)
        self.event_refresh_interval_s = int(event_refresh_interval_s)
        self.metagraph_refresh_interval_s = int(metagraph_refresh_interval_s)
        self.poll_interval_s = float(poll_interval_s)
        self.retry_interval_s = max(0.0, float(retry_interval_s))
        self.event_fetch_failure_grace_s = max(
            0.0, float(event_fetch_failure_grace_s)
        )

        self.lock = threading.Lock()
        self.background_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0

        self._cached_events: List[SOTAEvent] = []
        self._last_event_fetch_ts = 0.0
        self._last_metagraph_refresh_ts = 0.0
        self._last_interval: Optional[int] = None
        self._last_interval_success: bool = False
        self._last_attempt_ts = 0.0
        self._last_applied_target: Optional[str] = None
        self._last_good_events: List[SOTAEvent] = []
        self._last_good_events_ts = 0.0

    def start_background_worker(self):
        with self.lock:
            if self.is_running:
                logger.info("CapacitorlessWeightManager already running")
                return
            self.is_running = True
            self.background_thread = threading.Thread(
                target=self._loop, daemon=True
            )
            self.background_thread.start()
            logger.info("Started capacitorless weight background thread")

    def stop(self):
        with self.lock:
            self.is_running = False
        if self.background_thread:
            self.background_thread.join(timeout=10)

    def _loop(self):
        while True:
            with self.lock:
                if not self.is_running:
                    return

            try:
                self._tick()
                self.consecutive_failures = 0
            except Exception as e:
                self.consecutive_failures += 1
                self.last_error = str(e)
                logger.error(
                    f"Capacitorless weight loop failed ({self.consecutive_failures}x): {e}"
                )
                sleep_time = min(5 * (2**self.consecutive_failures), 300)
                logger.info(
                    "Capacitorless weight loop sleeping for %.1fs after failure",
                    sleep_time,
                )
                time.sleep(sleep_time)

            logger.info(
                "Capacitorless weight loop sleeping for %.1fs",
                self.poll_interval_s,
            )
            time.sleep(self.poll_interval_s)

    def _tick(self):
        now = time.time()

        if now - self._last_metagraph_refresh_ts > self.metagraph_refresh_interval_s:
            try:
                self.network.resync_metagraph(lite=True)
            except Exception as e:
                logger.warning(f"Metagraph refresh failed: {e}")
            self._last_metagraph_refresh_ts = now

        with getattr(self.network, "subtensor_lock", threading.Lock()):
            current_block = int(self.network.subtensor.get_current_block())

        current_interval = int(current_block // self.alignment_mod)
        if current_interval < 0:
            return

        # If we already successfully applied weights for this interval, do nothing.
        # This avoids needing to wake up on the exact boundary block.
        if self._last_interval == current_interval and self._last_interval_success:
            return

        if self.retry_interval_s and (now - self._last_attempt_ts) < self.retry_interval_s:
            return

        # Refresh events right before applying weights to avoid missing short T1 windows.
        fetched = self._fetch_events()
        if fetched is not None:
            self._cached_events = fetched
            self._last_event_fetch_ts = now
            self._last_good_events = fetched
            self._last_good_events_ts = now
        else:
            # If the relay fetch fails transiently, don't flip weights aggressively.
            # Best-effort: fall back to the last good events for a short grace window,
            # else no-op and try again later.
            if (
                self._last_good_events
                and self.event_fetch_failure_grace_s
                and (now - self._last_good_events_ts) <= self.event_fetch_failure_grace_s
            ):
                self._cached_events = list(self._last_good_events)
            else:
                self._last_attempt_ts = now
                return

        current_event = active_event(self._cached_events, current_block)
        desired = current_event.miner_hotkey if current_event else self.burn_hotkey
        should_gate = hasattr(self.network, "should_set_weights") and not self.network.should_set_weights()
        # Bypass the gate if we are in an active reward window, or we need to transition
        # to a new target (e.g., winner→burn at window end).
        if should_gate and current_event is None and self._last_applied_target == desired:
            logger.debug("Skipping weight set (rate-limited / epoch not reached yet)")
            self._last_attempt_ts = now
            return
        if should_gate and current_event is not None:
            logger.debug("Bypassing weight gate for active reward window")

        success = self._set_single_target_weight(desired)
        self._last_attempt_ts = now
        if success:
            self._last_interval = current_interval
            self._last_interval_success = True
            self._last_applied_target = desired
        else:
            self._last_interval_success = False

    def _fetch_events(self) -> Optional[List[SOTAEvent]]:
        if not self.relay_client:
            return []
        logger.info(
            "Fetching relay SOTA events (limit=%s) for windowed mode",
            self.events_limit,
        )
        events = self.relay_client.get_sota_events(limit=self.events_limit)
        if events is None:
            return None
        if not events:
            return []

        parsed: List[SOTAEvent] = []
        for e in events:
            try:
                parsed.append(
                    SOTAEvent(
                        event_id=int(e["id"]),
                        miner_hotkey=str(e["miner_hotkey"]),
                        start_block=int(e["start_block"]),
                        end_block=int(e["end_block"]),
                    )
                )
            except Exception:
                continue
        return parsed

    def _set_single_target_weight(self, target_hotkey: str) -> bool:
        metagraph = self.network.metagraph
        if not metagraph or not getattr(metagraph, "hotkeys", None):
            raise RuntimeError("Metagraph not available for setting weights")

        if target_hotkey not in metagraph.hotkeys:
            # Try one forced refresh (helps when a new miner just registered).
            try:
                self.network.resync_metagraph(lite=True)
                metagraph = self.network.metagraph
            except Exception:
                pass

        if target_hotkey not in metagraph.hotkeys:
            if target_hotkey != self.burn_hotkey and self.burn_hotkey in metagraph.hotkeys:
                logger.warning(
                    f"Target hotkey not in metagraph, falling back to burn: {target_hotkey[:8]}"
                )
                applied = self.burn_hotkey
            else:
                raise RuntimeError(
                    f"Target hotkey not found in metagraph: {target_hotkey}"
                )
        else:
            applied = target_hotkey

        ok = bool(self.network.set_weights({applied: 1.0}))
        if ok:
            logger.info(f"Set weights → {applied[:8]}")
        else:
            logger.warning(f"Set weights failed → {applied[:8]}")

        # Consider this interval "successful" only if we applied the desired target.
        # If we had to fall back (e.g., hotkey not yet in metagraph), keep retrying.
        if ok and applied != target_hotkey:
            return False
        return ok

    def get_status(self):
        with self.lock:
            return {
                "is_running": self.is_running,
                "thread_alive": (
                    self.background_thread.is_alive()
                    if self.background_thread
                    else False
                ),
                "last_error": self.last_error,
                "consecutive_failures": self.consecutive_failures,
                "alignment_mod": self.alignment_mod,
                "burn_hotkey": self.burn_hotkey,
                "cached_events": len(self._cached_events),
            }

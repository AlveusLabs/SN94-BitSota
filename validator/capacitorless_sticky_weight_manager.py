import logging
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from validator.relay_client import RelayClient


class CapacitorlessStickyBurnSplitWeightManager:
    """
    Weight manager for capacitorless "sticky best miner" mode.

    - Always assigns `burn_share` of weight to `burn_hotkey`.
    - Assigns `winner_share` of weight to the most recently accepted SOTA winner
      (latest relay SOTA event), if any exist.
    - Does not use SOTA time windows; weights change only when the accepted SOTA
      winner changes (i.e., validators finalize a new SOTA event).
    """

    def __init__(
        self,
        bittensor_network,
        relay_client: "RelayClient",
        burn_hotkey: str,
        *,
        burn_share: float = 0.9,
        winner_share: Optional[float] = None,
        winner_source: str = "relay",
        min_winner_improvement: float = 0.0,
        events_limit: int = 50,
        event_refresh_interval_s: int = 60,
        metagraph_refresh_interval_s: int = 600,
        poll_interval_s: float = 6.0,
        retry_interval_s: float = 5.0,
    ):
        self.network = bittensor_network
        self.relay_client = relay_client
        self.burn_hotkey = str(burn_hotkey)

        self.winner_source = str(winner_source).strip().lower()
        if self.winner_source not in {"relay", "local"}:
            raise ValueError("winner_source must be 'relay' or 'local'")
        self.min_winner_improvement = max(0.0, float(min_winner_improvement))

        burn_share_f = float(burn_share)
        winner_share_f = (
            float(winner_share) if winner_share is not None else 1.0 - burn_share_f
        )
        if burn_share_f < 0.0 or winner_share_f < 0.0:
            raise ValueError("burn_share and winner_share must be >= 0")
        total = burn_share_f + winner_share_f
        if total <= 0.0:
            raise ValueError("burn_share + winner_share must be > 0")
        self.burn_share = burn_share_f / total
        self.winner_share = winner_share_f / total

        self.events_limit = int(events_limit)
        self.event_refresh_interval_s = int(event_refresh_interval_s)
        self.metagraph_refresh_interval_s = int(metagraph_refresh_interval_s)
        self.poll_interval_s = float(poll_interval_s)
        self.retry_interval_s = max(0.0, float(retry_interval_s))

        self.lock = threading.Lock()
        self.background_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0

        self._cached_events: List[dict] = []
        self._last_event_fetch_ts = 0.0
        self._last_metagraph_refresh_ts = 0.0
        self._last_attempt_ts = 0.0
        self._best_event: Optional[Tuple[int, str]] = None

        self._local_best: Optional[Tuple[float, str]] = None
        self._local_generation: int = 0

        self._last_applied_signature: Optional[Tuple[str, int, Optional[str]]] = None
        self._last_apply_success: bool = False

        # Back-compat status fields (relay mode semantics).
        self._last_applied_event_id: Optional[int] = None
        self._last_applied_winner: Optional[str] = None

    def start_background_worker(self):
        with self.lock:
            if self.is_running:
                logger.info(
                    "CapacitorlessStickyBurnSplitWeightManager already running"
                )
                return
            self.is_running = True
            self.background_thread = threading.Thread(
                target=self._loop, daemon=True
            )
            self.background_thread.start()
            logger.info(
                "Started capacitorless sticky burn-split weight background thread"
            )

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
                    "Capacitorless sticky burn-split loop failed (%sx): %s",
                    self.consecutive_failures,
                    e,
                )
                sleep_time = min(5 * (2**self.consecutive_failures), 300)
                # logger.info(
                #     "Capacitorless sticky burn-split loop sleeping for %.1fs after failure",
                #     sleep_time,
                # )
                time.sleep(sleep_time)

            # logger.info(
            #     "Capacitorless sticky burn-split loop sleeping for %.1fs",
            #     self.poll_interval_s,
            # )
            time.sleep(self.poll_interval_s)

    def update_local_winner(self, miner_hotkey: str, score: float) -> bool:
        """
        Provide a locally-evaluated winner candidate (capacitorless mode).

        This allows validators to set weights without relying on relay finalization.
        """
        miner_hotkey_s = str(miner_hotkey).strip()
        if not miner_hotkey_s:
            return False
        try:
            score_f = float(score)
        except Exception:
            return False

        with self.lock:
            if self._local_best is not None:
                current_score, current_hotkey = self._local_best
                if score_f <= current_score + self.min_winner_improvement:
                    return False
                if miner_hotkey_s == current_hotkey and score_f <= current_score:
                    return False
            self._local_best = (score_f, miner_hotkey_s)
            self._local_generation += 1
            # Allow immediate apply on next tick (even if retry_interval_s is set).
            self._last_attempt_ts = 0.0
        logger.info(
            "Updated local winner candidate → %s (score=%.4f)",
            miner_hotkey_s[:8],
            score_f,
        )
        return True

    def apply_once(self, *, force: bool = False) -> bool:
        """
        Best-effort: attempt to apply the latest desired weights immediately.

        Useful in `winner_source="local"` mode to apply weights right after evaluation,
        rather than waiting for the next poll interval.
        """
        before = self._last_apply_success
        if force:
            with self.lock:
                self._last_attempt_ts = 0.0
        try:
            self._tick()
        except Exception as e:
            logger.warning("Immediate weight apply failed: %s", e)
            return False
        return (not before) and self._last_apply_success

    def _get_local_winner(self) -> Optional[Tuple[int, str]]:
        with self.lock:
            if self._local_best is None:
                return None
            _, hk = self._local_best
            return self._local_generation, hk

    def _tick(self):
        now = time.time()

        if now - self._last_metagraph_refresh_ts > self.metagraph_refresh_interval_s:
            try:
                self.network.resync_metagraph(lite=True)
            except Exception as e:
                logger.warning("Metagraph refresh failed: %s", e)
            self._last_metagraph_refresh_ts = now

        relay_latest = (
            self._update_and_get_best_event(now)
            if self.winner_source == "relay"
            else None
        )

        if hasattr(self.network, "should_set_weights"):
            try:
                if not bool(self.network.should_set_weights()):
                    # Chain weight rate-limit not satisfied; don't attempt `set_weights()`
                    # since bittensor will reject it with "Perhaps it is too soon...".
                    self._last_attempt_ts = now
                    return
            except Exception:
                # Fall back to attempting (safer than freezing weights forever).
                pass

        local_latest = self._get_local_winner() if self.winner_source == "local" else None
        if self.winner_source == "local" and local_latest is None:
            self._apply_weights(winner_hotkey=self.burn_hotkey)
            return
        desired_event_id: Optional[int] = None
        if relay_latest is not None:
            desired_event_id = int(relay_latest[0])
            desired_signature = ("relay", desired_event_id, str(relay_latest[1]))
            desired_winner = relay_latest[1]
        elif local_latest is not None:
            desired_signature = ("local", int(local_latest[0]), str(local_latest[1]))
            desired_winner = local_latest[1]
        else:
            desired_signature = ("none", 0, None)
            desired_winner = None

        if desired_signature == self._last_applied_signature and self._last_apply_success:
            self._apply_weights(winner_hotkey=desired_winner)
            return

        if self.retry_interval_s and (now - self._last_attempt_ts) < self.retry_interval_s:
            return

        ok, applied_winner = self._apply_weights(desired_winner)
        self._last_attempt_ts = now
        self._last_applied_winner = desired_winner
        self._last_applied_event_id = desired_event_id
        self._last_applied_signature = desired_signature
        self._last_apply_success = ok and (applied_winner == desired_winner)

    def _update_and_get_best_event(self, now: float) -> Optional[Tuple[int, str]]:
        if self.relay_client is None:
            return self._best_event

        if self.event_refresh_interval_s <= 0 or (
            now - self._last_event_fetch_ts > self.event_refresh_interval_s
        ):
            logger.info(
                "Fetching relay SOTA events (limit=%s) for sticky burn-split",
                self.events_limit,
            )
            events = self.relay_client.get_sota_events(limit=self.events_limit)
            if events is None:
                # Preserve last known winner if relay fetch fails.
                return self._best_event
            fetched = list(events)
            if fetched:
                self._cached_events = fetched
            self._last_event_fetch_ts = now

        for e in self._cached_events:
            try:
                event_id = int(e["id"])
                miner_hotkey = str(e["miner_hotkey"])
            except Exception:
                continue
            if self._best_event is None or event_id > self._best_event[0]:
                self._best_event = (event_id, miner_hotkey)
        return self._best_event

    def _apply_weights(self, winner_hotkey: Optional[str]) -> Tuple[bool, Optional[str]]:
        metagraph = self.network.metagraph
        if not metagraph or not getattr(metagraph, "hotkeys", None):
            raise RuntimeError("Metagraph not available for setting weights")

        if self.burn_hotkey not in metagraph.hotkeys:
            try:
                self.network.resync_metagraph(lite=True)
                metagraph = self.network.metagraph
            except Exception:
                pass

        if self.burn_hotkey not in getattr(metagraph, "hotkeys", []):
            raise RuntimeError(
                f"Burn hotkey not found in metagraph: {self.burn_hotkey}"
            )

        desired_scores: Dict[str, float]
        if not winner_hotkey or winner_hotkey == self.burn_hotkey:
            desired_scores = {self.burn_hotkey: 1.0}
        else:
            desired_scores = {
                self.burn_hotkey: float(self.burn_share),
                str(winner_hotkey): float(self.winner_share),
            }

        applied_scores = dict(desired_scores)
        applied_winner = winner_hotkey if winner_hotkey else None

        if winner_hotkey and winner_hotkey != self.burn_hotkey:
            if winner_hotkey not in metagraph.hotkeys:
                # Try one refresh (new miner registration).
                try:
                    self.network.resync_metagraph(lite=True)
                    metagraph = self.network.metagraph
                except Exception:
                    pass

            if winner_hotkey not in getattr(metagraph, "hotkeys", []):
                logger.warning(
                    "Winner hotkey not in metagraph yet; falling back to burn only: %s",
                    str(winner_hotkey)[:8],
                )
                applied_scores = {self.burn_hotkey: 1.0}
                applied_winner = None

        ok = bool(self.network.set_weights(applied_scores))
        if ok:
            if applied_winner:
                logger.info(
                    "Set weights → burn %.3f / winner %.3f (%s)",
                    self.burn_share,
                    self.winner_share,
                    str(applied_winner)[:8],
                )
            else:
                logger.info("Set weights → burn only (%s)", self.burn_hotkey[:8])
        else:
            logger.warning("Set weights failed")
        return ok, applied_winner

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
                "burn_hotkey": self.burn_hotkey,
                "burn_share": self.burn_share,
                "winner_share": self.winner_share,
                "winner_source": self.winner_source,
                "min_winner_improvement": self.min_winner_improvement,
                "cached_events": len(self._cached_events),
                "last_applied_event_id": self._last_applied_event_id,
                "last_applied_winner": self._last_applied_winner,
                "last_apply_success": self._last_apply_success,
            }

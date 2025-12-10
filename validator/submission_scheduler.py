import logging
from datetime import datetime, time, timedelta
from typing import List, Optional


class SubmissionScheduler:
    """
    Optional gate that controls when validators are allowed to submit votes.
    Supports three modes:
      - immediate (default): never blocks submissions
      - interval: enforce a minimum time between submissions
      - utc_times: allow at most one submission per configured UTC time slot
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.mode = str(config.get("mode", "immediate")).lower()
        if self.mode not in {"immediate", "interval", "utc_times"}:
            logging.warning(
                "Unknown submission schedule mode '%s'. Falling back to immediate.",
                self.mode,
            )
            self.mode = "immediate"

        self.interval_seconds = max(0, int(config.get("interval_seconds", 0) or 0))
        self._utc_times: List[time] = self._parse_times(config.get("utc_times", []))
        self.last_submission: Optional[datetime] = None
        self.last_slot_submission: Optional[datetime] = None

    def _parse_times(self, raw_times) -> List[time]:
        parsed: List[time] = []
        if not isinstance(raw_times, list):
            return parsed

        for entry in raw_times:
            if isinstance(entry, str):
                try:
                    hours, minutes = entry.split(":")
                    parsed.append(time(int(hours), int(minutes)))
                except ValueError:
                    logging.warning(
                        "Invalid utc_times entry '%s'. Expected 'HH:MM'. Skipping.",
                        entry,
                    )
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                try:
                    parsed.append(time(int(entry[0]), int(entry[1])))
                except ValueError:
                    logging.warning(
                        "Invalid utc_times entry '%s'. Expected numeric hour/minute.",
                        entry,
                    )
        parsed.sort()
        return parsed

    def can_submit(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.utcnow()
        if self.mode == "interval":
            if self.interval_seconds <= 0:
                return True
            if not self.last_submission:
                return True
            elapsed = (now - self.last_submission).total_seconds()
            return elapsed >= self.interval_seconds

        if self.mode == "utc_times":
            slot = self._current_slot(now)
            if slot is None:
                return False
            if self.last_slot_submission and slot <= self.last_slot_submission:
                return False
            return True

        return True

    def record_submission(self, now: Optional[datetime] = None):
        now = now or datetime.utcnow()
        self.last_submission = now
        if self.mode == "utc_times":
            slot = self._current_slot(now)
            if slot:
                self.last_slot_submission = slot

    def get_next_allowed_time(self, now: Optional[datetime] = None) -> Optional[datetime]:
        now = now or datetime.utcnow()
        if self.mode == "interval":
            if not self.last_submission or self.interval_seconds <= 0:
                return now
            return self.last_submission + timedelta(seconds=self.interval_seconds)

        if self.mode == "utc_times":
            return self._next_slot(now)

        return now

    def _current_slot(self, now: datetime) -> Optional[datetime]:
        if not self._utc_times:
            return None
        today = now.date()
        for slot_time in reversed(self._utc_times):
            slot_dt = datetime.combine(today, slot_time)
            if now >= slot_dt:
                return slot_dt
        return None

    def _next_slot(self, now: datetime) -> Optional[datetime]:
        if not self._utc_times:
            return None
        today = now.date()
        for slot_time in self._utc_times:
            slot_dt = datetime.combine(today, slot_time)
            if slot_dt >= now:
                return slot_dt
        next_day = today + timedelta(days=1)
        return datetime.combine(next_day, self._utc_times[0])

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import requests

logger = logging.getLogger(__name__)

WeightKey = Union[int, str]


@dataclass(frozen=True)
class BackendWeightOverride:
    scores: Dict[WeightKey, float]
    signature: str
    description: str
    mode: str


def _snapshot_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/api/v1/reward-snapshot"):
        return base
    return f"{base}/api/v1/reward-snapshot"


def _normalize_targets(raw_targets: Any) -> Optional[Dict[WeightKey, float]]:
    if not isinstance(raw_targets, list):
        return None
    aggregated: Dict[WeightKey, float] = {}
    total = 0.0
    for raw in raw_targets:
        if not isinstance(raw, dict):
            continue
        try:
            weight = float(raw.get("weight", 0.0))
        except Exception:
            continue
        if weight <= 0.0:
            continue

        key: WeightKey | None = None
        if raw.get("uid") is not None:
            try:
                uid = int(raw.get("uid"))
            except Exception:
                continue
            if uid < 0:
                continue
            key = uid
        else:
            hotkey = str(raw.get("hotkey") or "").strip()
            if hotkey:
                key = hotkey
        if key is None:
            continue
        aggregated[key] = aggregated.get(key, 0.0) + weight
        total += weight
    if total <= 0.0:
        return None
    return {key: value / total for key, value in aggregated.items() if value > 0.0}


def _parse_backend_weight_override(raw_policy: Any) -> Optional[BackendWeightOverride]:
    if not isinstance(raw_policy, dict):
        return None
    raw_validator_weights = raw_policy.get("validator_weights")
    if not isinstance(raw_validator_weights, dict):
        return None

    mode = str(raw_validator_weights.get("mode") or "local").strip().lower() or "local"
    if mode == "local":
        return None
    if mode == "burn_uid0":
        scores: Dict[WeightKey, float] = {0: 1.0}
        return BackendWeightOverride(
            scores=scores,
            signature="burn_uid0",
            description="backend burn via UID 0",
            mode=mode,
        )
    if mode != "targets":
        logger.warning("Unknown backend validator weight mode: %s", mode)
        return None

    scores = _normalize_targets(raw_validator_weights.get("targets"))
    if not scores:
        logger.warning("Backend validator weight mode=targets had no valid targets")
        return None
    signature = json.dumps(
        [
            {"key": key, "weight": round(float(weight), 12)}
            for key, weight in sorted(scores.items(), key=lambda item: (str(type(item[0])), str(item[0])))
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return BackendWeightOverride(
        scores=scores,
        signature=signature,
        description="backend explicit targets",
        mode=mode,
    )


class BackendWeightPolicyClient:
    def __init__(
        self,
        *,
        base_url: str,
        refresh_interval_s: float = 30.0,
        timeout_s: float = 10.0,
    ):
        self.snapshot_url = _snapshot_url(base_url)
        self.refresh_interval_s = max(1.0, float(refresh_interval_s))
        self.timeout_s = max(1.0, float(timeout_s))
        self._cached_override: Optional[BackendWeightOverride] = None
        self._last_fetch_ts = 0.0
        self._last_error: Optional[str] = None

    def get_override(self) -> Optional[BackendWeightOverride]:
        now = time.time()
        if now - self._last_fetch_ts < self.refresh_interval_s:
            return self._cached_override

        self._last_fetch_ts = now
        if not self.snapshot_url:
            self._cached_override = None
            self._last_error = None
            return None

        try:
            response = requests.get(self.snapshot_url, timeout=self.timeout_s)
            response.raise_for_status()
            payload = response.json()
            reward_policy = payload.get("reward_policy") if isinstance(payload, dict) else None
            self._cached_override = _parse_backend_weight_override(reward_policy)
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning(
                "Failed to fetch backend validator weight policy from %s: %s",
                self.snapshot_url,
                exc,
            )
        return self._cached_override

    def get_status(self) -> Dict[str, Any]:
        override = self._cached_override
        return {
            "snapshot_url": self.snapshot_url,
            "refresh_interval_s": self.refresh_interval_s,
            "timeout_s": self.timeout_s,
            "mode": override.mode if override is not None else "local",
            "description": override.description if override is not None else "",
            "last_error": self._last_error,
        }

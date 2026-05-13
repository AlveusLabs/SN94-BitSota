from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import requests

logger = logging.getLogger(__name__)

WeightKey = Union[int, str]

SN94_NETUID = 94
SN94_CHAIN_ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"
SN94_MAINNET_CONTRACT = "5CUo48Vuwidb4pTogCCqAeYyMRUwNieTjeEL8FyYvwmQ9XA5"
SN94_CONTRACT_HOTKEY = "5F7MJ2fAyxBG7ci4xP7kQPJanoMdNurk1QBP1AQuFT2Jmzg2"
SN94_REQUIRED_CONTRACT_WEIGHT = 0.10
SN94_BURN_UID = 0
SN94_REQUIRED_BURN_REST_WEIGHT = 1.0 - SN94_REQUIRED_CONTRACT_WEIGHT
SN94_POLICY_TOLERANCE = 1e-9


class BackendWeightPolicyError(ValueError):
    """Raised when a backend validator weight policy is unsafe to apply."""


@dataclass(frozen=True)
class BackendWeightOverride:
    scores: Dict[WeightKey, float]
    signature: str
    description: str
    mode: str
    transition_status: str = ""
    contract_hotkey_weight: Optional[float] = None
    burn_uid_weight: Optional[float] = None


def _snapshot_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/api/v1/reward-snapshot"):
        return base
    return f"{base}/api/v1/reward-snapshot"


def _as_float(raw_value: Any, field: str) -> float:
    try:
        value = float(raw_value)
    except Exception as exc:
        raise BackendWeightPolicyError(f"{field} must be numeric") from exc
    return value


def _transition_status(raw_validator_weights: dict[str, Any]) -> str:
    raw_transition = raw_validator_weights.get("transition_policy")
    if not isinstance(raw_transition, dict):
        return ""
    return str(raw_transition.get("status") or "").strip().lower()


def _validate_sn94_transition_policy(raw_validator_weights: dict[str, Any]) -> None:
    raw_transition = raw_validator_weights.get("transition_policy")
    if not isinstance(raw_transition, dict):
        return

    required_hotkey = str(
        raw_transition.get("required_contract_hotkey") or SN94_CONTRACT_HOTKEY
    ).strip()
    if required_hotkey != SN94_CONTRACT_HOTKEY:
        raise BackendWeightPolicyError(
            "transition_policy.required_contract_hotkey does not match the SN94 contract hotkey"
        )

    required_weight = _as_float(
        raw_transition.get(
            "required_contract_weight", SN94_REQUIRED_CONTRACT_WEIGHT
        ),
        "transition_policy.required_contract_weight",
    )
    if abs(required_weight - SN94_REQUIRED_CONTRACT_WEIGHT) > SN94_POLICY_TOLERANCE:
        raise BackendWeightPolicyError(
            "transition_policy.required_contract_weight must be exactly 0.10 for SN94"
        )

    remaining_weight = _as_float(
        raw_transition.get("remaining_weight", SN94_REQUIRED_BURN_REST_WEIGHT),
        "transition_policy.remaining_weight",
    )
    if abs(remaining_weight - SN94_REQUIRED_BURN_REST_WEIGHT) > SN94_POLICY_TOLERANCE:
        raise BackendWeightPolicyError(
            "transition_policy.remaining_weight must be exactly 0.90 for SN94"
        )


def _normalize_targets(raw_targets: Any) -> Dict[WeightKey, float]:
    if not isinstance(raw_targets, list):
        raise BackendWeightPolicyError("validator_weights.targets must be a list")

    aggregated: Dict[WeightKey, float] = {}
    total = 0.0
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            raise BackendWeightPolicyError(
                f"validator_weights.targets[{index}] must be an object"
            )

        raw_hotkey = raw.get("hotkey")
        hotkey = str(raw_hotkey or "").strip()
        has_uid = raw.get("uid") is not None
        has_hotkey = bool(hotkey)
        if has_uid == has_hotkey:
            raise BackendWeightPolicyError(
                f"validator_weights.targets[{index}] must define exactly one of uid or hotkey"
            )

        weight = _as_float(raw.get("weight"), f"validator_weights.targets[{index}].weight")
        if weight <= 0.0:
            raise BackendWeightPolicyError(
                f"validator_weights.targets[{index}].weight must be > 0"
            )

        if has_uid:
            try:
                uid = int(raw.get("uid"))
            except Exception as exc:
                raise BackendWeightPolicyError(
                    f"validator_weights.targets[{index}].uid must be an integer"
                ) from exc
            if uid < 0:
                raise BackendWeightPolicyError(
                    f"validator_weights.targets[{index}].uid must be >= 0"
                )
            key: WeightKey = uid
        else:
            key = hotkey

        aggregated[key] = aggregated.get(key, 0.0) + weight
        total += weight

    if total <= 0.0:
        raise BackendWeightPolicyError(
            "validator_weights.targets must contain at least one positive target"
        )
    return {
        key: value / total
        for key, value in aggregated.items()
        if value > 0.0
    }


def _sn94_contract_and_burn_weights(
    scores: Dict[WeightKey, float],
    *,
    require_burn_rest: bool = True,
) -> tuple[float, float]:
    contract_weight = float(scores.get(SN94_CONTRACT_HOTKEY, 0.0))
    burn_weight = float(scores.get(SN94_BURN_UID, 0.0))

    if abs(contract_weight - SN94_REQUIRED_CONTRACT_WEIGHT) > SN94_POLICY_TOLERANCE:
        raise BackendWeightPolicyError(
            "SN94 targets must normalize to exactly 0.10 weight for the contract hotkey"
        )

    if require_burn_rest:
        if abs(burn_weight - SN94_REQUIRED_BURN_REST_WEIGHT) > SN94_POLICY_TOLERANCE:
            raise BackendWeightPolicyError(
                "SN94 targets must normalize remaining 0.90 weight to burn UID 0"
            )
        extra_targets = {
            key: weight
            for key, weight in scores.items()
            if key not in {SN94_BURN_UID, SN94_CONTRACT_HOTKEY}
            and abs(float(weight)) > SN94_POLICY_TOLERANCE
        }
        if extra_targets:
            raise BackendWeightPolicyError(
                "SN94 burn-rest targets may only include burn UID 0 and the contract hotkey"
            )

    return contract_weight, burn_weight


def _signature_for_scores(scores: Dict[WeightKey, float]) -> str:
    return json.dumps(
        [
            {"key": key, "weight": round(float(weight), 12)}
            for key, weight in sorted(
                scores.items(),
                key=lambda item: (str(type(item[0])), str(item[0])),
            )
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_backend_weight_override(
    raw_policy: Any,
    *,
    enforce_sn94_contract_targets: bool = False,
    sn94_require_burn_rest: bool = True,
) -> Optional[BackendWeightOverride]:
    if not isinstance(raw_policy, dict):
        raise BackendWeightPolicyError("reward_policy must be an object")
    raw_validator_weights = raw_policy.get("validator_weights")
    if not isinstance(raw_validator_weights, dict):
        raise BackendWeightPolicyError("reward_policy.validator_weights must be an object")

    mode = str(raw_validator_weights.get("mode") or "local").strip().lower() or "local"
    transition_status = _transition_status(raw_validator_weights)

    if enforce_sn94_contract_targets:
        _validate_sn94_transition_policy(raw_validator_weights)

    if transition_status == "active" and mode != "targets":
        raise BackendWeightPolicyError(
            "active SN94 validator weight transition requires mode='targets'"
        )

    if mode == "local":
        return None

    if mode == "burn_uid0":
        return BackendWeightOverride(
            scores={SN94_BURN_UID: 1.0},
            signature="burn_uid0",
            description="backend burn via UID 0",
            mode=mode,
            transition_status=transition_status,
            contract_hotkey_weight=0.0,
            burn_uid_weight=1.0,
        )

    if mode != "targets":
        raise BackendWeightPolicyError(f"unknown backend validator weight mode: {mode}")

    scores = _normalize_targets(raw_validator_weights.get("targets"))
    contract_weight: Optional[float] = None
    burn_weight: Optional[float] = None
    if enforce_sn94_contract_targets:
        contract_weight, burn_weight = _sn94_contract_and_burn_weights(
            scores,
            require_burn_rest=sn94_require_burn_rest,
        )

    return BackendWeightOverride(
        scores=scores,
        signature=_signature_for_scores(scores),
        description="backend explicit targets",
        mode=mode,
        transition_status=transition_status,
        contract_hotkey_weight=contract_weight,
        burn_uid_weight=burn_weight,
    )


def _parse_backend_weight_override(raw_policy: Any) -> Optional[BackendWeightOverride]:
    try:
        return parse_backend_weight_override(raw_policy)
    except BackendWeightPolicyError as exc:
        logger.warning("Ignoring unsafe backend validator weight policy: %s", exc)
        return None


class BackendWeightPolicyClient:
    def __init__(
        self,
        *,
        base_url: str,
        refresh_interval_s: float = 30.0,
        timeout_s: float = 10.0,
        enforce_sn94_contract_targets: bool = False,
        sn94_require_burn_rest: bool = True,
    ):
        self.snapshot_url = _snapshot_url(base_url)
        self.refresh_interval_s = max(1.0, float(refresh_interval_s))
        self.timeout_s = max(1.0, float(timeout_s))
        self.enforce_sn94_contract_targets = bool(enforce_sn94_contract_targets)
        self.sn94_require_burn_rest = bool(sn94_require_burn_rest)
        self._cached_override: Optional[BackendWeightOverride] = None
        self._last_fetch_ts = 0.0
        self._last_success_ts = 0.0
        self._last_error: Optional[str] = None
        self._last_validation_error: Optional[str] = None
        self._last_fallback_reason: Optional[str] = None
        self._last_backend_mode = "unknown"
        self._last_transition_status = ""

    def get_override(self) -> Optional[BackendWeightOverride]:
        now = time.time()
        if now - self._last_fetch_ts < self.refresh_interval_s:
            return self._cached_override

        self._last_fetch_ts = now
        if not self.snapshot_url:
            self._cached_override = None
            self._last_error = None
            self._last_validation_error = None
            self._last_fallback_reason = "backend_policy_url not configured"
            self._last_backend_mode = "local"
            self._last_transition_status = ""
            return None

        try:
            response = requests.get(self.snapshot_url, timeout=self.timeout_s)
            response.raise_for_status()
            payload = response.json()
            reward_policy = payload.get("reward_policy") if isinstance(payload, dict) else None
            raw_validator_weights = (
                reward_policy.get("validator_weights")
                if isinstance(reward_policy, dict)
                else None
            )
            self._last_backend_mode = (
                str((raw_validator_weights or {}).get("mode") or "local").strip().lower()
                if isinstance(raw_validator_weights, dict)
                else "unknown"
            )
            self._last_transition_status = (
                _transition_status(raw_validator_weights)
                if isinstance(raw_validator_weights, dict)
                else ""
            )
            override = parse_backend_weight_override(
                reward_policy,
                enforce_sn94_contract_targets=self.enforce_sn94_contract_targets,
                sn94_require_burn_rest=self.sn94_require_burn_rest,
            )
            self._cached_override = override
            self._last_success_ts = now
            self._last_error = None
            self._last_validation_error = None
            self._last_fallback_reason = "backend mode local" if override is None else None
        except BackendWeightPolicyError as exc:
            self._cached_override = None
            self._last_error = None
            self._last_validation_error = str(exc)
            self._last_fallback_reason = f"invalid backend policy: {exc}"
            logger.warning(
                "Ignoring backend validator weight policy from %s: %s",
                self.snapshot_url,
                exc,
            )
        except Exception as exc:
            self._cached_override = None
            self._last_error = str(exc)
            self._last_validation_error = None
            self._last_fallback_reason = f"backend fetch failed: {exc}"
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
            "backend_mode": self._last_backend_mode,
            "effective_mode": override.mode if override is not None else "local_fallback",
            "description": override.description if override is not None else "",
            "transition_status": (
                override.transition_status if override is not None else self._last_transition_status
            ),
            "last_error": self._last_error,
            "last_validation_error": self._last_validation_error,
            "fallback_reason": self._last_fallback_reason,
            "last_fetch_ts": self._last_fetch_ts,
            "last_success_ts": self._last_success_ts,
            "sn94_policy": {
                "enforce_contract_targets": self.enforce_sn94_contract_targets,
                "require_burn_rest": self.sn94_require_burn_rest,
                "netuid": SN94_NETUID,
                "chain_endpoint": SN94_CHAIN_ENDPOINT,
                "mainnet_contract": SN94_MAINNET_CONTRACT,
                "contract_hotkey": SN94_CONTRACT_HOTKEY,
                "required_contract_weight": SN94_REQUIRED_CONTRACT_WEIGHT,
                "burn_uid": SN94_BURN_UID,
                "required_burn_rest_weight": SN94_REQUIRED_BURN_REST_WEIGHT,
                "contract_hotkey_weight": (
                    override.contract_hotkey_weight if override is not None else None
                ),
                "burn_uid_weight": override.burn_uid_weight if override is not None else None,
            },
        }

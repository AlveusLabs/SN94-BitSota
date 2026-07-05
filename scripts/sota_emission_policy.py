from __future__ import annotations

from typing import Any


ONE_SOTA = 10**18
SOTA_DAILY_EMISSION_UNITS = 7_200 * ONE_SOTA
SOTA_DEFAULT_EPOCH_SECONDS = 86_400
SOTA_DEFAULT_TESTNET_LANE_WEIGHT_BPS = 10_000
SOTA_DEFAULT_RND_CAPACITOR_MAX_EPOCHS = 30
SOTA_DEFAULT_VALIDATOR_REWARD_SHARE_BPS = 1_000


def sota_epoch_budget_units(
    *,
    daily_emission_units: int = SOTA_DAILY_EMISSION_UNITS,
    epoch_seconds: int = SOTA_DEFAULT_EPOCH_SECONDS,
    lane_weight_bps: int = SOTA_DEFAULT_TESTNET_LANE_WEIGHT_BPS,
) -> int:
    return max(0, int(daily_emission_units)) * max(0, int(epoch_seconds)) * max(0, int(lane_weight_bps)) // (
        SOTA_DEFAULT_EPOCH_SECONDS * 10_000
    )


def frontier_capacitor_reward_policy(
    *,
    daily_emission_units: int = SOTA_DAILY_EMISSION_UNITS,
    epoch_seconds: int = SOTA_DEFAULT_EPOCH_SECONDS,
    lane_weight_bps: int = SOTA_DEFAULT_TESTNET_LANE_WEIGHT_BPS,
    max_rollover_epochs: int = SOTA_DEFAULT_RND_CAPACITOR_MAX_EPOCHS,
    validator_reward_share_bps: int = SOTA_DEFAULT_VALIDATOR_REWARD_SHARE_BPS,
) -> dict[str, Any]:
    return {
        "version": 1,
        "source": "accepted_submissions",
        "allocation": "equal_per_accepted_submission",
        "release_mode": "frontier_capacitor",
        "schedule": {
            "daily_emission_units": str(int(daily_emission_units)),
            "epoch_seconds": int(epoch_seconds),
            "lane_weight_bps": int(lane_weight_bps),
        },
        "capacitor": {
            "enabled": True,
            "release_condition": "self_validation_frontier_consensus",
            "rollover": "carry_forward_until_frontier_release",
            "max_rollover_epochs": int(max_rollover_epochs),
            "validator_reward_share_bps": int(validator_reward_share_bps),
        },
    }

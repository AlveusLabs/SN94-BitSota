from __future__ import annotations

from types import SimpleNamespace

import pytest

from validator.backend_weight_setter import (
    apply_backend_weight_policy,
    extract_validator_weight_policy,
    resolve_validator_weight_scores,
)


class _FakeNetwork:
    def __init__(self, hotkeys: list[str], *, ready: bool = True, set_ok: bool = True) -> None:
        self.metagraph = SimpleNamespace(hotkeys=hotkeys)
        self.ready = ready
        self.set_ok = set_ok
        self.resync_calls = 0
        self.set_calls: list[dict[str, float]] = []

    def resync_metagraph(self, *, lite: bool = True):
        self.resync_calls += 1
        return self.metagraph

    def should_set_weights(self) -> bool:
        return self.ready

    def set_weights(self, scores: dict[str, float]) -> bool:
        self.set_calls.append(dict(scores))
        return self.set_ok


def test_extract_validator_weight_policy_from_reward_snapshot() -> None:
    snapshot = {
        "reward_policy": {
            "validator_weights": {
                "mode": "targets",
                "targets": [{"uid": 0, "weight": 1.0}],
            }
        }
    }

    assert extract_validator_weight_policy(snapshot) == {
        "mode": "targets",
        "targets": [{"uid": 0, "weight": 1.0}],
    }


def test_resolve_burn_uid0_policy_to_uid_zero_hotkey() -> None:
    scores = resolve_validator_weight_scores(
        {"mode": "burn_uid0"},
        metagraph_hotkeys=["5Burn", "5Contract"],
    )

    assert scores == {"5Burn": 1.0}


def test_resolve_targets_normalizes_uid_and_hotkey_weights() -> None:
    scores = resolve_validator_weight_scores(
        {
            "mode": "targets",
            "targets": [
                {"uid": 0, "weight": 60},
                {"hotkey": "5Contract", "weight": 40},
            ],
        },
        metagraph_hotkeys=["5Burn", "5Contract"],
    )

    assert scores == {"5Burn": 0.6, "5Contract": 0.4}


def test_resolve_targets_rejects_unknown_hotkey() -> None:
    with pytest.raises(RuntimeError, match="not in metagraph"):
        resolve_validator_weight_scores(
            {
                "mode": "targets",
                "targets": [{"hotkey": "5Missing", "weight": 1.0}],
            },
            metagraph_hotkeys=["5Burn"],
        )


def test_apply_local_policy_does_not_set_weights() -> None:
    network = _FakeNetwork(["5Burn"])

    outcome = apply_backend_weight_policy(
        policy={"mode": "local"},
        network=network,
    )

    assert outcome.status == "skipped_local"
    assert network.resync_calls == 1
    assert network.set_calls == []


def test_apply_targets_respects_chain_rate_limit() -> None:
    network = _FakeNetwork(["5Burn", "5Contract"], ready=False)

    outcome = apply_backend_weight_policy(
        policy={
            "mode": "targets",
            "targets": [
                {"uid": 0, "weight": 0.7},
                {"hotkey": "5Contract", "weight": 0.3},
            ],
        },
        network=network,
    )

    assert outcome.status == "skipped_rate_limit"
    assert outcome.scores == {"5Burn": 0.7, "5Contract": 0.3}
    assert network.set_calls == []


def test_apply_targets_submits_normalized_scores() -> None:
    network = _FakeNetwork(["5Burn", "5Contract"])

    outcome = apply_backend_weight_policy(
        policy={
            "mode": "targets",
            "targets": [
                {"uid": 0, "weight": 80},
                {"hotkey": "5Contract", "weight": 20},
            ],
        },
        network=network,
    )

    assert outcome.status == "submitted"
    assert network.set_calls == [{"5Burn": 0.8, "5Contract": 0.2}]


def test_apply_targets_dry_run_does_not_submit() -> None:
    network = _FakeNetwork(["5Burn", "5Contract"])

    outcome = apply_backend_weight_policy(
        policy={
            "mode": "targets",
            "targets": [
                {"uid": 0, "weight": 1},
                {"hotkey": "5Contract", "weight": 1},
            ],
        },
        network=network,
        dry_run=True,
    )

    assert outcome.status == "dry_run"
    assert outcome.scores == {"5Burn": 0.5, "5Contract": 0.5}
    assert network.set_calls == []

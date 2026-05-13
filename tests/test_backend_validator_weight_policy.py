from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

import pytest

if "bittensor" not in sys.modules:
    sys.modules["bittensor"] = types.SimpleNamespace(
        utils=types.SimpleNamespace(
            weight_utils=types.SimpleNamespace(
                convert_weights_and_uids_for_emit=lambda *, uids, weights: (
                    uids.tolist(),
                    [float(value) for value in weights.tolist()],
                )
            )
        )
    )

from bittensor_network import _state as state
from bittensor_network import _weights
from validator.backend_weight_policy import (
    BackendWeightOverride,
    BackendWeightPolicyError,
    BackendWeightPolicyClient,
    SN94_CONTRACT_HOTKEY,
    parse_backend_weight_override,
)
from validator.capacitorless_sticky_weight_manager import (
    CapacitorlessStickyBurnSplitWeightManager,
)
from validator.capacitorless_weight_manager import CapacitorlessWeightManager


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self):
        return self._payload


class _FakeRelayClient:
    def get_sota_events(self, limit: int = 50):
        return []


class _FakeBackendPolicyClient:
    def __init__(self, override: BackendWeightOverride | None):
        self._override = override

    def get_override(self):
        return self._override

    def get_status(self):
        return {"mode": self._override.mode if self._override else "local"}


class _FakeNetwork:
    def __init__(self):
        self.metagraph = SimpleNamespace(hotkeys=["5A", "5B", "5C"], netuid=94)
        self.subtensor = SimpleNamespace(get_current_block=lambda: 720)
        self.subtensor_lock = threading.RLock()
        self.applied = []

    def should_set_weights(self) -> bool:
        return True

    def set_weights(self, scores):
        self.applied.append(dict(scores))
        return True

    def resync_metagraph(self, lite: bool = True) -> None:
        return None


def test_backend_weight_policy_client_parses_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        "validator.backend_weight_policy.requests.get",
        lambda *args, **kwargs: _DummyResponse(
            {
                "reward_policy": {
                    "validator_weights": {
                        "mode": "targets",
                        "targets": [
                            {"uid": 0, "weight": 0.75},
                            {"hotkey": "5TargetHotkey", "weight": 0.25},
                        ],
                    }
                }
            }
        ),
    )

    client = BackendWeightPolicyClient(
        base_url="https://coordinator.example",
        refresh_interval_s=1.0,
        timeout_s=5.0,
    )
    override = client.get_override()

    assert override is not None
    assert override.mode == "targets"
    assert override.scores == {0: 0.75, "5TargetHotkey": 0.25}


def test_backend_weight_policy_enforces_sn94_contract_burn_rest_targets(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "validator.backend_weight_policy.requests.get",
        lambda *args, **kwargs: _DummyResponse(
            {
                "reward_policy": {
                    "validator_weights": {
                        "mode": "targets",
                        "targets": [
                            {"uid": 0, "weight": 9},
                            {"hotkey": SN94_CONTRACT_HOTKEY, "weight": 1},
                        ],
                        "transition_policy": {"status": "active"},
                    }
                }
            }
        ),
    )

    client = BackendWeightPolicyClient(
        base_url="https://coordinator.example",
        refresh_interval_s=1.0,
        timeout_s=5.0,
        enforce_sn94_contract_targets=True,
    )
    override = client.get_override()

    assert override is not None
    assert override.mode == "targets"
    assert override.transition_status == "active"
    assert override.scores == pytest.approx({0: 0.9, SN94_CONTRACT_HOTKEY: 0.1})
    assert override.contract_hotkey_weight == pytest.approx(0.1)
    assert override.burn_uid_weight == pytest.approx(0.9)
    status = client.get_status()
    assert status["effective_mode"] == "targets"
    assert status["sn94_policy"]["contract_hotkey_weight"] == pytest.approx(0.1)
    assert status["sn94_policy"]["burn_uid_weight"] == pytest.approx(0.9)


def test_backend_weight_policy_rejects_sn94_targets_without_contract_hotkey(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "validator.backend_weight_policy.requests.get",
        lambda *args, **kwargs: _DummyResponse(
            {
                "reward_policy": {
                    "validator_weights": {
                        "mode": "targets",
                        "targets": [{"hotkey": "5TeamHotkey", "weight": 1.0}],
                        "transition_policy": {"status": "blocked"},
                    }
                }
            }
        ),
    )

    client = BackendWeightPolicyClient(
        base_url="https://coordinator.example",
        refresh_interval_s=1.0,
        timeout_s=5.0,
        enforce_sn94_contract_targets=True,
    )

    assert client.get_override() is None
    status = client.get_status()
    assert status["effective_mode"] == "local_fallback"
    assert status["backend_mode"] == "targets"
    assert "contract hotkey" in status["last_validation_error"]
    assert status["fallback_reason"].startswith("invalid backend policy:")


def test_backend_weight_policy_rejects_ambiguous_targets() -> None:
    with pytest.raises(BackendWeightPolicyError, match="exactly one of uid or hotkey"):
        parse_backend_weight_override(
            {
                "validator_weights": {
                    "mode": "targets",
                    "targets": [
                        {"uid": 0, "hotkey": SN94_CONTRACT_HOTKEY, "weight": 1.0}
                    ],
                }
            }
        )


def test_backend_weight_policy_rejects_active_non_target_transition() -> None:
    with pytest.raises(BackendWeightPolicyError, match="requires mode='targets'"):
        parse_backend_weight_override(
            {
                "validator_weights": {
                    "mode": "burn_uid0",
                    "targets": [],
                    "transition_policy": {"status": "active"},
                }
            },
            enforce_sn94_contract_targets=True,
        )


def test_backend_weight_policy_fetch_failure_falls_back_to_local(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_get(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _DummyResponse(
                {
                    "reward_policy": {
                        "validator_weights": {
                            "mode": "burn_uid0",
                            "targets": [],
                        }
                    }
                }
            )
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr("validator.backend_weight_policy.requests.get", _fake_get)

    client = BackendWeightPolicyClient(
        base_url="https://coordinator.example",
        refresh_interval_s=1.0,
        timeout_s=5.0,
    )

    assert client.get_override() is not None
    client._last_fetch_ts = 0.0
    assert client.get_override() is None
    status = client.get_status()
    assert status["effective_mode"] == "local_fallback"
    assert status["fallback_reason"].startswith("backend fetch failed:")


def test_sticky_weight_manager_prefers_backend_override_uid0() -> None:
    network = _FakeNetwork()
    manager = CapacitorlessStickyBurnSplitWeightManager(
        network,
        relay_client=_FakeRelayClient(),
        burn_hotkey="5A",
        winner_source="relay",
        backend_weight_policy_client=_FakeBackendPolicyClient(
            BackendWeightOverride(
                scores={0: 1.0},
                signature="burn_uid0",
                description="backend burn via UID 0",
                mode="burn_uid0",
            )
        ),
    )

    manager._tick()

    assert network.applied == [{0: 1.0}]


def test_windowed_weight_manager_prefers_backend_override_targets() -> None:
    network = _FakeNetwork()
    manager = CapacitorlessWeightManager(
        network,
        relay_client=_FakeRelayClient(),
        burn_hotkey="5A",
        alignment_mod=360,
        backend_weight_policy_client=_FakeBackendPolicyClient(
            BackendWeightOverride(
                scores={0: 0.6, "5C": 0.4},
                signature="targets",
                description="backend explicit targets",
                mode="targets",
            )
        ),
    )

    manager._tick()

    assert network.applied == [{0: 0.6, "5C": 0.4}]


def test_set_weights_accepts_uid_and_hotkey_targets(monkeypatch) -> None:
    captured = {}

    def _fake_convert_weights_and_uids_for_emit(*, uids, weights):
        return uids.tolist(), [float(value) for value in weights.tolist()]

    class _DummySubtensor:
        def set_weights(self, **kwargs):
            captured.update(kwargs)
            return True

    monkeypatch.setattr(
        _weights.bt.utils.weight_utils,
        "convert_weights_and_uids_for_emit",
        _fake_convert_weights_and_uids_for_emit,
    )

    state.WalletHolder.wallet = object()
    state.WalletHolder.subtensor = _DummySubtensor()
    state.WalletHolder.metagraph = SimpleNamespace(
        hotkeys=["5A", "5B", "5C"],
        netuid=94,
    )
    state.WalletHolder.config = {"weights": {}}
    state.WalletHolder.uid = 1
    state.WalletHolder.device = "cpu"
    state.WalletHolder.base_scores = None
    state.WalletHolder.subtensor_lock = threading.RLock()

    ok = _weights.set_weights({0: 0.7, "5C": 0.3})

    assert ok is True
    assert captured["uids"] == [0, 1, 2]
    assert captured["weights"] == pytest.approx([0.7, 0.0, 0.3], rel=1e-5)

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
from validator import backend_weight_setter


class _FakeNetwork:
    def __init__(self, *, ready: bool = True):
        self.metagraph = SimpleNamespace(hotkeys=["5Burn", "5Miner", "5Contract"], netuid=94)
        self.ready = ready
        self.applied = []
        self.resyncs = 0

    def should_set_weights(self) -> bool:
        return self.ready

    def set_weights(self, scores):
        self.applied.append(dict(scores))
        return True

    def resync_metagraph(self, lite: bool = True) -> None:
        self.resyncs += 1


def test_backend_weight_setter_defaults_to_prod_autoresearch() -> None:
    args = backend_weight_setter._build_parser().parse_args([])

    assert args.coordinator_url == "https://autoresearch.bitsota.com"
    assert args.config == "validator_config.weights.yaml"


def test_resolve_validator_weight_scores_targets_uid_and_hotkey() -> None:
    scores = backend_weight_setter.resolve_validator_weight_scores(
        {
            "mode": "targets",
            "targets": [
                {"uid": 0, "weight": 0.9},
                {"hotkey": "5Contract", "weight": 0.1},
            ],
        },
        metagraph_hotkeys=["5Burn", "5Miner", "5Contract"],
    )

    assert scores == {"5Burn": 0.9, "5Contract": 0.1}


def test_backend_weight_setter_submits_backend_targets_only() -> None:
    network = _FakeNetwork()

    outcome = backend_weight_setter.apply_backend_weight_policy(
        policy={
            "mode": "targets",
            "targets": [
                {"uid": 0, "weight": 0.9},
                {"hotkey": "5Contract", "weight": 0.1},
            ],
        },
        network=network,
    )

    assert outcome.status == "submitted"
    assert network.applied == [{"5Burn": 0.9, "5Contract": 0.1}]


def test_backend_weight_setter_local_policy_does_not_fall_back() -> None:
    network = _FakeNetwork()

    outcome = backend_weight_setter.apply_backend_weight_policy(
        policy={"mode": "local"},
        network=network,
    )

    assert outcome.status == "skipped_local"
    assert "no legacy relay/local fallback" in outcome.message
    assert network.applied == []


def test_backend_weight_setter_respects_rate_limit() -> None:
    network = _FakeNetwork(ready=False)

    outcome = backend_weight_setter.apply_backend_weight_policy(
        policy={
            "mode": "targets",
            "targets": [{"hotkey": "5Contract", "weight": 1.0}],
        },
        network=network,
    )

    assert outcome.status == "skipped_rate_limit"
    assert network.applied == []


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

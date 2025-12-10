import threading
from typing import Any, Dict

from . import _state
from . import _weights


class _SingletonMeta(type):
    _inst: Any = None
    _lock = threading.Lock()

    def __call__(cls, *a, **kw):
        with cls._lock:
            if cls._inst is None:
                cls._inst = super().__call__(*a, **kw)
        return cls._inst


class BittensorNetwork(metaclass=_SingletonMeta):
    _default_config = None
    _initialized = False

    def __init__(self, config=None, force_reinit=False):
        if config is not None and (
            BittensorNetwork._default_config is None or force_reinit
        ):
            BittensorNetwork._default_config = config
            _state.WalletHolder.initialize(config)
            BittensorNetwork._initialized = True
        elif config is None and not BittensorNetwork._initialized:
            raise ValueError("Config must be provided on first initialization")

    @classmethod
    def _get_config(cls):
        if cls._default_config is None:
            raise RuntimeError(
                "BittensorNetwork(config) must be called once before use"
            )
        return cls._default_config

    wallet = property(lambda _: _state.WalletHolder.wallet)
    subtensor = property(lambda _: _state.WalletHolder.subtensor)
    metagraph = property(lambda _: _state.WalletHolder.metagraph)
    config = property(lambda _: _state.WalletHolder.config)
    uid = property(lambda _: _state.WalletHolder.uid)
    device = property(lambda _: _state.WalletHolder.device)
    base_scores = property(lambda _: _state.WalletHolder.base_scores)

    @staticmethod
    def should_set_weights() -> bool:
        """Check if weights should be set based on the last update time and epoch length."""
        return _weights.should_set_weights()

    @staticmethod
    def set_weights(scores: Dict[str, float]):
        """Set weights for the network based on the provided scores.

        Args:
            scores: A dictionary mapping hotkey addresses to scores
        """
        _weights.set_weights(scores)

    def discover_contract_bots(self):
        """Discover contract bots from settings.

        Returns a dictionary mapping hotkey addresses to scores for weighting.
        """
        config = self._get_config()
        contract_bots = config.get("contract_bots", [])

        # Return a dictionary mapping each contract bot's ss58 address to a weight of 1.0
        return {bot_address: 1.0 for bot_address in contract_bots}

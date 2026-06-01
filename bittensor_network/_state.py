import logging
import threading

import bittensor as bt
import torch



class WalletHolder:
    wallet = None
    subtensor = None
    metagraph = None
    config = None
    uid = 0
    device = "cpu"
    base_scores = None
    subtensor_lock = threading.RLock()

    @classmethod
    def initialize(cls, config, ignore_regs: bool = False):
        cls.wallet = bt.wallet(config=config)
        cls.subtensor = bt.subtensor(config=config)
        # Substrate websocket RPC is not thread-safe; serialize all subtensor calls.
        with cls.subtensor_lock:
            cls.metagraph = cls.subtensor.metagraph(config.netuid)
        cls.config = config

        if not ignore_regs:
            with cls.subtensor_lock:
                registered = cls.subtensor.is_hotkey_registered(
                    netuid=config.netuid, hotkey_ss58=cls.wallet.hotkey.ss58_address
                )
            if not registered:
                logging.error(
                    f"Wallet {config.wallet} not registered on netuid {config.netuid}"
                )
                exit(1)

        cls.uid = (
            cls.metagraph.hotkeys.index(cls.wallet.hotkey.ss58_address)
            if cls.wallet.hotkey.ss58_address in cls.metagraph.hotkeys
            else 0
        )
        cls.base_scores = torch.zeros(
            cls.metagraph.n, dtype=torch.float32, device=cls.device
        )

    @classmethod
    def close(cls):
        """Best-effort shutdown for subtensor websocket clients."""
        subtensor = cls.subtensor
        for candidate in (getattr(subtensor, "substrate", None), subtensor):
            close = getattr(candidate, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - shutdown best effort
                    logging.debug("failed to close bittensor client", exc_info=True)

        cls.wallet = None
        cls.subtensor = None
        cls.metagraph = None
        cls.config = None
        cls.uid = 0
        cls.device = "cpu"
        cls.base_scores = None

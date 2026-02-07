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

import logging
import torch



class WalletHolder:
    wallet = None
    subtensor = None
    metagraph = None
    config = None
    uid = 0
    device = "cpu"
    base_scores = None

    @classmethod
    def initialize(cls, config, ignore_regs: bool = False):
        cls.wallet = bt.wallet(config=config)
        cls.subtensor = bt.subtensor(config=config)
        cls.metagraph = cls.subtensor.metagraph(config.netuid)
        cls.config = config

        if not ignore_regs and not cls.subtensor.is_hotkey_registered(
            netuid=config.netuid, hotkey_ss58=cls.wallet.hotkey.ss58_address
        ):
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

import logging
import time
from typing import List

import bittensor as bt
import bittensor.utils.networking as net
import torch

__spec_version__ = 1337


class BittensorNetwork:
    def __init__(self, config):
        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(config.netuid)
        self.config = config
        self.device = "cpu"

        # Check registration
        if not self.subtensor.is_hotkey_registered(
            netuid=config.netuid, hotkey_ss58=self.wallet.hotkey.ss58_address
        ):
            raise ValueError(f"Wallet not registered on netuid {config.netuid}")

        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.last_sync_time = 0

    def sync_if_needed(self, sync_interval=600):
        if time.time() - self.last_sync_time > sync_interval:
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            self.last_sync_time = time.time()

    def should_set_weights(self) -> bool:
        return (
            self.subtensor.get_current_block() - self.metagraph.last_update[self.uid]
        ) > self.config.epoch_length

    def set_weights(
        self, contract_bot_addresses: List[str], equal_weights: bool = True
    ):
        """
        Set weights to point only to contract bot addresses.

        Args:
            contract_bot_addresses: List of hotkey addresses that are contract bots
            equal_weights: If True, distribute weight equally. If False, use custom weights
        """
        try:
            # Initialize all scores to 0
            self.base_scores = torch.zeros(
                self.metagraph.n, dtype=torch.float32, device=self.device
            )

            # Find UIDs for contract bot addresses
            contract_bot_uids = []
            for uid, hotkey in enumerate(self.metagraph.hotkeys):
                if hotkey in contract_bot_addresses:
                    contract_bot_uids.append(uid)

            if not contract_bot_uids:
                logging.error("No contract bot addresses found in metagraph!")
                return

            # Set weights for contract bots
            if equal_weights:
                weight_per_bot = 1.0 / len(contract_bot_uids)
                for uid in contract_bot_uids:
                    self.base_scores[uid] = weight_per_bot
            else:
                # You could implement custom weight distribution here
                # For now, just equal weights
                weight_per_bot = 1.0 / len(contract_bot_uids)
                for uid in contract_bot_uids:
                    self.base_scores[uid] = weight_per_bot

            # Create uid tensor for all UIDs (required by bittensor)
            uids = torch.tensor(list(range(self.metagraph.n)))

            logging.info(f"Setting weights for contract bots: {contract_bot_uids}")
            logging.info(
                f"Contract bot addresses: {[self.metagraph.hotkeys[uid] for uid in contract_bot_uids]}"
            )
            logging.info(f"Weights: {self.base_scores[self.base_scores > 0]}")

            # Convert to uint16 weights and uids
            (
                uint_uids,
                uint_weights,
            ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
                uids=uids, weights=self.base_scores
            )

            logging.info("Sending weights to subtensor")

            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.metagraph.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_inclusion=False,
                version_key=__spec_version__,
            )

        except Exception as e:
            logging.error(f"Error setting weights: {e}")

    def set_weights_for_contract_bots(cls):
        """
        Convenience method to set weights for predefined contract bot addresses
        """
        # Define your contract bot addresses here
        CONTRACT_BOT_ADDRESSES = [
            "5FHneW46...",  # Bot 1 address #FIXME I am ugly
            "5GrwvaEF...",  # Bot 2 address
            # Add more as needed
        ]

        self.set_weights(CONTRACT_BOT_ADDRESSES, equal_weights=True)

    def resync_metagraph(self, lite=True):

        # Fetch the latest state of the metagraph from the Bittensor network
        logging.info("Resynchronizing metagraph...")
        # Update the metagraph with the latest information from the network
        self.metagraph = self.subtensor.metagraph(self.config.netuid, lite=lite)
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid, hotkey_ss58=self.wallet.hotkey.ss58_address
        ):
            logging.error(
                f"Wallet: {self.config.wallet} is not registered on netuid {self.config.netuid}. Please register the hotkey before trying again"
            )
            exit()
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        logging.info("Metagraph resynchronization complete.")

    @staticmethod
    def should_sync_metagraph(last_sync_time, sync_interval):
        current_time = time.time()
        return (current_time - last_sync_time) > sync_interval

    def sync(self, lite=True):
        if self.should_sync_metagraph(self.last_sync_time, self.sync_interval):
            # Assuming resync_metagraph is a method to update the metagraph with the latest state from the network.
            # This method would need to be defined or adapted from the BaseNeuron implementation.
            try:
                self.resync_metagraph(lite)
                self.last_sync_time = time.time()
            except Exception as e:
                logging.warning(f"Failed to resync metagraph: {e}")
        else:
            logging.info("Metagraph Sync Interval not yet passed")

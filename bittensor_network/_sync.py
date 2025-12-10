import logging
import time
from typing import List

from . import _state as S


def resync_metagraph(lite=True):
    logging.info("Resynchronizing metagraph…")
    S.WalletHolder.metagraph = S.WalletHolder.subtensor.metagraph(
        S.WalletHolder.config.netuid, lite=lite
    )
    if not S.WalletHolder.subtensor.is_hotkey_registered(
        netuid=S.WalletHolder.config.netuid,
        hotkey_ss58=S.WalletHolder.wallet.hotkey.ss58_address,
    ):
        logging.error(
            f"Wallet {S.WalletHolder.config.wallet} not registered on netuid "
            f"{S.WalletHolder.config.netuid}"
        )
        exit(1)
    logging.info("Metagraph resync complete.")


def should_sync_metagraph(last_sync_time: float, sync_interval: int) -> bool:
    return (time.time() - last_sync_time) > sync_interval


_sync_last_time = 0


def sync(lite=True):
    global _sync_last_time
    if should_sync_metagraph(_sync_last_time, S.WalletHolder.config.sync_interval):
        try:
            resync_metagraph(lite)
            _sync_last_time = time.time()
        except Exception as e:
            logging.warning(f"Failed to resync metagraph: {e}")
    else:
        logging.debug("Metagraph sync interval not yet passed.")


def get_validator_uids(vpermit_tao_limit: int = 1024) -> List[int]:
    return [
        uid
        for uid, stake in enumerate(S.WalletHolder.metagraph.S)
        if stake >= vpermit_tao_limit
    ]

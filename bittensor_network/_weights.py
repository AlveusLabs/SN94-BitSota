import logging
import threading

import bittensor as bt
import torch

from . import _state as S

_weights_lock = threading.Lock()
__spec_version__ = 1337


def set_weights(scores: dict):
    with _weights_lock:
        try:
            base_scores = S.WalletHolder.base_scores
            uids = []
            for uid, hk in enumerate(S.WalletHolder.metagraph.hotkeys):
                base_scores[uid] = scores.get(hk, 0.0)
                uids.append(uid)

            uids_tensor = torch.tensor(uids)
            logging.info(f"raw_weights {base_scores}")
            logging.info(f"raw_weight_uids {uids_tensor}")

            uint_uids, uint_weights = (
                bt.utils.weight_utils.convert_weights_and_uids_for_emit(
                    uids=uids_tensor, weights=base_scores
                )
            )

            result = S.WalletHolder.subtensor.set_weights(
                wallet=S.WalletHolder.wallet,
                netuid=S.WalletHolder.metagraph.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_inclusion=False,
                version_key=__spec_version__,
            )
            logging.info(f"set_weights result: {result}")
        except Exception as e:
            logging.error(f"Error setting weights: {e}")


def should_set_weights() -> bool:
    try:
        last = S.WalletHolder.metagraph.last_update[S.WalletHolder.uid]
        current = S.WalletHolder.subtensor.get_current_block()
        return (current - last) > S.WalletHolder.config.epoch_length
    except Exception:
        logging.exception("Failed to check if weights should be set")
        return True  # safer fallback

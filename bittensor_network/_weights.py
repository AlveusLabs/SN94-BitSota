import logging
import threading
from typing import Dict

import bittensor as bt
import torch
import re

from . import _state as S

_weights_lock = threading.Lock()
__spec_version__ = 1337


def _result_ok(result) -> bool:
    if isinstance(result, tuple) and result:
        return bool(result[0])
    return bool(result)


def set_weights(
    scores: dict,
    *,
    wait_for_inclusion: bool | None = None,
    wait_for_finalization: bool | None = None,
    #wait_for_finality: bool = False,
):
    with _weights_lock:
        try:
            if wait_for_inclusion is None:
                try:
                    wait_for_inclusion = bool(
                        (S.WalletHolder.config.get("weights", {}) or {}).get(
                            "wait_for_inclusion", False
                        )
                    )
                except Exception:
                    wait_for_inclusion = False

            if wait_for_finalization is None:
                try:
                    wait_for_finalization = bool(
                        (S.WalletHolder.config.get("weights", {}) or {}).get(
                            "wait_for_finalization", False
                        )
                    )
                except Exception:
                    wait_for_finalization = False

            base_scores = S.WalletHolder.base_scores
            metagraph_size = len(S.WalletHolder.metagraph.hotkeys)

            if base_scores is None or len(base_scores) != metagraph_size:
                logging.info(f"Resizing base_scores from {len(base_scores) if base_scores is not None else 0} to {metagraph_size}")
                base_scores = torch.zeros(metagraph_size, dtype=torch.float32, device=S.WalletHolder.device)
                S.WalletHolder.base_scores = base_scores

            direct_uid_scores: Dict[int, float] = {}
            direct_hotkey_scores: Dict[str, float] = {}
            for raw_key, raw_weight in dict(scores or {}).items():
                try:
                    weight_value = float(raw_weight)
                except Exception:
                    logging.warning("Ignoring non-numeric weight target %r=%r", raw_key, raw_weight)
                    continue
                if weight_value <= 0.0:
                    continue
                if isinstance(raw_key, int):
                    direct_uid_scores[int(raw_key)] = direct_uid_scores.get(int(raw_key), 0.0) + weight_value
                    continue
                key_text = str(raw_key or "").strip()
                if not key_text:
                    continue
                if key_text.isdigit():
                    uid_value = int(key_text)
                    direct_uid_scores[uid_value] = direct_uid_scores.get(uid_value, 0.0) + weight_value
                    continue
                direct_hotkey_scores[key_text] = direct_hotkey_scores.get(key_text, 0.0) + weight_value

            uids = []
            for uid, hk in enumerate(S.WalletHolder.metagraph.hotkeys):
                weight_value = float(direct_uid_scores.get(uid, 0.0)) + float(direct_hotkey_scores.get(str(hk), 0.0))
                base_scores[uid] = weight_value
                uids.append(uid)

            uids_tensor = torch.tensor(uids)
            logging.info("raw_weight_uids %s", uids_tensor)

            uint_uids, uint_weights = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
                uids=uids_tensor, weights=base_scores
            )

            with S.WalletHolder.subtensor_lock:
                kwargs = {
                    "wallet": S.WalletHolder.wallet,
                    "netuid": S.WalletHolder.metagraph.netuid,
                    "uids": uint_uids,
                    "weights": uint_weights,
                    "wait_for_inclusion": bool(wait_for_inclusion),
                    "wait_for_finalization": bool(wait_for_finalization),
                    #"wait_for_finality": bool(wait_for_finality),
                    "version_key": __spec_version__,
                }
                # Compatibility: different bittensor versions accept different keyword args.
                # Retry by dropping unknown kwargs like `wait_for_finality`.
                last_error: Exception | None = None
                for _ in range(4):
                    try:
                        result = S.WalletHolder.subtensor.set_weights(**kwargs)
                        break
                    except TypeError as e:
                        last_error = e
                        m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
                        if not m:
                            raise
                        bad = m.group(1)
                        if bad not in kwargs:
                            raise
                        logging.warning(
                            "Subtensor.set_weights() does not accept %r; retrying without it",
                            bad,
                        )
                        kwargs.pop(bad, None)
                else:
                    raise last_error  # pragma: no cover
            ok = _result_ok(result)
            logging.info("set_weights result: %s (ok=%s)", result, ok)
            return ok
        except Exception as e:
            logging.error(f"Error setting weights: {e}")
            return False


def should_set_weights() -> bool:
    try:
        netuid = int(S.WalletHolder.metagraph.netuid)
        uid = int(S.WalletHolder.uid)
        min_blocks = int(getattr(S.WalletHolder.config, "epoch_length", 0) or 0)
        with S.WalletHolder.subtensor_lock:
            try:
                bslu = int(S.WalletHolder.subtensor.blocks_since_last_update(netuid, uid))
                wrl = S.WalletHolder.subtensor.weights_rate_limit(netuid)
                if wrl is not None:
                    min_blocks = max(min_blocks, int(wrl))
                # Match bittensor's internal gate: allow only when strictly greater.
                return bslu > min_blocks
            except Exception:
                current = int(S.WalletHolder.subtensor.get_current_block())
                last = int(S.WalletHolder.metagraph.last_update[uid])
                return (current - last) > min_blocks
    except Exception:
        logging.exception("Failed to check if weights should be set")
        return True  # safer fallback

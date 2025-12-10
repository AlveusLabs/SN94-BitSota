import logging
from substrateinterface import Keypair, KeypairType

logger = logging.getLogger(__name__)


class ValidatorAuth:
    @staticmethod
    def verify_miner_signature(
        miner_hotkey: str, timestamp: str, signature: str
    ) -> bool:
        """
        Verifies the signature from a miner.

        Args:
            miner_hotkey (str): The miner's hotkey.
            timestamp (str): The timestamp of the message.
            signature (str): The signature to verify.

        Returns:
            bool: True if the signature is valid, False otherwise.
        """
        try:
            signature_bytes = bytes.fromhex(signature)
            keypair = Keypair(
                ss58_address=miner_hotkey, crypto_type=KeypairType.SR25519
            )
            is_valid = keypair.verify(timestamp.encode("utf-8"), signature_bytes)
            return is_valid
        except Exception as e:
            logger.error(
                f"Error verifying miner signature for hotkey {miner_hotkey}: {e}"
            )
            return False

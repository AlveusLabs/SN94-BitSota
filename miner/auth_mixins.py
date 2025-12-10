from abc import abstractmethod
from typing import Any, Dict

class AuthMixin:
    @abstractmethod
    def _auth_payload(self) -> Dict[str, Any]:
        """
        Return whatever the validator / pool expects.
        Example implementations are provided in concrete mix-ins.
        """
        pass

class NoAuthMixin(AuthMixin):
    def _auth_payload(self):
        return {}


class BittensorAuthMixin(AuthMixin):
    def __init__(self, wallet, *args, **kw):
        super().__init__(*args, **kw, public_address=wallet.hotkey.ss58_address)
        self.wallet = wallet

    def _auth_payload(self):
        import time

        msg = f"auth:{int(time.time())}"
        sig = self.wallet.hotkey.sign(msg).hex()
        return {
            "public_address": self.wallet.hotkey.ss58_address,
            "signature": sig,
            "message": msg,
        }

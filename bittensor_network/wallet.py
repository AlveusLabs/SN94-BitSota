import os
from typing import Optional

from substrateinterface import Keypair

from .keyfile import keyfile


class Wallet:
    """Windows-compatible implementation of the bittensor wallet"""

    def __init__(
        self,
        name: str = "default",
        hotkey: str = "default",
        path: str = "~/.bittensor/wallets/",
    ):
        self.name = name
        self.hotkey_str = hotkey
        self.path = path
        self._hotkey = None
        self._coldkey = None
        self._coldkeypub = None

    def __str__(self):
        return f"wallet({self.name}, {self.hotkey_str}, {self.path})"

    def __repr__(self):
        return self.__str__()

    @property
    def hotkey_file(self) -> keyfile:
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        hotkey_path = os.path.join(wallet_path, "hotkeys", self.hotkey_str)
        return keyfile(path=hotkey_path)

    @property
    def coldkey_file(self) -> keyfile:
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        coldkey_path = os.path.join(wallet_path, "coldkey")
        return keyfile(path=coldkey_path)

    @property
    def coldkeypub_file(self) -> keyfile:
        wallet_path = os.path.expanduser(os.path.join(self.path, self.name))
        coldkeypub_path = os.path.join(wallet_path, "coldkeypub.txt")
        return keyfile(path=coldkeypub_path)

    def create_if_non_existent(
        self, coldkey_use_password: bool = True, hotkey_use_password: bool = False
    ) -> "Wallet":
        if (
            not self.coldkey_file.exists_on_device()
            and not self.coldkeypub_file.exists_on_device()
        ):
            self.create_new_coldkey(n_words=12, use_password=coldkey_use_password)
        if not self.hotkey_file.exists_on_device():
            self.create_new_hotkey(n_words=12, use_password=hotkey_use_password)
        return self

    def create_new_coldkey(
        self,
        n_words: int = 12,
        use_password: bool = True,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> tuple:
        mnemonic = Keypair.generate_mnemonic(n_words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        if not suppress:
            print(f"Generated new coldkey with mnemonic: {mnemonic}")
        self.set_coldkey(keypair, encrypt=use_password, overwrite=overwrite)
        self.set_coldkeypub(keypair, overwrite=overwrite)
        return self, mnemonic

    def create_new_hotkey(
        self,
        n_words: int = 12,
        use_password: bool = False,
        overwrite: bool = False,
        suppress: bool = False,
    ) -> tuple:
        mnemonic = Keypair.generate_mnemonic(n_words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        if not suppress:
            print(f"Generated new hotkey with mnemonic: {mnemonic}")
        self.set_hotkey(keypair, encrypt=use_password, overwrite=overwrite)
        return self, mnemonic

    def set_hotkey(
        self,
        keypair: Keypair,
        *,
        encrypt: bool = False,
        overwrite: bool = False,
        password: Optional[str] = None,
    ) -> None:
        self._hotkey = keypair
        self.hotkey_file.set_keypair(
            keypair, encrypt=encrypt, overwrite=overwrite, password=password
        )

    def set_coldkey(
        self,
        keypair: Keypair,
        *,
        encrypt: bool = True,
        overwrite: bool = False,
        password: Optional[str] = None,
    ) -> None:
        self._coldkey = keypair
        self.coldkey_file.set_keypair(
            keypair, encrypt=encrypt, overwrite=overwrite, password=password
        )

    def set_coldkeypub(
        self, keypair: Keypair, encrypt: bool = False, overwrite: bool = False
    ) -> None:
        self._coldkeypub = Keypair(ss58_address=keypair.ss58_address)
        self.coldkeypub_file.set_keypair(
            self._coldkeypub, encrypt=encrypt, overwrite=overwrite
        )

    def get_hotkey(self, password: str = None) -> Keypair:
        return self.hotkey_file.get_keypair(password=password)

    def get_coldkey(self, password: str = None) -> Keypair:
        return self.coldkey_file.get_keypair(password=password)

    def get_coldkeypub(self, password: str = None) -> Keypair:
        return self.coldkeypub_file.get_keypair(password=password)

    def import_coldkey_from_mnemonic(
        self, mnemonic: str, *, password: Optional[str] = None, overwrite: bool = False
    ) -> "Wallet":
        """
        Re-create a coldkey from an existing 12/24-word seed and store it
        with optional encryption.  Enables painless Bittensor → native migration.
        """
        kp = Keypair.create_from_mnemonic(mnemonic)
        self.set_coldkey(
            kp, encrypt=bool(password), overwrite=overwrite, password=password
        )
        self.set_coldkeypub(kp, overwrite=overwrite)
        return self

    def import_hotkey_from_mnemonic(
        self, mnemonic: str, *, password: Optional[str] = None, overwrite: bool = False
    ) -> "Wallet":
        kp = Keypair.create_from_mnemonic(mnemonic)
        self.set_hotkey(
            kp, encrypt=bool(password), overwrite=overwrite, password=password
        )
        return self

    @property
    def hotkey(self) -> Keypair:
        if self._hotkey is None:
            self._hotkey = self.hotkey_file.keypair
        return self._hotkey

    @property
    def coldkey(self) -> Keypair:
        if self._coldkey is None:
            self._coldkey = self.coldkey_file.keypair
        return self._coldkey

    @property
    def coldkeypub(self) -> Keypair:
        if self._coldkeypub is None:
            self._coldkeypub = self.coldkeypub_file.keypair
        return self._coldkeypub

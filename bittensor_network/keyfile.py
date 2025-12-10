import getpass
import json
import os
import stat
from typing import Optional

from nacl import pwhash, secret  # PyNaCl – pure-python wheels on all OSes
from substrateinterface import Keypair, KeypairType  # SR25519 by default

SS58_FORMAT = 42
MAGIC_NACL = b"$NACL"
NACL_SALT = (
    b"\x13q\x83\xdf\xf1Z\t\xbc\x9c\x90\xb5Q\x879\xe9\xb1"  # constant, non-secret
)


def _argon2_key(password: bytes) -> bytes:
    """Derive a 32-byte key from the given password (Argon2-i, interactive params)."""
    return pwhash.argon2i.kdf(
        secret.SecretBox.KEY_SIZE,
        password,
        NACL_SALT,
        opslimit=pwhash.argon2i.OPSLIMIT_MODERATE,
        memlimit=pwhash.argon2i.MEMLIMIT_MODERATE,
    )


def _secure_permissions(filepath: str) -> None:
    try:
        os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass

class KeyFileError(Exception): ...


def encrypt_keyfile_data(plain: bytes, password: Optional[str] = None) -> bytes:
    pw = password or getpass.getpass(
        "Choose a password to encrypt the key (leave empty for none): "
    )
    if not pw:
        return plain
    key = _argon2_key(pw.encode())
    box = secret.SecretBox(key)
    return MAGIC_NACL + box.encrypt(plain)


def decrypt_keyfile_data(blob: bytes, password: Optional[str] = None) -> bytes:
    if not blob.startswith(MAGIC_NACL):
        return blob
    pw = password or getpass.getpass("Enter password to unlock key: ")
    key = _argon2_key(pw.encode())
    box = secret.SecretBox(key)
    try:
        return box.decrypt(blob[len(MAGIC_NACL) :])
    except Exception:
        raise KeyFileError("Invalid password or corrupted keyfile")


def serialize_keypair(kp: Keypair) -> bytes:
    seed_hex = None
    if kp.seed_hex:
        seed_hex = (
            kp.seed_hex
            if isinstance(kp.seed_hex, str)
            else "0x" + kp.seed_hex.hex()
        )

    return json.dumps(
        {
            "ss58Address": kp.ss58_address,
            "secretPhrase": kp.mnemonic,
            "secretSeed": seed_hex,
        }
    ).encode()


def deserialize_keypair(data: bytes) -> Keypair:
    j = json.loads(data.decode())
    if j.get("secretPhrase"):
        return Keypair.create_from_mnemonic(j["secretPhrase"], ss58_format=SS58_FORMAT)
    if j.get("secretSeed"):
        return Keypair.create_from_seed(j["secretSeed"], ss58_format=SS58_FORMAT)
    if j.get("ss58Address"):
        return Keypair(ss58_address=j["ss58Address"], crypto_type=KeypairType.SR25519)
    raise KeyFileError("Unable to reconstruct keypair from keyfile")


class keyfile:
    """
    Minimal, dependency-free container around a (possibly encrypted) keypair file.
    """

    def __init__(self, path: str):
        self.path = os.path.expanduser(path)

    def _read_raw(self) -> bytes:
        if not os.path.isfile(self.path):
            raise KeyFileError(f"Keyfile not found: {self.path}")
        with open(self.path, "rb") as f:
            return f.read()

    def _write_raw(self, data: bytes, overwrite: bool = False) -> None:
        if os.path.isfile(self.path) and not overwrite:
            raise KeyFileError(f"Refusing to overwrite existing keyfile: {self.path}")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(data)
        _secure_permissions(self.path)

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def set_keypair(
        self,
        kp: Keypair,
        *,
        encrypt: bool = False,
        overwrite: bool = False,
        password: Optional[str] = None,
    ) -> None:
        """
        Save a Keypair to disk.

        Args:
            kp:            the Keypair object.
            encrypt:       store it encrypted (True) or plaintext (False).
            overwrite:     allow overwriting an existing file.
            password:      optional password.  If None and encrypt=True the user
                           is prompted; if encrypt=False this is ignored.
        """
        data = serialize_keypair(kp)
        if encrypt:
            data = encrypt_keyfile_data(data, password)
        self._write_raw(data, overwrite=overwrite)

    def get_keypair(self, password: Optional[str] = None) -> Keypair:
        raw = self._read_raw()
        data = decrypt_keyfile_data(raw, password)
        return deserialize_keypair(data)

    @property
    def keypair(self):
        """Return the (decrypted) Keypair stored in this keyfile."""
        return self.get_keypair()

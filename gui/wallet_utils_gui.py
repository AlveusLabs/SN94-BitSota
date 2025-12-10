import json
import os
import platform
from pathlib import Path
from typing import Optional, Tuple


def get_wallet_dir() -> Path:
    if platform.system().lower() == "windows":
        app_data = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(app_data) / "BitSota" / "wallets"
    else:
        return Path.home() / ".bitsota" / "wallets"


def get_bittensor_wallet_dir() -> Path:
    if platform.system().lower() == "windows":
        potential_paths = [
            Path.home() / ".bittensor" / "wallets",
            Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "bittensor" / "wallets",
            Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "bittensor" / "wallets"
        ]
        for path in potential_paths:
            if path.exists():
                return path
        return potential_paths[0]
    else:
        return Path.home() / ".bittensor" / "wallets"


def get_settings_file() -> Path:
    return get_wallet_dir().parent / "wallet_settings.json"


def load_wallet_settings() -> dict:
    try:
        settings_file = get_settings_file()
        if settings_file.exists():
            with open(settings_file, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load wallet settings: {e}")

    return {"last_wallet": None, "last_hotkey": None, "coldkey_address": None}


def save_wallet_settings(wallet_name: str, hotkey_name: str, coldkey_address: str = None):
    try:
        settings_file = get_settings_file()
        settings_file.parent.mkdir(parents=True, exist_ok=True)

        settings = load_wallet_settings()
        settings["last_wallet"] = wallet_name
        settings["last_hotkey"] = hotkey_name
        if coldkey_address:
            settings["coldkey_address"] = coldkey_address

        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Failed to save wallet settings: {e}")


def get_last_wallet() -> Tuple[Optional[str], Optional[str]]:
    settings = load_wallet_settings()
    return settings.get("last_wallet"), settings.get("last_hotkey")


def get_coldkey_address() -> Optional[str]:
    settings = load_wallet_settings()
    return settings.get("coldkey_address")


def save_coldkey_address(coldkey_address: str):
    try:
        settings_file = get_settings_file()
        settings_file.parent.mkdir(parents=True, exist_ok=True)

        settings = load_wallet_settings()
        settings["coldkey_address"] = coldkey_address

        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Failed to save coldkey address: {e}")


def get_mining_stats_file() -> Path:
    return get_wallet_dir().parent / "mining_stats.json"


def load_mining_stats() -> dict:
    try:
        stats_file = get_mining_stats_file()
        if stats_file.exists():
            with open(stats_file, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load mining stats: {e}")

    return {"tasks_completed": 0, "successful_submissions": 0, "best_score": None}


def save_mining_stats(tasks_completed: int, successful_submissions: int, best_score: float = None):
    try:
        stats_file = get_mining_stats_file()
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "tasks_completed": tasks_completed,
            "successful_submissions": successful_submissions,
            "best_score": best_score
        }

        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Failed to save mining stats: {e}")


def validate_coldkey_address(address: str) -> Tuple[bool, str]:
    if not address or not address.strip():
        return False, "Coldkey address cannot be empty"

    address = address.strip()

    if len(address) != 48 and len(address) != 47:
        return False, "Coldkey address must be 47-48 characters long"

    if not address.isalnum():
        return False, "Coldkey address can only contain alphanumeric characters"

    if not address.startswith("5"):
        return False, "Coldkey address must start with '5'"

    return True, ""


def discover_wallets() -> list:
    wallets = []
    bitsota_dir = get_wallet_dir()
    wallets.extend(_scan_wallet_directory(bitsota_dir, source="bitsota"))
    bittensor_dir = get_bittensor_wallet_dir()
    wallets.extend(_scan_wallet_directory(bittensor_dir, source="bittensor"))

    return wallets


def _scan_wallet_directory(wallet_dir: Path, source: str) -> list:
    wallets = []
    if not wallet_dir.exists():
        return wallets
    try:
        for wallet_folder in wallet_dir.iterdir():
            if wallet_folder.is_dir():
                wallet_name = wallet_folder.name
                hotkeys = []
                hotkeys_dir = wallet_folder / "hotkeys"
                if hotkeys_dir.exists():
                    for hotkey_file in hotkeys_dir.iterdir():
                        if hotkey_file.is_file() and not hotkey_file.suffix:
                            hotkeys.append(hotkey_file.name)

                if hotkeys:
                    wallets.append((wallet_name, hotkeys, source))
    except Exception as e:
        print(f"Failed to scan wallet directory: {e}")
        return wallets

    return wallets


def validate_wallet_name(name: str) -> Tuple[bool, str]:
    if not name or not name.strip():
        return False, "Wallet name cannot be empty"

    name = name.strip()
    if len(name) < 3:
        return False, "Wallet name must be at least 3 characters"

    if len(name) > 50:
        return False, "Wallet name must be less than 50 characters"

    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        return False, "Wallet name can only contain letters, numbers, underscore, and hyphen"

    if platform.system().lower() == "windows":
        reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
                          "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2",
                          "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
        if name.upper() in reserved_names:
            return False, f"'{name}' is a reserved name on Windows"

    return True, ""


def validate_hotkey_name(name: str) -> Tuple[bool, str]:
    if not name or not name.strip():
        return False, "Hotkey name cannot be empty"

    name = name.strip()
    if len(name) < 3:
        return False, "Hotkey name must be at least 3 characters"

    if len(name) > 50:
        return False, "Hotkey name must be less than 50 characters"

    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        return False, "Hotkey name can only contain letters, numbers, underscore, and hyphen"

    if platform.system().lower() == "windows":
        reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
                          "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2",
                          "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
        if name.upper() in reserved_names:
            return False, f"'{name}' is a reserved name on Windows"

    return True, ""


def ensure_wallet_directory(use_bitsota_dir: bool = True) -> Path:
    wallet_dir = get_wallet_dir() if use_bitsota_dir else get_bittensor_wallet_dir()
    wallet_dir.mkdir(parents=True, exist_ok=True)
    return wallet_dir


def get_coldkey_address_from_wallet(wallet_name: str, source: str = "bitsota") -> Optional[str]:
    try:
        if source == "bittensor":
            wallet_dir = get_bittensor_wallet_dir()
        else:
            wallet_dir = get_wallet_dir()

        wallet_path = wallet_dir / wallet_name
        coldkeypub_path = wallet_path / "coldkeypub.txt"

        if not coldkeypub_path.exists():
            return None

        with open(coldkeypub_path, "r") as f:
            data = json.load(f)
            return data.get("ss58Address")
    except Exception as e:
        print(f"Failed to read coldkey address from wallet: {e}")
        return None

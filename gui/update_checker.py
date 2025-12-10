import requests
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from gui.version import VERSION, VERSION_CODE


class UpdateChecker:
    UPDATE_MANIFEST_URL = "https://relay.bitsota.com/version.json"
    CHECK_INTERVAL_HOURS = 24

    def __init__(self, settings_dir: Optional[Path] = None):
        if settings_dir is None:
            settings_dir = Path.home() / ".bitsota"
        self.settings_dir = settings_dir
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        self.update_settings_file = self.settings_dir / "update_settings.json"

    def _load_update_settings(self) -> Dict[str, Any]:
        if not self.update_settings_file.exists():
            return {
                "last_check_timestamp": 0,
                "skipped_version_code": None
            }

        try:
            with open(self.update_settings_file, "r") as f:
                return json.load(f)
        except:
            return {
                "last_check_timestamp": 0,
                "skipped_version_code": None
            }

    def _save_update_settings(self, settings: Dict[str, Any]):
        try:
            with open(self.update_settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Failed to save update settings: {e}")

    def should_check_for_updates(self) -> bool:
        settings = self._load_update_settings()
        last_check = settings.get("last_check_timestamp", 0)
        current_time = time.time()

        hours_since_last_check = (current_time - last_check) / 3600
        return hours_since_last_check >= self.CHECK_INTERVAL_HOURS

    def check_for_updates(self, force: bool = False) -> Optional[Dict[str, Any]]:
        print(f"[UpdateChecker] Checking for updates from {self.UPDATE_MANIFEST_URL}")
        print(f"[UpdateChecker] Current version: {VERSION} (code: {VERSION_CODE})")

        settings = self._load_update_settings()

        try:
            response = requests.get(self.UPDATE_MANIFEST_URL, timeout=10)
            response.raise_for_status()
            manifest = response.json()
            print(f"[UpdateChecker] Server response: {manifest}")

            remote_version_code = manifest.get("versionCode")
            if remote_version_code is None:
                print("[UpdateChecker] No versionCode in server response")
                return None

            print(f"[UpdateChecker] Remote version: {manifest.get('version')} (code: {remote_version_code})")

            skipped_version = settings.get("skipped_version_code")
            if skipped_version == remote_version_code:
                print(f"[UpdateChecker] User previously skipped version {remote_version_code}")
                return None

            if remote_version_code > VERSION_CODE:
                print(f"[UpdateChecker] Update available! {VERSION_CODE} -> {remote_version_code}")
                return {
                    "current_version": VERSION,
                    "current_version_code": VERSION_CODE,
                    "new_version": manifest.get("version", "Unknown"),
                    "new_version_code": remote_version_code,
                    "description": manifest.get("desc", ""),
                    "mac_url": manifest.get("mac"),
                    "linux_url": manifest.get("linux"),
                    "windows_url": manifest.get("windows")
                }
            else:
                print(f"[UpdateChecker] No update needed - running latest version")

            return None

        except Exception as e:
            print(f"[UpdateChecker] Failed to check for updates: {e}")
            import traceback
            traceback.print_exc()
            return None

    def skip_version(self, version_code: int):
        settings = self._load_update_settings()
        settings["skipped_version_code"] = version_code
        self._save_update_settings(settings)

    def get_download_url(self, update_info: Dict[str, Any]) -> Optional[str]:
        import platform
        system = platform.system().lower()

        if system == "darwin":
            return update_info.get("mac_url")
        elif system == "linux":
            return update_info.get("linux_url")
        elif system == "windows":
            return update_info.get("windows_url")

        return None

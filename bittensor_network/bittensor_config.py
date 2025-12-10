from pathlib import Path
import yaml
from collections import UserDict


class ConfigWrapper:
    def __init__(self, bt_config, yaml_data):
        self.bt_config = bt_config
        self.yaml_data = yaml_data

        for key, value in yaml_data.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        # Delegate attribute access to the bittensor config
        return getattr(self.bt_config, name)

    def get(self, key, default=None):
        """Provide dict-like get access to YAML data"""
        return self.yaml_data.get(key, default)


_DEFAULT_YAML = """
wallet_name: "your_wallet"
wallet_hotkey: "your_hotkey"
path: "~/.bittensor/wallets/"
network: "finney"
epoch_length: 100
subtensor_chain_endpoint: null   # null → let bittensor pick default
"""


class BittensorConfig:
    _yaml_path = Path("validator_config.yaml").expanduser().resolve()

    @classmethod
    def _load_yaml(cls) -> dict:
        """Return dict from YAML, or empty dict if file missing / broken."""
        if not cls._yaml_path.exists():
            cls._yaml_path.write_text(_DEFAULT_YAML)
            return yaml.safe_load(_DEFAULT_YAML)
        try:
            return yaml.safe_load(cls._yaml_path.read_text()) or {}
        except yaml.YAMLError:
            return {}

    @classmethod
    def get_bittensor_config(cls):
        y = cls._load_yaml()
        bt_config = bt.config()
        if not hasattr(bt_config, "wallet") or bt_config.wallet is None:
            bt_config.wallet = bt.Config()  # Create empty config object
        setattr(bt_config.wallet, "name", y.get("wallet_name", "your_wallet"))
        setattr(bt_config.wallet, "hotkey", y.get("wallet_hotkey", "your_hotkey"))
        if not hasattr(bt_config, "subtensor") or bt_config.subtensor is None:
            bt_config.subtensor = bt.Config()  # Create empty config object
        setattr(bt_config.subtensor, "network", y.get("network", "finney"))
        endpoint = y.get("subtensor_chain_endpoint")
        if endpoint:
            setattr(bt_config.subtensor, "chain_endpoint", endpoint)
        if not hasattr(bt_config, "netuid") or bt_config.netuid is None:
            bt_config.netuid = y.get("netuid", 94)

        if not hasattr(bt_config, "epoch_length") or bt_config.epoch_length is None:
            bt_config.epoch_length = y.get("epoch_length", 100)

        return ConfigWrapper(bt_config, y)
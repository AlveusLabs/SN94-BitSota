#!/usr/bin/env python3
"""
CLI Miner for AutoML Infinite
A command-line miner that supports both direct validator and pool modes
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miner.client import BittensorDirectClient, BittensorPoolClient




def setup_logging(config):
    """Setup logging based on configuration"""
    log_level = getattr(logging, config["logging"]["level"].upper())
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = []

    # Console handler
    handlers.append(logging.StreamHandler(sys.stdout))

    # File handler if specified
    if config["logging"]["file"]:
        log_file = Path(config["logging"]["file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        sys.exit(1)


def create_wallet(config):
    """Create real bittensor wallet from config"""
    try:
        import bittensor as bt

        wallet = bt.wallet(
            name=config["wallet"]["wallet_name"],
            hotkey=config["wallet"]["hotkey_name"],
        )
        return wallet
    except ImportError:
        print("ERROR: Bittensor not available. Please install bittensor: pip install bittensor")
        sys.exit(1)


def create_client(config, wallet):
    """Create appropriate client based on configuration"""
    if config["pool_url"] is None:
        # Direct validator mode - use first validator as relay endpoint
        validators = config["validators"]
        relay_endpoint = validators[0] if validators else "http://127.0.0.1:8002"
        return BittensorDirectClient(
            wallet=wallet,
            relay_endpoint=relay_endpoint,
            verbose=config["evolution"]["verbose"],
        )
    else:
        # Pool mode
        return BittensorPoolClient(
            public_address=wallet.hotkey.ss58_address,
            wallet=wallet,  # FIXME repetitive args wallet + public_address
            base_url=config["pool_url"],
        )


def run_miner(config):
    logger = logging.getLogger(__name__)

    os.environ["MAX_EVOLUTION_GENERATIONS"] = str(
        config["evolution"]["max_generations"]
    )

    wallet = create_wallet(config)
    client = create_client(config, wallet)

    logger.info(f"Starting miner with config: {config}")
    logger.info(f"Wallet address: {wallet.hotkey.ss58_address}")
    logger.info(f"Client type: {type(client).__name__}")

    try:
        with client:
            if config["pool_url"] is None:
                # Direct mode
                client.run_continuous_mining(
                    engine_type=config["mining"]["engine_type"],
                    task_type=config["mining"]["task_type"],
                )
            else:
                # Pool mode
                client.run_continuous_mining(
                    cycles=config["mining"]["cycles"],
                    alternate=config["mining"]["alternate_tasks"],
                    delay=config["mining"]["delay"],
                    max_retries=config["mining"]["max_retries"],
                )
    except KeyboardInterrupt:
        logger.info("Miner stopped by user")
    except Exception as e:
        logger.error(f"Miner error: {e}", exc_info=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AutoML Infinite CLI Miner")
    parser.add_argument(
        "--config",
        default="miner_config.yaml",
        help="Path to configuration file (default: miner_config.yaml)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        help="Number of mining cycles (0=infinite, overrides config)",
    )
    parser.add_argument(
        "--delay", type=float, help="Delay between cycles in seconds (overrides config)"
    )
    parser.add_argument(
        "--pool-url", help="Pool URL (null for direct mode, overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.cycles is not None:
        config["mining"]["cycles"] = args.cycles
    if args.delay is not None:
        config["mining"]["delay"] = args.delay
    if args.pool_url is not None:
        config["pool_url"] = args.pool_url

    setup_logging(config)

    run_miner(config)


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
import os
import threading
import traceback
from typing import Any, Dict, Optional

from bittensor_network.wallet import Wallet
from miner.client import BittensorDirectClient

from miner.island_model import IslandEngineWrapper, seed_worker_rng


class QueueLogHandler(logging.Handler):
    def __init__(self, out_queue: Any, worker_id: int):
        super().__init__()
        self._out_queue = out_queue
        self._worker_id = int(worker_id)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        try:
            self._out_queue.put(
                {
                    "type": "log",
                    "worker_id": self._worker_id,
                    "message": str(msg),
                    "logger": record.name,
                    "level": record.levelname,
                }
            )
        except Exception:
            # Logging should never crash the worker.
            pass


def _install_island_engine_wrapper(
    client: BittensorDirectClient,
    *,
    worker_id: int,
    migration_generations: int,
    out_queue: Any,
    in_queue: Any,
    stop_event: Any,
) -> None:
    migration_generations = max(0, int(migration_generations))
    if migration_generations <= 0:
        return

    wrapped_cache: Dict[tuple[str, str], IslandEngineWrapper] = {}
    original_get_engine = client._get_engine

    def _wrapped_get_engine(task_type: str, engine_type: str = "archive"):
        key = (str(task_type), str(engine_type))
        cached = wrapped_cache.get(key)
        if cached is not None:
            return cached
        engine = original_get_engine(task_type, engine_type)
        wrapped = IslandEngineWrapper(
            engine,
            worker_id=int(worker_id),
            migration_generations=migration_generations,
            out_queue=out_queue,
            in_queue=in_queue,
            stop_event=stop_event,
        )
        wrapped_cache[key] = wrapped
        return wrapped

    client._get_engine = _wrapped_get_engine  # type: ignore[method-assign]


def run_direct_mining_worker(
    worker_config: Dict[str, Any],
    worker_id: int,
    out_queue: Any,
    in_queue: Any,
    stop_event: Any,
) -> None:
    """
    Worker process entrypoint for GUI multi-process mining.

    Expects worker_config keys:
      - wallet_name, wallet_hotkey, wallet_path
      - relay_endpoint
      - miner_task_count
      - validator_task_count (optional)
      - validate_every_n_generations (optional)
      - engine_params (optional)
      - env_overrides (optional)
      - task_type, engine_type, checkpoint_generations
      - seed (optional), migration_generations (optional)
    """

    worker_id = int(worker_id)
    try:
        env_overrides = worker_config.get("env_overrides")
        if isinstance(env_overrides, dict):
            for key, value in env_overrides.items():
                name = str(key).strip()
                if not name:
                    continue
                os.environ[name] = str(value)

        seed_worker_rng(worker_config.get("seed"), worker_id)

        handler = QueueLogHandler(out_queue, worker_id)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.INFO)

        tracked_loggers = [
            logging.getLogger("miner"),
            logging.getLogger("core"),
        ]
        for lg in tracked_loggers:
            lg.addHandler(handler)
            lg.setLevel(logging.INFO)

        wallet = Wallet(
            name=str(worker_config.get("wallet_name") or "default"),
            hotkey=str(worker_config.get("wallet_hotkey") or "default"),
            path=str(worker_config.get("wallet_path") or "~/.bittensor/wallets/"),
        )

        relay_endpoint = str(worker_config.get("relay_endpoint") or "")
        miner_task_count = worker_config.get("miner_task_count")
        validator_task_count = worker_config.get("validator_task_count")
        validate_every_n_generations = worker_config.get("validate_every_n_generations")
        engine_params = worker_config.get("engine_params")
        verbose = bool(worker_config.get("verbose", True))

        client = BittensorDirectClient(
            wallet=wallet,
            relay_endpoint=relay_endpoint,
            verbose=verbose,
            contract_manager=None,
            miner_task_count=miner_task_count,
            validator_task_count=validator_task_count,
            validate_every_n_generations=validate_every_n_generations,
            engine_params=engine_params if isinstance(engine_params, dict) else None,
            worker_id=worker_id,
            state_dir=worker_config.get("state_dir"),
        )

        migration_generations = int(worker_config.get("migration_generations") or 0)
        _install_island_engine_wrapper(
            client,
            worker_id=worker_id,
            migration_generations=migration_generations,
            out_queue=out_queue,
            in_queue=in_queue,
            stop_event=stop_event,
        )

        def _stop_watcher():
            try:
                stop_event.wait()
            except Exception:
                return
            try:
                client.stop_mining()
            except Exception:
                pass

        stopper = threading.Thread(target=_stop_watcher, daemon=True)
        stopper.start()

        task_type = str(worker_config.get("task_type") or "cifar10_binary")
        engine_type = str(worker_config.get("engine_type") or "baseline")
        checkpoint_generations = int(worker_config.get("checkpoint_generations") or 10)
        result = client.run_continuous_mining(
            task_type=task_type,
            engine_type=engine_type,
            checkpoint_generations=checkpoint_generations,
        )

        out_queue.put({"type": "done", "worker_id": worker_id, "result": result})
    except KeyboardInterrupt:
        try:
            out_queue.put({"type": "done", "worker_id": worker_id, "result": {"status": "stopped"}})
        except Exception:
            pass
    except Exception:
        try:
            out_queue.put(
                {
                    "type": "error",
                    "worker_id": worker_id,
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        try:
            stop_event.set()
        except Exception:
            pass

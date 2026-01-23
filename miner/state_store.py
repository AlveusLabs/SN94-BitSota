from __future__ import annotations

import logging
import os
import pickle
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from core.dsl_parser import DSLParser
from miner.engines.archive_engine import ArchiveAwareBaselineEvolution
from miner.engines.base_engine import BaseEvolutionEngine
from miner.engines.ga_engine import BaselineEvolutionEngine

logger = logging.getLogger("miner.state")

_SCHEMA_VERSION = 1


def _default_state_dir() -> Path:
    raw = (
        os.getenv("MINER_STATE_DIR")
        or os.getenv("BITSOTA_MINER_STATE_DIR")
        or os.getenv("BITSOTA_STATE_DIR")
    )
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".bitsota" / "miner_state").resolve()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp.write(payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _algo_to_payload(algo: object) -> Dict[str, object]:
    try:
        input_dim = int(getattr(algo, "input_dim", 0) or 0)
    except Exception:
        input_dim = 0
    return {"format": "dsl", "dsl": DSLParser.to_dsl(algo), "input_dim": input_dim}


def _payload_to_algo(payload: object) -> object:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, got {type(payload).__name__}")
    if payload.get("format") != "dsl":
        raise ValueError(f"Unsupported algo payload format: {payload.get('format')}")
    dsl = str(payload.get("dsl") or "")
    input_dim = payload.get("input_dim")
    if input_dim is None:
        raise ValueError("Algo payload missing input_dim")
    return DSLParser.from_dsl(dsl, int(input_dim))


def _serialize_engine(engine: BaseEvolutionEngine) -> Dict[str, Any]:
    if isinstance(engine, BaselineEvolutionEngine):
        pop_queue = getattr(engine, "_population_queue", None)
        if pop_queue is None or len(pop_queue) == 0:
            raise ValueError("baseline engine has no population queue to persist")

        queue_payload = []
        for entry in list(pop_queue):
            algo = entry.get("algo")
            if algo is None:
                continue
            try:
                fitness = float(entry.get("fitness", -np.inf))
            except Exception:
                fitness = -np.inf
            queue_payload.append({"algo": _algo_to_payload(algo), "fitness": float(fitness)})

        best_algo = getattr(engine, "best_algo", None)
        best_payload = _algo_to_payload(best_algo) if best_algo is not None else None

        return {
            "engine_class": type(engine).__name__,
            "engine_kind": "baseline",
            "pop_size": int(getattr(engine, "pop_size", 0) or 0),
            "generation": int(getattr(engine, "generation", 0) or 0),
            "best_fitness": float(getattr(engine, "best_fitness", -np.inf) or -np.inf),
            "best_algo": best_payload,
            "queue": queue_payload,
            "stagnation": int(getattr(engine, "_stagnation", 0) or 0),
            "tournament_size": int(getattr(engine, "tournament_size", 0) or 0),
            "mutation_prob": float(getattr(engine, "mutation_prob", 0.0) or 0.0),
            "miner_task_count": int(getattr(engine, "miner_task_count", 0) or 0),
            "cifar_seed": int(getattr(engine, "_cifar_seed", 0) or 0),
        }

    if isinstance(engine, ArchiveAwareBaselineEvolution):
        population = getattr(engine, "population", None)
        if population is None or len(population) == 0:
            raise ValueError("archive engine has no population to persist")

        pop_payload = []
        for algo in list(population or []):
            pop_payload.append(_algo_to_payload(algo))

        archive_payload = []
        archive = getattr(engine, "archive", None)
        if archive is not None:
            for entry in list(archive):
                algo = entry.get("algorithm") if isinstance(entry, dict) else None
                if algo is None:
                    continue
                try:
                    fitness = float(entry.get("fitness", -np.inf))
                except Exception:
                    fitness = -np.inf
                archive_payload.append(
                    {
                        "algorithm": _algo_to_payload(algo),
                        "fitness": float(fitness),
                        "generation": int(entry.get("generation", 0) or 0),
                        "complexity": int(entry.get("complexity", 0) or 0),
                        "signature": str(entry.get("signature", "")),
                    }
                )

        best_algo = getattr(engine, "best_algo", None)
        best_payload = _algo_to_payload(best_algo) if best_algo is not None else None

        archive_maxlen = getattr(archive, "maxlen", None) if archive is not None else None
        try:
            archive_size = int(archive_maxlen) if archive_maxlen is not None else 0
        except Exception:
            archive_size = 0

        return {
            "engine_class": type(engine).__name__,
            "engine_kind": "archive",
            "pop_size": int(getattr(engine, "pop_size", 0) or 0),
            "generation": int(getattr(engine, "generation", 0) or 0),
            "generation_counter": int(getattr(engine, "generation_counter", 0) or 0),
            "best_fitness": float(getattr(engine, "best_fitness", -np.inf) or -np.inf),
            "best_algo": best_payload,
            "population": pop_payload,
            "archive_size": int(archive_size),
            "archive": archive_payload,
            "archive_diversity": dict(getattr(engine, "archive_diversity", {}) or {}),
            "miner_task_count": int(getattr(engine, "miner_task_count", 0) or 0),
            "cifar_seed": int(getattr(engine, "_cifar_seed", 0) or 0),
        }

    raise TypeError(f"Unsupported engine type for persistence: {type(engine).__name__}")


def _apply_engine_state(engine: BaseEvolutionEngine, payload: Dict[str, Any]) -> None:
    kind = str(payload.get("engine_kind") or "")

    if kind == "baseline":
        if not isinstance(engine, BaselineEvolutionEngine):
            raise TypeError(
                f"Cannot apply baseline state to {type(engine).__name__}"
            )

        saved_pop_size = int(payload.get("pop_size", engine.pop_size) or engine.pop_size)
        if saved_pop_size > 0 and int(engine.pop_size) != saved_pop_size:
            engine.pop_size = int(saved_pop_size)

        queue_items = list(payload.get("queue") or [])
        restored_entries = []
        for item in queue_items:
            if not isinstance(item, dict):
                continue
            algo_payload = item.get("algo")
            if algo_payload is None:
                continue
            algo = _payload_to_algo(algo_payload)
            try:
                fitness = float(item.get("fitness", -np.inf))
            except Exception:
                fitness = -np.inf
            restored_entries.append({"algo": algo, "fitness": float(fitness)})

        if not restored_entries:
            raise ValueError("baseline engine state has empty queue")

        engine._population_queue = deque(restored_entries, maxlen=int(engine.pop_size))  # type: ignore[attr-defined]
        engine.population = [entry["algo"] for entry in engine._population_queue]

        best_payload = payload.get("best_algo")
        engine.best_algo = _payload_to_algo(best_payload) if best_payload else None
        engine.best_fitness = float(payload.get("best_fitness", -np.inf) or -np.inf)
        engine.generation = int(payload.get("generation", 0) or 0)
        engine._stagnation = int(payload.get("stagnation", 0) or 0)  # type: ignore[attr-defined]

        tournament_size = payload.get("tournament_size")
        if tournament_size is not None:
            try:
                engine.tournament_size = max(
                    1, min(int(engine.pop_size), int(tournament_size))
                )
            except Exception:
                pass

        mutation_prob = payload.get("mutation_prob")
        if mutation_prob is not None:
            try:
                engine.mutation_prob = float(np.clip(float(mutation_prob), 0.0, 1.0))
            except Exception:
                pass

        return

    if kind == "archive":
        if not isinstance(engine, ArchiveAwareBaselineEvolution):
            raise TypeError(
                f"Cannot apply archive state to {type(engine).__name__}"
            )

        saved_pop_size = int(payload.get("pop_size", engine.pop_size) or engine.pop_size)
        if saved_pop_size > 0 and int(engine.pop_size) != saved_pop_size:
            engine.pop_size = int(saved_pop_size)

        pop_items = list(payload.get("population") or [])
        restored_population = []
        for algo_payload in pop_items:
            if algo_payload is None:
                continue
            restored_population.append(_payload_to_algo(algo_payload))
        if not restored_population:
            raise ValueError("archive engine state has empty population")
        engine.population = restored_population

        archive_size = int(payload.get("archive_size", 0) or 0)
        restored_archive = deque(maxlen=archive_size if archive_size > 0 else None)
        for entry in list(payload.get("archive") or []):
            if not isinstance(entry, dict):
                continue
            algo_payload = entry.get("algorithm")
            if algo_payload is None:
                continue
            restored_archive.append(
                {
                    "algorithm": _payload_to_algo(algo_payload),
                    "fitness": float(entry.get("fitness", -np.inf) or -np.inf),
                    "generation": int(entry.get("generation", 0) or 0),
                    "complexity": int(entry.get("complexity", 0) or 0),
                    "signature": str(entry.get("signature", "")),
                }
            )
        engine.archive = restored_archive

        engine.archive_diversity = dict(payload.get("archive_diversity") or {})
        engine.generation = int(payload.get("generation", 0) or 0)
        engine.generation_counter = int(payload.get("generation_counter", engine.generation) or engine.generation)

        best_payload = payload.get("best_algo")
        engine.best_algo = _payload_to_algo(best_payload) if best_payload else None
        engine.best_fitness = float(payload.get("best_fitness", -np.inf) or -np.inf)
        return

    raise ValueError(f"Unknown engine_kind in state: {kind!r}")


class MinerStateStore:
    def __init__(
        self,
        *,
        public_address: str,
        worker_id: int = 0,
        state_dir: Optional[Path] = None,
    ):
        self.public_address = str(public_address or "unknown")
        self.worker_id = int(worker_id)
        self.state_dir = (state_dir or _default_state_dir()).resolve()

    def _root(self) -> Path:
        return self.state_dir / self.public_address / f"worker_{int(self.worker_id)}"

    def client_state_path(self) -> Path:
        return self._root() / "client_state.pkl"

    def engine_state_path(self, *, task_type: str, engine_type: str) -> Path:
        return self._root() / str(task_type) / str(engine_type) / "engine_state.pkl"

    def save_client_state(self, payload: Dict[str, Any]) -> Optional[Path]:
        wrapped = {
            "schema_version": _SCHEMA_VERSION,
            "saved_at": float(time.time()),
            "public_address": self.public_address,
            "worker_id": int(self.worker_id),
            "payload": dict(payload or {}),
        }
        path = self.client_state_path()
        try:
            _atomic_write_bytes(path, pickle.dumps(wrapped, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:
            logger.warning("Failed to persist client state to %s: %s", str(path), str(e))
            return None
        return path

    def load_client_state(self) -> Optional[Dict[str, Any]]:
        path = self.client_state_path()
        if not path.exists():
            return None
        try:
            wrapped = pickle.loads(path.read_bytes())
        except Exception as e:
            logger.warning("Failed to load client state from %s: %s", str(path), str(e))
            return None
        if not isinstance(wrapped, dict) or int(wrapped.get("schema_version", 0) or 0) != _SCHEMA_VERSION:
            return None
        payload = wrapped.get("payload")
        return dict(payload) if isinstance(payload, dict) else None

    def save_engine_state(
        self,
        *,
        task_type: str,
        engine_type: str,
        engine: BaseEvolutionEngine,
    ) -> Optional[Path]:
        try:
            engine_payload = _serialize_engine(engine)
        except ValueError:
            # Engine is not initialized yet; avoid overwriting any previous good checkpoint.
            return None
        wrapped = {
            "schema_version": _SCHEMA_VERSION,
            "saved_at": float(time.time()),
            "public_address": self.public_address,
            "worker_id": int(self.worker_id),
            "task_type": str(task_type),
            "engine_type": str(engine_type),
            "engine": engine_payload,
        }
        path = self.engine_state_path(task_type=task_type, engine_type=engine_type)
        try:
            _atomic_write_bytes(path, pickle.dumps(wrapped, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception as e:
            logger.warning(
                "Failed to persist engine state %s/%s to %s: %s",
                str(task_type),
                str(engine_type),
                str(path),
                str(e),
            )
            return None
        return path

    def load_engine_state(
        self,
        *,
        task_type: str,
        engine_type: str,
        engine: BaseEvolutionEngine,
    ) -> bool:
        path = self.engine_state_path(task_type=task_type, engine_type=engine_type)
        if not path.exists():
            return False
        try:
            wrapped = pickle.loads(path.read_bytes())
        except Exception as e:
            logger.warning("Failed to load engine state from %s: %s", str(path), str(e))
            return False
        if not isinstance(wrapped, dict) or int(wrapped.get("schema_version", 0) or 0) != _SCHEMA_VERSION:
            return False
        if str(wrapped.get("task_type") or "") != str(task_type):
            return False
        if str(wrapped.get("engine_type") or "") != str(engine_type):
            return False
        engine_payload = wrapped.get("engine")
        if not isinstance(engine_payload, dict):
            return False
        try:
            _apply_engine_state(engine, engine_payload)
        except Exception as e:
            logger.warning(
                "Failed to apply engine state from %s (task=%s engine=%s): %s",
                str(path),
                str(task_type),
                str(engine_type),
                str(e),
            )
            return False
        return True

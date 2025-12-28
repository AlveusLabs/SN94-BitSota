import math
import tempfile
from collections import deque
from pathlib import Path
import unittest

import numpy as np

from core.algorithm_array import AlgorithmArray, OPCODES
from core.tasks.base import Task
from miner.client import DirectClient
from miner.engines.ga_engine import BaselineEvolutionEngine
from miner.state_store import (
    default_state_path,
    read_state_file,
    score_from_json,
    score_to_json,
    write_state_file,
)


class DummyTask(Task):
    def __init__(self):
        super().__init__("dummy", "classification")
        self.input_dim = 3

    def load_data(self, **kwargs):
        self.X_train = np.zeros((1, self.input_dim), dtype=np.float32)
        self.y_train = np.zeros(1, dtype=np.float32)
        self.X_val = np.zeros((1, self.input_dim), dtype=np.float32)
        self.y_val = np.zeros(1, dtype=np.float32)

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        return 0.0

    def get_task_description(self) -> str:
        return "dummy task"

    def get_baseline_fitness(self) -> float:
        return 0.0


class DummyClient(DirectClient):
    def _auth_payload(self):
        return {}


class StateStoreTests(unittest.TestCase):
    def test_score_roundtrip(self):
        values = [0.0, -1.25, float("inf"), -float("inf")]
        for value in values:
            encoded = score_to_json(value)
            decoded = score_from_json(encoded)
            self.assertEqual(decoded, value)

        encoded_nan = score_to_json(float("nan"))
        decoded_nan = score_from_json(encoded_nan)
        self.assertTrue(math.isnan(decoded_nan))

        self.assertEqual(score_from_json(None, default=-5.0), -5.0)

    def test_state_file_roundtrip(self):
        payload = {"hello": "world", "number": 3, "nested": {"ok": True}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "state.json"
            write_state_file(path, payload)
            loaded = read_state_file(path)
        self.assertEqual(loaded, payload)

    def test_default_state_path(self):
        path = default_state_path()
        self.assertEqual(path.name, "mining_state.json")

    def test_algorithm_array_serialization(self):
        algo = AlgorithmArray.create_empty(
            input_dim=4,
            phases=["setup", "predict", "learn"],
            max_sizes={"setup": 3, "predict": 3, "learn": 3},
            scalar_count=7,
            vector_count=9,
            matrix_count=2,
            vector_dim=42,
        )
        algo.add_instruction("predict", "CONST", dest=1, const1=0.5)
        data = algo.to_dict()
        restored = AlgorithmArray.from_dict(data)
        self.assertEqual(restored.scalar_count, 7)
        self.assertEqual(restored.vector_count, 9)
        self.assertEqual(restored.matrix_count, 2)
        self.assertEqual(restored.vector_dim, 42)
        self.assertEqual(restored.get_phase_size("predict"), 1)
        ops, _, _, dests, const1, _ = restored.get_phase_ops("predict")
        self.assertEqual(int(ops[0]), OPCODES["CONST"])
        self.assertEqual(int(dests[0]), 1)
        self.assertAlmostEqual(float(const1[0]), 0.5, places=6)

    def test_engine_state_roundtrip(self):
        task = DummyTask()
        engine = BaselineEvolutionEngine(task, pop_size=2)
        algo_a = engine.create_initial_algorithm()
        algo_a.add_instruction("predict", "CONST", dest=0, const1=0.1)
        algo_b = engine.create_initial_algorithm()
        algo_b.add_instruction("predict", "CONST", dest=1, const1=0.9)
        engine.best_algo = algo_b
        engine.best_fitness = 2.5
        engine.population = [algo_a, algo_b]
        engine._population_queue = deque(maxlen=engine.pop_size)
        engine._population_queue.append({"algo": algo_a, "fitness": 1.1})
        engine._population_queue.append({"algo": algo_b, "fitness": 2.5})

        state = engine.get_state()

        engine_restored = BaselineEvolutionEngine(task, pop_size=2)
        engine_restored.load_state(state)
        self.assertEqual(engine_restored.best_fitness, 2.5)
        self.assertIsNotNone(engine_restored.best_algo)
        self.assertEqual(engine_restored.best_algo.fingerprint(), algo_b.fingerprint())
        self.assertEqual(len(engine_restored.population), 2)
        self.assertIsNotNone(engine_restored._population_queue)
        self.assertEqual(len(engine_restored._population_queue), 2)

    def test_client_save_load_and_clear(self):
        task = DummyTask()
        engine = BaselineEvolutionEngine(task, pop_size=2)
        algo = engine.create_initial_algorithm()
        algo.add_instruction("predict", "CONST", dest=0, const1=0.7)
        engine.population = [algo]
        engine.best_algo = algo
        engine.best_fitness = 1.25
        engine.generation = 4

        client = DummyClient(public_address="test")
        client._local_best_verified_score["dummy"] = 1.11
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "state.json"
            client._save_population_state(
                state_path=state_path,
                task_type="dummy",
                engine_type="baseline",
                engine=engine,
                generation=engine.generation,
                best_ever_score=engine.best_fitness,
                generations_since_improvement=2,
            )

            restored_engine = BaselineEvolutionEngine(task, pop_size=2)
            run_state = client._load_population_state(
                state_path=state_path,
                task_type="dummy",
                engine_type="baseline",
                engine=restored_engine,
            )
            self.assertIsNotNone(run_state)
            self.assertEqual(restored_engine.best_fitness, 1.25)
            self.assertEqual(restored_engine.generation, 4)
            self.assertEqual(len(restored_engine.population), 1)

            result = client.clear_population_state(
                task_type="dummy",
                engine_type="baseline",
                state_path=str(state_path),
            )
            self.assertEqual(result.get("status"), "cleared")
            self.assertFalse(state_path.exists())


if __name__ == "__main__":
    unittest.main()

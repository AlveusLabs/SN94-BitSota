import os
import unittest
from contextlib import contextmanager

import numpy as np

from core.algorithm_array import AlgorithmArray, ADDR_MATRICES, ADDR_VECTORS
from core.tasks.base import Task
import core.array_executor as array_executor


class DummyTask(Task):
    def __init__(self, input_dim: int):
        super().__init__("dummy", "classification")
        self.input_dim = input_dim

    def load_data(self, **kwargs):
        pass

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        return 0.0

    def get_task_description(self) -> str:
        return "dummy"

    def get_baseline_fitness(self) -> float:
        return 0.0


@contextmanager
def _set_env(**values):
    old = {}
    for key, value in values.items():
        old[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, prev in old.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _build_full_coverage_algorithm() -> AlgorithmArray:
    algo = AlgorithmArray.create_empty(
        input_dim=3,
        phases=["setup", "predict", "learn"],
        max_sizes={"setup": 64, "predict": 512, "learn": 1},
        scalar_count=20,
        vector_count=10,
        matrix_count=5,
        vector_dim=3,
    )

    def s(idx: int) -> int:
        return idx

    def v(idx: int) -> int:
        return ADDR_VECTORS + idx

    def m(idx: int) -> int:
        return ADDR_MATRICES + idx

    # setup base constants
    algo.add_instruction("setup", "CONST", -1, -1, s(1), 0.8, 0.0)
    algo.add_instruction("setup", "CONST", -1, -1, s(2), -0.6, 0.0)

    v1_vals = [0.9, -0.5, 0.7]
    v2_vals = [-0.4, 0.6, -0.3]
    ones = [1.0, 1.0, 1.0]

    for idx, val in enumerate(v1_vals):
        algo.add_instruction("setup", "CONST_VEC", -1, -1, v(1), float(idx), val)
    for idx, val in enumerate(v2_vals):
        algo.add_instruction("setup", "CONST_VEC", -1, -1, v(2), float(idx), val)
    for idx, val in enumerate(ones):
        algo.add_instruction("setup", "CONST_VEC", -1, -1, v(4), float(idx), val)

    algo.add_instruction("setup", "OUTER", v(1), v(2), m(0), 0.0, 0.0)
    algo.add_instruction("setup", "OUTER", v(2), v(1), m(1), 0.0, 0.0)

    # predict ops
    algo.add_instruction("predict", "NOOP", -1, -1, 0, 0.0, 0.0)
    algo.add_instruction("predict", "CONST", -1, -1, s(0), 0.0, 0.0)

    def add_scalar(src: int) -> None:
        algo.add_instruction("predict", "ADD", s(0), src, s(0), 0.0, 0.0)

    def add_vector(src: int) -> None:
        algo.add_instruction("predict", "DOT", src, v(4), s(6), 0.0, 0.0)
        add_scalar(s(6))

    def add_matrix(src: int) -> None:
        algo.add_instruction("predict", "COPY", src, -1, s(6), 0.0, 0.0)
        add_scalar(s(6))

    scalar_a = s(1)
    scalar_b = s(2)
    vector_a = v(1)
    vector_b = v(2)
    matrix_a = m(0)
    matrix_b = m(1)
    scalar_tmp = s(6)
    vector_tmp = v(5)
    matrix_tmp = m(2)

    arithmetic_ops = ["ADD", "SUB", "MUL", "DIV"]
    for op_name in arithmetic_ops:
        algo.add_instruction("predict", op_name, scalar_a, scalar_b, scalar_tmp)
        add_scalar(scalar_tmp)

        algo.add_instruction("predict", op_name, vector_a, vector_b, vector_tmp)
        add_vector(vector_tmp)
        algo.add_instruction("predict", op_name, scalar_a, vector_a, vector_tmp)
        add_vector(vector_tmp)
        algo.add_instruction("predict", op_name, vector_a, scalar_b, vector_tmp)
        add_vector(vector_tmp)

        algo.add_instruction("predict", op_name, matrix_a, matrix_b, matrix_tmp)
        add_matrix(matrix_tmp)
        algo.add_instruction("predict", op_name, scalar_a, matrix_a, matrix_tmp)
        add_matrix(matrix_tmp)
        algo.add_instruction("predict", op_name, matrix_a, scalar_b, matrix_tmp)
        add_matrix(matrix_tmp)
        algo.add_instruction("predict", op_name, vector_a, matrix_a, matrix_tmp)
        add_matrix(matrix_tmp)
        algo.add_instruction("predict", op_name, matrix_a, vector_b, matrix_tmp)
        add_matrix(matrix_tmp)

    unary_ops = ["ABS", "EXP", "LOG", "SIN", "COS", "TAN", "HEAVISIDE"]
    for op_name in unary_ops:
        algo.add_instruction("predict", op_name, scalar_a, -1, scalar_tmp)
        add_scalar(scalar_tmp)
        algo.add_instruction("predict", op_name, vector_a, -1, vector_tmp)
        add_vector(vector_tmp)
        algo.add_instruction("predict", op_name, matrix_a, -1, matrix_tmp)
        add_matrix(matrix_tmp)

    random_ops = [("GAUSSIAN", 0.25, 0.0), ("UNIFORM", -0.1, -0.1)]
    for op_name, c1, c2 in random_ops:
        algo.add_instruction("predict", op_name, -1, -1, scalar_tmp, c1, c2)
        add_scalar(scalar_tmp)
        algo.add_instruction("predict", op_name, -1, -1, vector_tmp, c1, c2)
        add_vector(vector_tmp)
        algo.add_instruction("predict", op_name, -1, -1, matrix_tmp, c1, c2)
        add_matrix(matrix_tmp)

    algo.add_instruction("predict", "DOT", vector_a, vector_b, scalar_tmp)
    add_scalar(scalar_tmp)
    algo.add_instruction("predict", "MATMUL", matrix_a, vector_a, vector_tmp)
    add_vector(vector_tmp)
    algo.add_instruction("predict", "OUTER", vector_a, vector_b, matrix_tmp)
    add_matrix(matrix_tmp)

    for op_name in ["NORM", "MEAN", "STD"]:
        algo.add_instruction("predict", op_name, vector_a, -1, scalar_tmp)
        add_scalar(scalar_tmp)
        algo.add_instruction("predict", op_name, vector_b, -1, vector_tmp)
        add_vector(vector_tmp)

    algo.add_instruction("predict", "COPY", scalar_a, -1, scalar_tmp)
    add_scalar(scalar_tmp)
    algo.add_instruction("predict", "COPY", vector_a, -1, scalar_tmp)
    add_scalar(scalar_tmp)
    algo.add_instruction("predict", "COPY", matrix_a, -1, scalar_tmp)
    add_scalar(scalar_tmp)

    return algo


def _run_predictions(
    task: DummyTask,
    algo: AlgorithmArray,
    *,
    shared: bool,
    numba: bool,
    numba_batch: bool,
    train_len: int = 3,
    val_len: int = 2,
):
    X_train = np.zeros((train_len, algo.input_dim), dtype=np.float32)
    y_train = np.zeros((train_len,), dtype=np.float32)
    X_val = np.zeros((val_len, algo.input_dim), dtype=np.float32)

    env = {
        "AUTOML_ZERO_SHARED_MEMORY": "1" if shared else "0",
        "AUTOML_ZERO_USE_NUMBA": "1" if numba else "0",
        "AUTOML_ZERO_USE_NUMBA_BATCH": "1" if numba_batch else "0",
    }

    with _set_env(**env):
        array_executor.get_exec_stats(reset=True)
        preds = task._predict_after_training(
            algo,
            X_train,
            y_train,
            X_val,
            epochs=1,
            rng_seed=42,
        )
        stats = array_executor.get_exec_stats(reset=True)

    return preds, stats


class ExecutorOpCoverageTests(unittest.TestCase):
    def setUp(self):
        self.algo = _build_full_coverage_algorithm()
        self.task = DummyTask(self.algo.input_dim)
        self.assertEqual(self.algo.validate_addresses(), [])
        self.assertEqual(self.algo.validate_semantics(), [])

    def test_all_ops_python_shared_and_batch(self):
        preds_batch, stats_batch = _run_predictions(
            self.task,
            self.algo,
            shared=False,
            numba=False,
            numba_batch=False,
        )
        self.assertEqual(preds_batch.shape, (2,))
        self.assertGreater(stats_batch.get("batch_numpy", 0), 0)
        self.assertEqual(stats_batch.get("batch_numba", 0), 0)

        preds_shared, stats_shared = _run_predictions(
            self.task,
            self.algo,
            shared=True,
            numba=False,
            numba_batch=False,
        )
        self.assertEqual(preds_shared.shape, (2,))
        self.assertGreater(stats_shared.get("shared_python", 0), 0)

        np.testing.assert_allclose(preds_shared, preds_batch, rtol=1e-4, atol=1e-5)

    @unittest.skipUnless(
        getattr(array_executor, "_NUMBA_AVAILABLE", False), "numba not available"
    )
    def test_all_ops_numba_shared_and_batch(self):
        baseline_preds, _ = _run_predictions(
            self.task,
            self.algo,
            shared=False,
            numba=False,
            numba_batch=False,
        )

        preds_batch, stats_batch = _run_predictions(
            self.task,
            self.algo,
            shared=False,
            numba=True,
            numba_batch=True,
        )
        self.assertGreater(stats_batch.get("batch_numba", 0), 0)
        self.assertEqual(stats_batch.get("batch_numpy", 0), 0)

        preds_shared, stats_shared = _run_predictions(
            self.task,
            self.algo,
            shared=True,
            numba=True,
            numba_batch=False,
        )
        self.assertGreater(stats_shared.get("shared_numba", 0), 0)

        np.testing.assert_allclose(preds_batch, baseline_preds, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(preds_shared, baseline_preds, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

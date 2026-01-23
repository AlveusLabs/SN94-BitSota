from abc import ABC, abstractmethod
import logging
from typing import Optional
import os

import numpy as np
from core.algorithm_array import AlgorithmArray
from core.array_executor import ArrayExecutor
from core.dsl_parser import DSLParser

logger = logging.getLogger(__name__)


class Task(ABC):
    """Abstract base class for all tasks"""

    def __init__(self, name: str, task_type: str):
        self.name = name
        self.task_type = task_type  # 'classification' or 'regression'
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.input_dim = None

    @abstractmethod
    def load_data(self, **kwargs):
        """Load and prepare the dataset"""
        pass

    @abstractmethod
    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate predictions and return a fitness score"""
        pass

    @abstractmethod
    def get_task_description(self) -> str:
        """Return a description of the task for LLM prompts"""
        pass

    def cache_descriptor(self):
        """Return a hashable descriptor for deterministic data; default disables caching."""
        return None

    def train_epoch(
        self, algo_array: AlgorithmArray, batch_size: int = 32
    ) -> ArrayExecutor:
        """Train for one epoch using vectorized execution"""
        executor = ArrayExecutor(algo_array)
        return executor

    def _predict_after_training(
        self,
        algo_array: AlgorithmArray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        *,
        epochs: int = 1,
        rng_seed: Optional[int] = None,
    ) -> np.ndarray:
        """Return validation predictions after training on the provided data."""
        if X_train is None or y_train is None or X_val is None:
            return np.array([], dtype=np.float32)
        try:
            executor = ArrayExecutor(algo_array, rng_seed=rng_seed)
            epochs = max(1, int(epochs))

            train_len = len(X_train)
            val_len = len(X_val)
            if train_len <= 0 or val_len <= 0:
                return np.array([], dtype=np.float32)

            input_dim = int(self.input_dim or X_train.shape[1])
            batch_size = max(train_len, 1)

            setup_input = np.zeros((batch_size, input_dim), dtype=np.float32)
            executor.execute_batch(setup_input, phases=["setup"], reset_state=True)

            train_input = np.zeros((batch_size, input_dim), dtype=np.float32)
            train_input[:train_len] = X_train
            train_labels = np.zeros(batch_size, dtype=np.float32)
            train_labels[:train_len] = y_train
            for _ in range(epochs):
                executor.execute_batch(
                    train_input,
                    y=train_labels,
                    phases=["predict", "learn"],
                    reset_state=False,
                )

            predictions_chunks = []
            for start in range(0, val_len, batch_size):
                end = min(start + batch_size, val_len)
                val_batch = np.zeros((batch_size, input_dim), dtype=np.float32)
                val_batch[: end - start] = X_val[start:end]
                pred_batch = executor.execute_batch(
                    val_batch, phases=["predict"], reset_state=False
                )
                predictions_chunks.append(pred_batch[: end - start])

            if not predictions_chunks:
                return np.array([], dtype=np.float32)

            return np.concatenate(predictions_chunks).astype(np.float32, copy=False)
        except Exception:
            logger.debug("Error during prediction trace", exc_info=True)
            return np.array([], dtype=np.float32)

    def evaluate_algorithm(
        self,
        algo_array: AlgorithmArray,
        epochs: int = 1,
        rng_seed: Optional[int] = None,
        **kwargs,
    ) -> float:
        """Evaluate an algorithm on this task using batched predict/learn loops."""
        try:
            if (
                self.X_train is None
                or self.y_train is None
                or self.X_val is None
                or self.y_val is None
            ):
                raise RuntimeError("Task data must be loaded before evaluation")

            executor = ArrayExecutor(algo_array, rng_seed=rng_seed)
            epochs = max(1, int(epochs))

            train_len = len(self.X_train)
            val_len = len(self.X_val)
            if train_len <= 0 or val_len <= 0:
                return self.get_baseline_fitness()

            use_shared_memory = str(
                os.getenv("AUTOML_ZERO_SHARED_MEMORY", "1")
            ).strip().lower() in {"1", "true", "yes", "on"}

            if use_shared_memory:
                batch_size = 1

                setup_input = np.zeros((batch_size, self.input_dim), dtype=np.float32)
                executor.execute_batch(setup_input, phases=["setup"], reset_state=True)

                train_input = np.zeros((batch_size, self.input_dim), dtype=np.float32)
                train_labels = np.zeros(batch_size, dtype=np.float32)
                for _ in range(epochs):
                    for idx in range(train_len):
                        train_input[0] = self.X_train[idx]
                        train_labels[0] = self.y_train[idx]
                        executor.execute_batch(
                            train_input,
                            y=train_labels,
                            phases=["predict", "learn"],
                            reset_state=False,
                        )

                predictions = np.zeros(val_len, dtype=np.float32)
                val_input = np.zeros((batch_size, self.input_dim), dtype=np.float32)
                for idx in range(val_len):
                    val_input[0] = self.X_val[idx]
                    pred_batch = executor.execute_batch(
                        val_input, phases=["predict"], reset_state=False
                    )
                    predictions[idx] = pred_batch[0]

                labels = self.y_val[: len(predictions)]
                if len(predictions) == 0 or len(labels) == 0:
                    return self.get_baseline_fitness()

                return self.evaluate(predictions, labels)

            batch_size = max(train_len, 1)

            # Run setup once with the chosen batch size so state persists.
            setup_input = np.zeros((batch_size, self.input_dim), dtype=np.float32)
            executor.execute_batch(setup_input, phases=["setup"], reset_state=True)

            # Batched training: full train set per epoch.
            train_input = np.zeros((batch_size, self.input_dim), dtype=np.float32)
            train_input[:train_len] = self.X_train
            train_labels = np.zeros(batch_size, dtype=np.float32)
            train_labels[:train_len] = self.y_train
            for _ in range(epochs):
                executor.execute_batch(
                    train_input,
                    y=train_labels,
                    phases=["predict", "learn"],
                    reset_state=False,
                )

            # Batched validation without resetting state; pad to keep batch size stable.
            predictions_chunks = []
            for start in range(0, val_len, batch_size):
                end = min(start + batch_size, val_len)
                val_batch = np.zeros((batch_size, self.input_dim), dtype=np.float32)
                val_batch[: end - start] = self.X_val[start:end]
                pred_batch = executor.execute_batch(
                    val_batch, phases=["predict"], reset_state=False
                )
                predictions_chunks.append(pred_batch[: end - start])

            if not predictions_chunks:
                return self.get_baseline_fitness()

            predictions = np.concatenate(predictions_chunks).astype(np.float32, copy=False)
            labels = self.y_val[: len(predictions)]
            if len(predictions) == 0 or len(labels) == 0:
                return self.get_baseline_fitness()

            return self.evaluate(predictions, labels)

        except Exception as e:
            logger.debug("Error during evaluation", exc_info=True)
            return self.get_baseline_fitness()

    @abstractmethod
    def get_baseline_fitness(self) -> float:
        """Return the baseline fitness (e.g., random guessing)"""
        pass

    def create_initial_algorithm(self) -> AlgorithmArray:
        if self.task_type == "classification":
            dsl = """
# setup
m0 = gaussian(0, 0.1)
s10 = 0.01

# predict
s0 = uniform(0, 1)

# learn

"""
        else:
            dsl = """
# setup
m0 = gaussian(0, 0.1)
s10 = 0.01

# predict
s0 = uniform(-1, 1)

# learn

"""

        return DSLParser.from_dsl(dsl, self.input_dim)

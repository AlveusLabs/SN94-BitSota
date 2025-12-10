from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .base import Task


DEFAULT_FEATURE_DIM = 16
DEFAULT_TRAIN_SAMPLES = 1000
DEFAULT_VAL_SAMPLES = 1000

DEFAULT_PARAM_SEEDS: List[int] = [
    1001,
    1012,
    1010,
    1000,
    1006,
    1008,
    1007,
    1003,
]

DEFAULT_DATA_SEEDS: List[int] = [
    11001,
    11012,
    11010,
    11000,
    11006,
    11008,
    11007,
    11003,
]


@dataclass(frozen=True)
class ScalarLinearTaskSpec:
    task_id: int
    param_seed: int
    data_seed: int
    n_features: int = DEFAULT_FEATURE_DIM
    train_samples: int = DEFAULT_TRAIN_SAMPLES
    val_samples: int = DEFAULT_VAL_SAMPLES


def _seed_for_index(index: int, base_seeds: List[int]) -> int:
    if index < len(base_seeds):
        return base_seeds[index]
    extra = index - len(base_seeds) + 1
    return base_seeds[-1] + extra


class ScalarLinearRegressionTask(Task):
    """AutoML-Zero style scalar linear regression task."""

    def __init__(self, seed_offset: int = 0):
        super().__init__("ScalarLinearRegression", "regression")
        self._rng = np.random.default_rng(seed_offset)
        self.current_task_spec: Optional[ScalarLinearTaskSpec] = None
        self._train_samples = DEFAULT_TRAIN_SAMPLES
        self._val_samples = DEFAULT_VAL_SAMPLES

    @classmethod
    def get_task_spec(
        cls,
        task_id: int,
        n_features: int = DEFAULT_FEATURE_DIM,
        train_samples: int = DEFAULT_TRAIN_SAMPLES,
        val_samples: int = DEFAULT_VAL_SAMPLES,
    ) -> ScalarLinearTaskSpec:
        if task_id < 0:
            raise ValueError("task_id must be non-negative")
        param_seed = _seed_for_index(task_id, DEFAULT_PARAM_SEEDS)
        data_seed = _seed_for_index(task_id, DEFAULT_DATA_SEEDS)
        return ScalarLinearTaskSpec(
            task_id=task_id,
            param_seed=param_seed,
            data_seed=data_seed,
            n_features=n_features,
            train_samples=train_samples,
            val_samples=val_samples,
        )

    @classmethod
    def sample_task_specs(
        cls,
        num_tasks: int = 1,
        *,
        replace: bool = True,
        seed: Optional[int] = None,
        n_features: int = DEFAULT_FEATURE_DIM,
        train_samples: int = DEFAULT_TRAIN_SAMPLES,
        val_samples: int = DEFAULT_VAL_SAMPLES,
    ):
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        rng = np.random.default_rng(seed)
        max_len = max(len(DEFAULT_PARAM_SEEDS), len(DEFAULT_DATA_SEEDS))
        if replace:
            indices = rng.integers(0, max_len, size=num_tasks)
        else:
            indices = rng.choice(max_len, size=num_tasks, replace=False)
        return [
            cls.get_task_spec(
                int(task_id),
                n_features=n_features,
                train_samples=train_samples,
                val_samples=val_samples,
            )
            for task_id in np.atleast_1d(indices)
        ]

    def _sample_random_task_spec(
        self,
        n_features: int,
        train_samples: int,
        val_samples: int,
    ) -> ScalarLinearTaskSpec:
        task_id = int(
            self._rng.integers(
                0,
                max(len(DEFAULT_PARAM_SEEDS), len(DEFAULT_DATA_SEEDS)),
            )
        )
        return self.get_task_spec(
            task_id,
            n_features=n_features,
            train_samples=train_samples,
            val_samples=val_samples,
        )

    def sample_miner_task_spec(
        self,
        input_dim: int,
        *,
        seed: Optional[int] = None,
        train_samples: Optional[int] = None,
        val_samples: Optional[int] = None,
    ) -> ScalarLinearTaskSpec:
        """Sample a deterministic task spec for miner evaluations."""

        ts = train_samples or self._train_samples or DEFAULT_TRAIN_SAMPLES
        vs = val_samples or self._val_samples or DEFAULT_VAL_SAMPLES
        if seed is not None:
            rng = np.random.default_rng(seed)
            task_id = int(
                rng.integers(
                    0,
                    max(len(DEFAULT_PARAM_SEEDS), len(DEFAULT_DATA_SEEDS)),
                )
            )
            return self.get_task_spec(
                task_id,
                n_features=input_dim,
                train_samples=ts,
                val_samples=vs,
            )
        return self._sample_random_task_spec(
            n_features=input_dim,
            train_samples=ts,
            val_samples=vs,
        )

    def load_data(
        self,
        *,
        task_spec: Optional[ScalarLinearTaskSpec] = None,
        task_id: Optional[int] = None,
        param_seed: Optional[int] = None,
        data_seed: Optional[int] = None,
        n_features: Optional[int] = None,
        train_samples: Optional[int] = None,
        val_samples: Optional[int] = None,
    ):
        if task_spec is None:
            feat_dim = n_features or self.input_dim or DEFAULT_FEATURE_DIM
            train_count = train_samples or self._train_samples or DEFAULT_TRAIN_SAMPLES
            val_count = val_samples or self._val_samples or DEFAULT_VAL_SAMPLES
            if task_id is not None:
                task_spec = self.get_task_spec(
                    task_id,
                    n_features=feat_dim,
                    train_samples=train_count,
                    val_samples=val_count,
                )
            else:
                param = param_seed if param_seed is not None else int(self._rng.integers(1, 2**31 - 1))
                data = data_seed if data_seed is not None else int(self._rng.integers(1, 2**31 - 1))
                task_spec = ScalarLinearTaskSpec(
                    task_id=-1,
                    param_seed=param,
                    data_seed=data,
                    n_features=feat_dim,
                    train_samples=train_count,
                    val_samples=val_count,
                )

        self.current_task_spec = task_spec
        self._train_samples = task_spec.train_samples
        self._val_samples = task_spec.val_samples

        data_rng = np.random.default_rng(task_spec.data_seed)
        X_train = data_rng.standard_normal(
            (task_spec.train_samples, task_spec.n_features)
        )
        X_val = data_rng.standard_normal(
            (task_spec.val_samples, task_spec.n_features)
        )

        weight_rng = np.random.default_rng(task_spec.param_seed)
        weights = weight_rng.standard_normal(task_spec.n_features)

        y_train = X_train @ weights
        y_val = X_val @ weights

        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train.astype(np.float32)
        self.X_val = X_val.astype(np.float32)
        self.y_val = y_val.astype(np.float32)
        self.input_dim = task_spec.n_features

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        if not np.all(np.isfinite(predictions)):
            return -np.inf
        mse = np.mean((predictions - labels) ** 2)
        if not np.isfinite(mse):
            return -np.inf
        rmse = float(np.sqrt(mse))
        # Match AutoML-Zero's RMS error squashing: 1 - (2/pi) * atan(rmse)
        fitness = 1.0 - (2.0 / np.pi) * np.arctan(rmse)
        return float(fitness)

    def get_task_description(self) -> str:
        return f"Scalar linear regression ({self.input_dim}D inputs) - Gaussian features and weights"

    def get_baseline_fitness(self) -> float:
        return -np.inf

    def cache_descriptor(self):
        if self.current_task_spec is None:
            return None
        ts = self.current_task_spec
        return (
            "scalar_linear",
            ts.param_seed,
            ts.data_seed,
            ts.train_samples,
            ts.val_samples,
            ts.n_features,
        )

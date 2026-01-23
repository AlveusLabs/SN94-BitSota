import logging
import os
import ssl
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import fetch_openml

from .base import Task


logger = logging.getLogger(__name__)

DEFAULT_COMPONENTS = 16
DEFAULT_SAMPLE_COUNT = 2000
DEFAULT_TRAIN_SPLIT = 0.5
TOTAL_TASKS = 10_000

DATA_SEED_OFFSET = 21001
PROJECTION_SEED_OFFSET = 17345
SPLIT_SEED_OFFSET = 97531

_MNIST_01_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None

try:
    _MNIST_TASK_CACHE_MAXSIZE = max(0, int(os.getenv("MNIST_TASK_CACHE_MAXSIZE", "256")))
except Exception:
    _MNIST_TASK_CACHE_MAXSIZE = 256


def _fetch_mnist_01() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch (and cache) MNIST digits 0/1 as float32 + int32 arrays."""

    global _MNIST_01_CACHE
    if _MNIST_01_CACHE is not None:
        return _MNIST_01_CACHE

    logger.info("Loading MNIST dataset...")
    original_context = ssl._create_default_https_context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    finally:
        ssl._create_default_https_context = original_context

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)

    mask = (y == 0) | (y == 1)
    X_01 = X[mask]
    y_01 = y[mask]
    _MNIST_01_CACHE = (X_01, y_01)
    logger.info("MNIST loaded successfully (0/1 samples=%d)", int(len(X_01)))
    return _MNIST_01_CACHE


def _build_projection(seed: int, input_dim: int, n_components: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((input_dim, n_components)).astype(np.float32)
    norms = np.linalg.norm(proj, axis=0, keepdims=True)
    proj /= np.clip(norms, 1e-6, None)
    return proj


def _standardize_features(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)


@dataclass(frozen=True)
class MNISTBinaryTaskSpec:
    task_id: int
    data_seed: int
    projection_seed: int
    split_seed: int
    train_split: float = DEFAULT_TRAIN_SPLIT
    n_components: int = DEFAULT_COMPONENTS
    n_samples: int = DEFAULT_SAMPLE_COUNT

    def describe(self) -> str:
        return f"task {self.task_id} :: projection={self.projection_seed} split={self.split_seed}"


@lru_cache(maxsize=_MNIST_TASK_CACHE_MAXSIZE)
def _cached_task_arrays(task_spec: MNISTBinaryTaskSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_all, y_all = _fetch_mnist_01()
    if len(X_all) == 0:
        raise RuntimeError("MNIST dataset returned no samples for digits 0/1")

    count = max(1, int(task_spec.n_samples))
    rng = np.random.default_rng(int(task_spec.data_seed))
    replace = count > len(X_all)
    indices = rng.choice(len(X_all), size=count, replace=replace)
    X = X_all[indices]
    y = y_all[indices]

    y_binary = (y == 1).astype(np.float32)

    proj = _build_projection(int(task_spec.projection_seed), X.shape[1], int(task_spec.n_components))
    X_proj = _standardize_features(X @ proj)

    split_rng = np.random.default_rng(int(task_spec.split_seed))
    perm = split_rng.permutation(len(X_proj))
    n_train = max(1, int(float(task_spec.train_split) * len(X_proj)))
    n_train = min(n_train, len(X_proj) - 1) if len(X_proj) > 1 else 1
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    X_train = X_proj[train_idx]
    y_train = y_binary[train_idx]
    X_val = X_proj[val_idx]
    y_val = y_binary[val_idx]

    for arr in (X_train, y_train, X_val, y_val):
        try:
            arr.setflags(write=False)
        except Exception:
            pass

    return X_train, y_train, X_val, y_val


class MNISTBinaryTask(Task):
    """Binary classification task on MNIST (digit 0 vs 1)."""

    def __init__(self, sampler_seed: Optional[int] = None):
        super().__init__("MNISTBinary", "classification")
        self._rng = np.random.default_rng(sampler_seed)
        self.current_task_spec: Optional[MNISTBinaryTaskSpec] = None
        self._n_samples = DEFAULT_SAMPLE_COUNT
        self._train_split = DEFAULT_TRAIN_SPLIT

    @staticmethod
    def total_tasks() -> int:
        return TOTAL_TASKS

    @classmethod
    def get_task_spec(
        cls,
        task_id: int,
        n_components: int = DEFAULT_COMPONENTS,
        n_samples: int = DEFAULT_SAMPLE_COUNT,
        train_split: float = DEFAULT_TRAIN_SPLIT,
    ) -> MNISTBinaryTaskSpec:
        if task_id < 0:
            raise ValueError("task_id must be non-negative")
        return MNISTBinaryTaskSpec(
            task_id=int(task_id),
            data_seed=DATA_SEED_OFFSET + int(task_id),
            projection_seed=PROJECTION_SEED_OFFSET + int(task_id),
            split_seed=SPLIT_SEED_OFFSET + int(task_id),
            train_split=float(train_split),
            n_components=int(n_components),
            n_samples=int(n_samples),
        )

    @classmethod
    def sample_task_specs(
        cls,
        num_tasks: int = 1,
        *,
        replace: bool = True,
        seed: Optional[int] = None,
        n_components: int = DEFAULT_COMPONENTS,
        n_samples: int = DEFAULT_SAMPLE_COUNT,
        train_split: float = DEFAULT_TRAIN_SPLIT,
    ) -> List[MNISTBinaryTaskSpec]:
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        rng = np.random.default_rng(seed)
        effective_replace = replace or num_tasks > TOTAL_TASKS
        task_ids = rng.choice(TOTAL_TASKS, size=num_tasks, replace=effective_replace)
        return [
            cls.get_task_spec(
                int(task_id),
                n_components=n_components,
                n_samples=n_samples,
                train_split=train_split,
            )
            for task_id in np.atleast_1d(task_ids)
        ]

    def _sample_random_task_spec(
        self,
        *,
        n_components: int,
        n_samples: int,
        train_split: float,
    ) -> MNISTBinaryTaskSpec:
        task_id = int(self._rng.integers(0, TOTAL_TASKS))
        return self.get_task_spec(
            task_id,
            n_components=n_components,
            n_samples=n_samples,
            train_split=train_split,
        )

    def sample_miner_task_spec(
        self,
        input_dim: int,
        *,
        seed: Optional[int] = None,
        n_samples: Optional[int] = None,
        train_split: Optional[float] = None,
    ) -> MNISTBinaryTaskSpec:
        """Sample a deterministic task spec for miner evaluations."""

        ns = int(n_samples or self._n_samples or DEFAULT_SAMPLE_COUNT)
        ts = float(train_split or self._train_split or DEFAULT_TRAIN_SPLIT)
        if seed is not None:
            rng = np.random.default_rng(int(seed))
            task_id = int(rng.integers(0, TOTAL_TASKS))
            return self.get_task_spec(
                task_id,
                n_components=int(input_dim),
                n_samples=ns,
                train_split=ts,
            )
        return self._sample_random_task_spec(
            n_components=int(input_dim),
            n_samples=ns,
            train_split=ts,
        )

    def load_data(
        self,
        n_components: int = DEFAULT_COMPONENTS,
        n_samples: int = DEFAULT_SAMPLE_COUNT,
        train_split: float = DEFAULT_TRAIN_SPLIT,
        *,
        task_spec: Optional[MNISTBinaryTaskSpec] = None,
        task_id: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        if task_spec is not None and task_id is not None:
            raise ValueError("Provide either task_spec or task_id, not both")

        if task_spec is None:
            if task_id is not None:
                task_spec = self.get_task_spec(
                    int(task_id),
                    n_components=n_components,
                    n_samples=n_samples,
                    train_split=train_split,
                )
            else:
                if rng_seed is not None:
                    rng = np.random.default_rng(int(rng_seed))
                    sampled_id = int(rng.integers(0, TOTAL_TASKS))
                    task_spec = self.get_task_spec(
                        sampled_id,
                        n_components=n_components,
                        n_samples=n_samples,
                        train_split=train_split,
                    )
                else:
                    task_spec = self._sample_random_task_spec(
                        n_components=n_components,
                        n_samples=n_samples,
                        train_split=train_split,
                    )

        self.current_task_spec = task_spec
        self._n_samples = int(task_spec.n_samples)
        self._train_split = float(task_spec.train_split)

        self.X_train, self.y_train, self.X_val, self.y_val = _cached_task_arrays(task_spec)
        self.input_dim = int(task_spec.n_components)
        logger.debug("MNIST task %s loaded (train_split=%.2f)", task_spec.describe(), float(task_spec.train_split))

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        pred_classes = (predictions > 0.5).astype(int)
        return float(np.mean(pred_classes == labels))

    def get_task_description(self) -> str:
        return f"Binary classification ({self.input_dim}D inputs) - MNIST digit 0 vs 1"

    def get_baseline_fitness(self) -> float:
        return -np.inf

    def cache_descriptor(self):
        if self.current_task_spec is None:
            return None
        ts = self.current_task_spec
        return (
            "mnist_binary",
            ts.task_id,
            ts.data_seed,
            ts.projection_seed,
            ts.split_seed,
            ts.n_components,
            ts.n_samples,
            ts.train_split,
        )

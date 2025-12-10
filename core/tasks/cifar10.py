import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import floating
from sklearn.datasets import fetch_openml

from .base import Task


logger = logging.getLogger(__name__)


CLASS_LABELS: Tuple[int, ...] = tuple(range(10))
CLASS_PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    (i, j) for i in CLASS_LABELS for j in CLASS_LABELS if j > i
)
PROJECTIONS_PER_PAIR = 100
TOTAL_TASKS = len(CLASS_PAIRS) * PROJECTIONS_PER_PAIR
DEFAULT_TRAIN_SPLIT = 0.5
DEFAULT_COMPONENTS = 16
DEFAULT_SAMPLE_COUNT = 2000
PROJECTION_SEED_OFFSET = 17345
SPLIT_SEED_OFFSET = 97531

_CIFAR_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


@dataclass(frozen=True)
class CIFAR10TaskSpec:
    """Immutable description of a CIFAR-10 binary projection task."""

    task_id: int
    class_pair: Tuple[int, int]
    projection_seed: int
    split_seed: int
    train_split: float = DEFAULT_TRAIN_SPLIT
    n_components: int = DEFAULT_COMPONENTS
    n_samples: int = DEFAULT_SAMPLE_COUNT

    def describe(self) -> str:
        idx_within_pair = self.task_id % PROJECTIONS_PER_PAIR
        return (
            f"task {self.task_id} :: classes {self.class_pair[0]} vs {self.class_pair[1]}"
            f" :: projection {idx_within_pair}"
        )


def _fetch_cifar_subset(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch (and cache) a subset of CIFAR-10."""

    cache = _CIFAR_CACHE.get(n_samples)
    if cache is not None:
        return cache

    cifar = fetch_openml("CIFAR_10_small", version=1, as_frame=False, parser="auto")
    X = cifar.data[:n_samples].astype(np.float32) / 255.0
    y = cifar.target[:n_samples].astype(np.int32)
    _CIFAR_CACHE[n_samples] = (X, y)
    return X, y


def _build_projection(seed: int, input_dim: int, n_components: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((input_dim, n_components)).astype(np.float32)
    norms = np.linalg.norm(proj, axis=0, keepdims=True)
    proj /= np.clip(norms, 1e-6, None)
    return proj


def _standardize_features(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)


class CIFAR10BinaryTask(Task):

    def __init__(self, sampler_seed: Optional[int] = None):
        super().__init__("CIFAR10Binary", "classification")
        self._rng = np.random.default_rng(sampler_seed)
        self.current_task_spec: Optional[CIFAR10TaskSpec] = None
        self._n_samples = DEFAULT_SAMPLE_COUNT
        self._train_split = DEFAULT_TRAIN_SPLIT

    # ------------------------------------------------------------------
    # Task sampling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def total_tasks() -> int:
        return TOTAL_TASKS

    @staticmethod
    def _pair_for_task(task_id: int) -> Tuple[int, int]:
        if task_id < 0 or task_id >= TOTAL_TASKS:
            raise ValueError(f"task_id must be in [0, {TOTAL_TASKS}), got {task_id}")
        pair_idx = task_id // PROJECTIONS_PER_PAIR
        return CLASS_PAIRS[pair_idx]

    @classmethod
    def get_task_spec(
        cls,
        task_id: int,
        n_components: int = DEFAULT_COMPONENTS,
        n_samples: int = DEFAULT_SAMPLE_COUNT,
        train_split: float = DEFAULT_TRAIN_SPLIT,
    ) -> CIFAR10TaskSpec:
        class_pair = cls._pair_for_task(task_id)
        return CIFAR10TaskSpec(
            task_id=task_id,
            class_pair=class_pair,
            projection_seed=PROJECTION_SEED_OFFSET + task_id,
            split_seed=SPLIT_SEED_OFFSET + task_id,
            train_split=train_split,
            n_components=n_components,
            n_samples=n_samples,
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
    ) -> List[CIFAR10TaskSpec]:
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        rng = np.random.default_rng(seed)
        effective_replace = replace or num_tasks > TOTAL_TASKS
        task_indices = rng.choice(TOTAL_TASKS, size=num_tasks, replace=effective_replace)
        return [
            cls.get_task_spec(
                int(task_id),
                n_components=n_components,
                n_samples=n_samples,
                train_split=train_split,
            )
            for task_id in np.atleast_1d(task_indices)
        ]

    def _sample_random_task_spec(
        self,
        n_components: int,
        n_samples: int,
        train_split: float,
    ) -> CIFAR10TaskSpec:
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
    ) -> CIFAR10TaskSpec:
        """Sample a task spec suitable for miner evaluations."""

        params = {
            "n_components": input_dim,
            "n_samples": self._n_samples,
            "train_split": self._train_split,
        }
        if seed is not None:
            rng = np.random.default_rng(seed)
            task_id = int(rng.integers(0, TOTAL_TASKS))
            return self.get_task_spec(task_id, **params)
        return self._sample_random_task_spec(**params)

    # ------------------------------------------------------------------
    # Data loading + evaluation
    # ------------------------------------------------------------------
    def load_data(
        self,
        n_components: int = DEFAULT_COMPONENTS,
        n_samples: int = DEFAULT_SAMPLE_COUNT,
        train_split: float = DEFAULT_TRAIN_SPLIT,
        *,
        task_spec: Optional[CIFAR10TaskSpec] = None,
        task_id: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ):
        if task_spec is not None and task_id is not None:
            raise ValueError("Provide either task_spec or task_id, not both")

        if task_spec is None:
            if task_id is not None:
                task_spec = self.get_task_spec(
                    task_id,
                    n_components=n_components,
                    n_samples=n_samples,
                    train_split=train_split,
                )
            else:
                # Allow deterministic sampling via rng_seed when requested
                if rng_seed is not None:
                    rng = np.random.default_rng(rng_seed)
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
        n_components = task_spec.n_components
        n_samples = task_spec.n_samples
        train_split = task_spec.train_split
        self._n_samples = n_samples
        self._train_split = train_split

        class_a, class_b = task_spec.class_pair
        X, y = _fetch_cifar_subset(n_samples)
        mask = (y == class_a) | (y == class_b)
        X_pair = X[mask]
        y_pair = y[mask]
        if len(X_pair) == 0:
            raise RuntimeError(
                f"No samples found for class pair {class_a}-{class_b} with n_samples={n_samples}"
            )

        y_binary = (y_pair == class_b).astype(np.float32)
        proj_matrix = _build_projection(task_spec.projection_seed, X_pair.shape[1], n_components)
        X_proj = _standardize_features(X_pair @ proj_matrix)

        split_rng = np.random.default_rng(task_spec.split_seed)
        indices = split_rng.permutation(len(X_proj))
        n_train = max(1, int(train_split * len(X_proj)))
        n_train = min(n_train, len(X_proj) - 1) if len(X_proj) > 1 else 1
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        self.X_train = X_proj[train_idx]
        self.y_train = y_binary[train_idx]
        self.X_val = X_proj[val_idx]
        self.y_val = y_binary[val_idx]
        self.input_dim = n_components

        logger.debug(
            "CIFAR10 task %s: %s -> %s",
            task_spec.describe(),
            X_pair.shape,
            X_proj.shape,
        )
        logger.debug(
            "   class balance %.2f (train_split=%.2f)",
            y_binary.mean(),
            train_split,
        )

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> floating[Any]:
        pred_classes = (predictions > 0.5).astype(int)
        return np.mean(pred_classes == labels)

    def get_task_description(self) -> str:
        if self.current_task_spec:
            a, b = self.current_task_spec.class_pair
            return (
                f"Binary classification ({self.input_dim}D inputs) - class {a} vs {b}"
            )
        return f"Binary classification ({self.input_dim}D inputs) - CIFAR10 pair"

    def get_baseline_fitness(self) -> float:
        return -np.inf

    def cache_descriptor(self):
        if self.current_task_spec is None:
            return None
        ts = self.current_task_spec
        return (
            "cifar10_binary",
            ts.task_id,
            ts.class_pair,
            ts.projection_seed,
            ts.split_seed,
            ts.n_components,
            ts.n_samples,
            ts.train_split,
        )

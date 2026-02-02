import hashlib
import logging
import os
import pickle
import ssl
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import floating
from sklearn.datasets import fetch_openml
import urllib.request

from .base import Task


logger = logging.getLogger(__name__)

CIFAR10_EXPECTED_HASH = "8bc18551ba7a01c82b49249d78537c0f08d25136988375435dc5e01397852a90"


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

_CIFAR_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None
try:
    _CIFAR_TASK_CACHE_MAXSIZE = max(0, int(os.getenv("CIFAR_TASK_CACHE_MAXSIZE", "512")))
except Exception:
    _CIFAR_TASK_CACHE_MAXSIZE = 512


@lru_cache(maxsize=_CIFAR_TASK_CACHE_MAXSIZE)
def _cached_task_arrays(task_spec: "CIFAR10TaskSpec") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache fully prepared train/val splits for deterministic task specs."""

    class_a, class_b = task_spec.class_pair
    X, y = _fetch_cifar()
    mask = (y == class_a) | (y == class_b)
    X_pair_all = X[mask]
    y_pair_all = y[mask]
    if len(X_pair_all) == 0:
        raise RuntimeError(
            f"No samples found for class pair {class_a}-{class_b} with n_samples={task_spec.n_samples}"
        )

    requested = max(1, int(task_spec.n_samples))
    sample_rng = np.random.default_rng(int(task_spec.split_seed))
    replace = requested > len(X_pair_all)
    sample_indices = sample_rng.choice(len(X_pair_all), size=requested, replace=replace)
    X_pair = X_pair_all[sample_indices]
    y_pair = y_pair_all[sample_indices]

    y_binary = (y_pair == class_b).astype(np.float32)
    proj_matrix = _build_projection(
        task_spec.projection_seed, X_pair.shape[1], task_spec.n_components
    )
    X_proj = _standardize_features(X_pair @ proj_matrix)

    split_rng = np.random.default_rng(int(task_spec.split_seed) + 1)
    indices = split_rng.permutation(len(X_proj))
    n_train = max(1, int(task_spec.train_split * len(X_proj)))
    n_train = min(n_train, len(X_proj) - 1) if len(X_proj) > 1 else 1
    train_idx, val_idx = indices[:n_train], indices[n_train:]

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


def _get_sklearn_data_home() -> str:
    """Get the sklearn data directory."""
    from pathlib import Path
    data_home = os.environ.get('SCIKIT_LEARN_DATA')
    if data_home:
        return data_home
    return str(Path.home() / 'scikit_learn_data')

def _ensure_cifar_cached() -> None:
    """Ensure CIFAR-10 ARFF file is cached locally, downloading from custom URL if needed."""
    custom_url = os.getenv("CIFAR10_DATASET_URL")

    if not custom_url:
        try:
            from gui.app_config import get_app_config
            config = get_app_config()
            custom_url = config.cifar10_dataset_url
        except Exception:
            pass

    if not custom_url:
        return

    from pathlib import Path
    data_home = Path(_get_sklearn_data_home())
    cache_dir = data_home / "openml" / "openml.org" / "data" / "v1" / "download" / "16797612"
    cache_file = cache_dir / "CIFAR_10_small.arff.gz"

    if cache_file.exists():
        logger.debug("CIFAR-10 already cached locally")
        return

    try:
        logger.info(f"Downloading CIFAR-10 from {custom_url}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        context = ssl._create_unverified_context()
        with urllib.request.urlopen(custom_url, context=context, timeout=120) as response:
            data = response.read()

        file_hash = hashlib.sha256(data).hexdigest()
        if file_hash != CIFAR10_EXPECTED_HASH:
            logger.error(f"CIFAR-10 hash mismatch! Expected {CIFAR10_EXPECTED_HASH}, got {file_hash}")
            raise ValueError("CIFAR-10 dataset hash verification failed - file may be corrupted or tampered")

        with open(cache_file, 'wb') as f:
            f.write(data)

        logger.info(f"CIFAR-10 cached successfully at {cache_file} (hash verified)")
    except Exception as e:
        logger.warning(f"Failed to download CIFAR-10 from custom URL: {e}")
        if cache_file.exists():
            cache_file.unlink()

def _fetch_cifar() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch (and cache) CIFAR-10."""

    global _CIFAR_CACHE
    cache = _CIFAR_CACHE
    if cache is not None:
        return cache

    _ensure_cifar_cached()

    logger.info("Loading CIFAR-10 dataset...")
    original_context = ssl._create_default_https_context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        cifar = fetch_openml("CIFAR_10_small", version=1, as_frame=False, parser="auto")
    finally:
        ssl._create_default_https_context = original_context

    X = cifar.data.astype(np.float32) / 255.0
    y = cifar.target.astype(np.int32)
    _CIFAR_CACHE = (X, y)
    logger.info("CIFAR-10 loaded successfully")
    return _CIFAR_CACHE


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

        self.X_train, self.y_train, self.X_val, self.y_val = _cached_task_arrays(task_spec)
        self.input_dim = n_components
        logger.debug("CIFAR10 task %s loaded (train_split=%.2f)", task_spec.describe(), train_split)

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

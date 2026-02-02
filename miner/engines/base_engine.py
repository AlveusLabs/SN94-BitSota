import hashlib
import os
from collections import OrderedDict
from typing import Tuple, List, Optional, Dict

import numpy as np

from core.algorithm_array import AlgorithmArray
from core.hyperparams import get_miner_hyperparams
from core.tasks.base import Task
from core.tasks.cifar10 import CIFAR10BinaryTask


_DEFAULT_MINER_HP = get_miner_hyperparams()
DEFAULT_MINER_TASK_COUNT = int(_DEFAULT_MINER_HP.miner_task_count)
DEFAULT_MINER_TASK_SEED = int(_DEFAULT_MINER_HP.miner_task_seed)
DEFAULT_FEC_CACHE_SIZE = int(_DEFAULT_MINER_HP.fec_cache_size)
DEFAULT_FEC_TRAIN_EXAMPLES = int(_DEFAULT_MINER_HP.fec_train_examples)
DEFAULT_FEC_VALID_EXAMPLES = int(_DEFAULT_MINER_HP.fec_valid_examples)
DEFAULT_FEC_FORGET_EVERY = int(_DEFAULT_MINER_HP.fec_forget_every)
_FEC_PROBE_SEED = 1337
_FEC_TRACE_DECIMALS = 3

FECProbeData = Tuple[
    "Task", object, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


class BaseEvolutionEngine:
    """Base class for evolution engines using the new array-based format"""

    def __init__(
        self,
        task: Task,
        pop_size: int = 8,
        verbose: bool = False,
        miner_task_count: Optional[int] = None,
        phase_max_sizes: Optional[Dict[str, int]] = None,
        scalar_count: Optional[int] = None,
        vector_count: Optional[int] = None,
        matrix_count: Optional[int] = None,
        vector_dim: Optional[int] = None,
        fec_cache_size: Optional[int] = None,
        fec_train_examples: Optional[int] = None,
        fec_valid_examples: Optional[int] = None,
        fec_forget_every: Optional[int] = None,
        cifar_seed: Optional[int] = None,
    ):
        self.task = task
        self.pop_size = pop_size
        self.verbose = verbose
        self.population = None
        self.best_algo = None
        self.best_fitness = -np.inf
        self.generation = 0
        self.miner_task_count = max(1, miner_task_count or DEFAULT_MINER_TASK_COUNT)
        self._phase_order = ["setup", "predict", "learn"]
        default_sizes = {phase: 64 for phase in self._phase_order}
        if phase_max_sizes:
            default_sizes.update(phase_max_sizes)
        self._default_phase_sizes = default_sizes
        self._scalar_count = scalar_count
        self._vector_count = vector_count
        self._matrix_count = matrix_count
        self._vector_dim = vector_dim
        if fec_cache_size is None:
            fec_cache_size = DEFAULT_FEC_CACHE_SIZE
        if fec_train_examples is None:
            fec_train_examples = DEFAULT_FEC_TRAIN_EXAMPLES
        if fec_valid_examples is None:
            fec_valid_examples = DEFAULT_FEC_VALID_EXAMPLES
        if fec_forget_every is None:
            fec_forget_every = DEFAULT_FEC_FORGET_EVERY

        self._fec_cache_size = max(0, int(fec_cache_size))
        self._fec_train_examples = max(1, int(fec_train_examples))
        self._fec_valid_examples = max(1, int(fec_valid_examples))
        self._fec_forget_every = max(0, int(fec_forget_every))
        self._fec_cache: OrderedDict = OrderedDict()
        self._fec_probe_cache: Dict[int, FECProbeData] = {}
        self._fec_inserts = 0
        self._fec_lookups = 0
        self._fec_hits = 0
        self._fec_misses = 0
        self._fec_key_failures = 0
        self._fec_probe_seed = _FEC_PROBE_SEED
        self._fec_trace_decimals = _FEC_TRACE_DECIMALS
        self._cifar_seed = DEFAULT_MINER_TASK_SEED if cifar_seed is None else int(cifar_seed)
        self._fixed_task_specs_by_dim: Dict[int, List[object]] = {}

    def initialize_population(self) -> List[AlgorithmArray]:
        """Initialize the population - to be overridden by subclasses"""
        population = []
        for _ in range(self.pop_size):
            algo = self.create_initial_algorithm()
            population.append(algo)
        return population

    def evolve_generation(
        self,
    ) -> Tuple[AlgorithmArray, float, List[AlgorithmArray], List[float]]:
        """
        Evolve a single generation and return results.

        Returns:
            Tuple of (best_algo, best_score, population, scores)
        """
        raise NotImplementedError("Subclasses must implement evolve_generation method")

    def evolve(self, generations: int) -> Tuple[AlgorithmArray, float]:
        """
        Run evolution for multiple generations.
        Uses evolve_generation internally for backward compatibility.
        """
        for _ in range(generations):
            self.evolve_generation()

        return self.best_algo, self.best_fitness

    def create_initial_algorithm(self) -> AlgorithmArray:
        """Create an empty algorithm with predefined phase budgets."""

        input_dim = getattr(self.task, "input_dim", None)
        if input_dim is None:
            raise ValueError("Task input dimension must be set before initialization")

        default_vector_dim = self._vector_dim
        if default_vector_dim is None:
            try:
                default_vector_dim = int(input_dim)
            except Exception:
                default_vector_dim = None

        return AlgorithmArray.create_empty(
            input_dim=input_dim,
            phases=self._phase_order,
            max_sizes=self._default_phase_sizes,
            scalar_count=self._scalar_count,
            vector_count=self._vector_count,
            matrix_count=self._matrix_count,
            vector_dim=default_vector_dim,
        )

    def _get_fixed_miner_task_specs(self, input_dim: int):
        """Return the fixed miner task suite for a given input dimension."""

        input_dim = int(input_dim)
        specs = self._fixed_task_specs_by_dim.get(input_dim)
        if specs is not None:
            return specs

        if not isinstance(self.task, CIFAR10BinaryTask):
            self._fixed_task_specs_by_dim[input_dim] = []
            return self._fixed_task_specs_by_dim[input_dim]

        # Deterministic per (seed, input_dim) so every genome is scored on the same task suite.
        seed = int(self._cifar_seed) + 7919 * max(1, input_dim)
        rng = np.random.default_rng(seed)
        specs = []
        for _ in range(self.miner_task_count):
            task_seed = int(rng.integers(0, 2**31 - 1))
            specs.append(self.task.sample_miner_task_spec(input_dim, seed=task_seed))

        self._fixed_task_specs_by_dim[input_dim] = specs
        return specs

    def _get_fec_task_descriptor(self, task: Task, input_dim: int):
        descriptor = None
        try:
            descriptor = task.cache_descriptor()
        except Exception:
            descriptor = None
        if descriptor is None:
            name = getattr(task, "name", type(task).__name__)
            task_type = getattr(task, "task_type", None)
            try:
                dim = int(input_dim)
            except Exception:
                dim = None
            descriptor = (name, task_type, dim)
        return descriptor

    def _select_fec_subset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if X is None or y is None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        n = len(X)
        if n <= 0:
            return X[:0], y[:0]
        if count >= n:
            return X, y
        indices = rng.choice(n, size=count, replace=False)
        return X[indices], y[indices]

    def _get_fec_probe_data(self, input_dim: int) -> Optional[FECProbeData]:
        try:
            input_dim = int(input_dim)
        except Exception:
            return None

        cached = self._fec_probe_cache.get(input_dim)
        if cached is not None:
            return cached

        probe_task: Optional[Task] = None
        if isinstance(self.task, CIFAR10BinaryTask):
            task_specs = self._get_fixed_miner_task_specs(input_dim)
            if not task_specs:
                return None
            probe_spec = task_specs[0]
            probe_task = CIFAR10BinaryTask()
            try:
                if hasattr(self.task, "_n_samples"):
                    probe_task._n_samples = self.task._n_samples
                if hasattr(self.task, "_train_split"):
                    probe_task._train_split = self.task._train_split
            except Exception:
                pass
            probe_task.load_data(task_spec=probe_spec)
        else:
            probe_task = self.task

        if (
            probe_task is None
            or probe_task.X_train is None
            or probe_task.y_train is None
            or probe_task.X_val is None
            or probe_task.y_val is None
        ):
            return None

        rng_train = np.random.default_rng(self._fec_probe_seed)
        rng_val = np.random.default_rng(self._fec_probe_seed + 1)
        X_train, y_train = self._select_fec_subset(
            probe_task.X_train,
            probe_task.y_train,
            self._fec_train_examples,
            rng_train,
        )
        X_val, y_val = self._select_fec_subset(
            probe_task.X_val,
            probe_task.y_val,
            self._fec_valid_examples,
            rng_val,
        )

        descriptor = self._get_fec_task_descriptor(probe_task, input_dim)
        if descriptor is None:
            return None

        data = (probe_task, descriptor, X_train, y_train, X_val, y_val)
        self._fec_probe_cache[input_dim] = data
        return data

    def _compute_fec_trace(
        self,
        probe_task: Task,
        algo: AlgorithmArray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Optional[np.ndarray]:
        try:
            predictions = probe_task._predict_after_training(
                algo,
                X_train,
                y_train,
                X_val,
                epochs=1,
                rng_seed=self._fec_probe_seed,
            )
        except Exception:
            return None

        if predictions is None or len(predictions) == 0:
            return None

        labels = None
        if y_val is not None:
            labels = y_val[: len(predictions)]
        if labels is None or len(labels) == 0:
            trace = predictions
        else:
            trace = predictions - labels

        trace = np.round(trace, decimals=self._fec_trace_decimals)
        trace = np.nan_to_num(
            trace.astype(np.float32, copy=False),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        return np.ascontiguousarray(trace, dtype=np.float32)

    def _hash_fec_trace(
        self,
        trace: np.ndarray,
        descriptor: object,
        train_count: int,
        valid_count: int,
        input_dim: int,
    ) -> str:
        hasher = hashlib.sha1()
        hasher.update(b"fec-v1")
        hasher.update(repr(descriptor).encode("utf-8"))
        for value in (
            int(train_count),
            int(valid_count),
            int(input_dim),
            int(self.miner_task_count),
            int(self._cifar_seed),
            int(self._fec_probe_seed),
            int(self._fec_trace_decimals),
        ):
            hasher.update(int(value).to_bytes(8, "little", signed=True))
        hasher.update(trace.tobytes())
        return hasher.hexdigest()

    def _evaluate_on_miner_tasks(self, algo: AlgorithmArray) -> float:
        """Evaluate an algorithm across multiple sampled tasks and return median fitness."""

        cache_key = self._make_cache_key(algo)
        if cache_key is not None:
            cached = self._fec_cache.get(cache_key)
            if cached is not None:
                self._fec_hits += 1
                # LRU bump
                self._fec_cache.move_to_end(cache_key)
                return cached
            self._fec_misses += 1

        scores = []
        if isinstance(self.task, CIFAR10BinaryTask):
            task_specs = self._get_fixed_miner_task_specs(algo.input_dim)
            for spec in task_specs:
                self.task.load_data(task_spec=spec)
                scores.append(self.task.evaluate_algorithm(algo))
        else:
            for _ in range(self.miner_task_count):
                scores.append(self.task.evaluate_algorithm(algo))

        if not scores:
            return -np.inf

        finite_scores = [s for s in scores if np.isfinite(s)]
        if not finite_scores:
            return -np.inf

        median_score = float(np.median(finite_scores))

        if cache_key is not None:
            if self._fec_forget_every > 0 and self._fec_inserts >= self._fec_forget_every:
                self._fec_cache.clear()
                self._fec_inserts = 0
            self._fec_cache[cache_key] = median_score
            self._fec_cache.move_to_end(cache_key)
            self._fec_inserts += 1
            while len(self._fec_cache) > self._fec_cache_size > 0:
                self._fec_cache.popitem(last=False)

        return median_score

    def _make_cache_key(self, algo: AlgorithmArray):
        if self._fec_cache_size <= 0:
            return None
        self._fec_lookups += 1
        probe_data = self._get_fec_probe_data(algo.input_dim)
        if not probe_data:
            self._fec_key_failures += 1
            return None
        probe_task, descriptor, X_train, y_train, X_val, y_val = probe_data
        trace = self._compute_fec_trace(probe_task, algo, X_train, y_train, X_val, y_val)
        if trace is None:
            self._fec_key_failures += 1
            return None
        return self._hash_fec_trace(
            trace,
            descriptor,
            len(X_train),
            int(trace.shape[0]),
            algo.input_dim,
        )

    def get_fec_stats(self) -> Dict[str, int]:
        return {
            "lookups": int(self._fec_lookups),
            "hits": int(self._fec_hits),
            "misses": int(self._fec_misses),
            "key_failures": int(self._fec_key_failures),
            "size": int(len(self._fec_cache)),
            "capacity": int(self._fec_cache_size),
        }

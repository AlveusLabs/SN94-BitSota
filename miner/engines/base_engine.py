import os
from collections import OrderedDict
from typing import Tuple, List, Optional, Dict

import numpy as np

from core.algorithm_array import AlgorithmArray
from core.tasks.base import Task
from core.tasks.cifar10 import CIFAR10BinaryTask


DEFAULT_MINER_TASK_COUNT = int(os.getenv("MINER_TASK_COUNT", "32"))
DEFAULT_MINER_TASK_SEED = int(os.getenv("MINER_TASK_SEED", "0"))


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
        fec_cache_size: int = 0,
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
        self._fec_cache_size = max(0, int(fec_cache_size))
        self._fec_cache: OrderedDict = OrderedDict()
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

        return AlgorithmArray.create_empty(
            input_dim=input_dim,
            phases=self._phase_order,
            max_sizes=self._default_phase_sizes,
            scalar_count=self._scalar_count,
            vector_count=self._vector_count,
            matrix_count=self._matrix_count,
            vector_dim=self._vector_dim,
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

    def _evaluate_on_miner_tasks(self, algo: AlgorithmArray) -> float:
        """Evaluate an algorithm across multiple sampled tasks and return median fitness."""

        cache_key = self._make_cache_key(algo)
        if cache_key is not None:
            cached = self._fec_cache.get(cache_key)
            if cached is not None:
                # LRU bump
                self._fec_cache.move_to_end(cache_key)
                return cached

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
            self._fec_cache[cache_key] = median_score
            self._fec_cache.move_to_end(cache_key)
            while len(self._fec_cache) > self._fec_cache_size > 0:
                self._fec_cache.popitem(last=False)

        return median_score

    def _make_cache_key(self, algo: AlgorithmArray):
        if self._fec_cache_size <= 0:
            return None
        descriptor = None
        try:
            descriptor = self.task.cache_descriptor()
        except Exception:
            descriptor = None
        if descriptor is None:
            return None
        try:
            fingerprint = algo.fingerprint()
        except Exception:
            return None
        return (fingerprint, descriptor, self.miner_task_count)

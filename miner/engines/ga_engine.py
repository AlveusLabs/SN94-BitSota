from collections import deque
from copy import deepcopy
from typing import Deque, Dict, List, Tuple, Optional

import numpy as np

from core.algorithm_array import AlgorithmArray
from .base_engine import BaseEvolutionEngine


class BaselineEvolutionEngine(BaseEvolutionEngine):
    """Regularized evolution (aging evolution) baseline engine."""

    def __init__(
        self,
        task,
        pop_size: int = 100,
        verbose: bool = False,
        miner_task_count: Optional[int] = None,
        tournament_size: int = 10,
        mutation_prob: float = 0.9,
        phase_max_sizes: Optional[Dict[str, int]] = None,
        scalar_count: Optional[int] = None,
        vector_count: Optional[int] = None,
        matrix_count: Optional[int] = None,
        vector_dim: Optional[int] = None,
        fec_cache_size: int = 0,
        cifar_seed: Optional[int] = None,
    ):
        super().__init__(
            task,
            pop_size=pop_size,
            verbose=verbose,
            miner_task_count=miner_task_count,
            phase_max_sizes=phase_max_sizes,
            scalar_count=scalar_count,
            vector_count=vector_count,
            matrix_count=matrix_count,
            vector_dim=vector_dim,
            fec_cache_size=fec_cache_size,
            cifar_seed=cifar_seed,
        )
        self.tournament_size = max(1, min(pop_size, tournament_size))
        self.mutation_prob = np.clip(mutation_prob, 0.0, 1.0)
        self._population_queue: Optional[Deque[Dict[str, object]]] = None
        self._stagnation = 0

    # ------------------------------------------------------------------
    # Core regularized evolution loop
    # ------------------------------------------------------------------
    def evolve_generation(
        self,
    ) -> Tuple[AlgorithmArray, float, List[AlgorithmArray], List[float]]:
        """Run a single aging-evolution cycle."""

        if self._population_queue is None:
            self._initialize_population()

        # Aging step: remove oldest individual from the queue
        oldest = self._population_queue.popleft()

        # Tournament selection among remaining individuals
        parent_entry = self._select_tournament_parent(default_entry=oldest)
        parent_algo = parent_entry["algo"]

        # Child creation + mutation (with probability U)
        child = deepcopy(parent_algo)
        if np.random.rand() < self.mutation_prob:
            self._apply_regularized_mutation(child)

        # Evaluate child on D tasks (median fitness)
        try:
            child_fitness = self._evaluate_on_miner_tasks(child)
        except Exception:
            child_fitness = -np.inf

        # FIFO insertion keeps queue size == pop_size
        self._population_queue.append({"algo": child, "fitness": child_fitness})

        # Track best-so-far
        if child_fitness > self.best_fitness or self.best_algo is None:
            self.best_fitness = child_fitness
            self.best_algo = child
            self._stagnation = 0
        else:
            self._stagnation += 1

        # Prepare outputs for external callers
        self.population = [entry["algo"] for entry in self._population_queue]
        scores = [entry["fitness"] for entry in self._population_queue]

        success_count = np.sum(np.isfinite(scores))
        failure_count = len(scores) - success_count

        if self.verbose:
            queue_best = np.max(scores) if scores else -np.inf
            queue_mean = float(np.mean(scores)) if scores else -np.inf
            print(
                f"[regularized-evo] pop={len(self.population)} best={self.best_fitness:.4f} "
                f"queue_best={queue_best:.4f} queue_mean={queue_mean:.4f} "
                f"stagnation={self._stagnation} success={success_count} failed={failure_count}"
            )

        self.generation += 1
        return self.best_algo, self.best_fitness, self.population, scores

    # ------------------------------------------------------------------
    # Population management helpers
    # ------------------------------------------------------------------
    def _initialize_population(self) -> None:
        """Create the initial FIFO population of empty programs."""

        self._population_queue = deque(maxlen=self.pop_size)
        for _ in range(self.pop_size):
            algo = self.create_initial_algorithm()
            try:
                fitness = self._evaluate_on_miner_tasks(algo)
            except Exception:
                fitness = -np.inf

            self._population_queue.append({"algo": algo, "fitness": fitness})
            if self.best_algo is None or fitness > self.best_fitness:
                self.best_algo = algo
                self.best_fitness = fitness

        self.population = [entry["algo"] for entry in self._population_queue]

    def _select_tournament_parent(self, default_entry: Dict[str, object]):
        """Sample T individuals uniformly and return the best fitness entry."""

        assert self._population_queue is not None
        queue_list = list(self._population_queue)

        if not queue_list:
            return default_entry

        sample_size = min(self.tournament_size, len(queue_list))
        if sample_size <= 0:
            return default_entry

        indices = np.random.choice(len(queue_list), size=sample_size, replace=False)
        candidates = [queue_list[idx] for idx in indices]
        parent_entry = max(candidates, key=lambda entry: entry["fitness"])
        return parent_entry

    # ------------------------------------------------------------------
    # Mutation operators (per AutoML-Zero spec)
    # ------------------------------------------------------------------
    def _apply_regularized_mutation(self, algo: AlgorithmArray) -> None:
        mutation_type = np.random.choice(["edit", "randomize", "tweak"])

        if mutation_type == "edit":
            self._mutate_insert_or_remove(algo)
        elif mutation_type == "randomize":
            self._randomize_component(algo)
        else:
            self._tweak_instruction(algo)

    def _mutate_insert_or_remove(self, algo: AlgorithmArray) -> None:
        phases = algo.get_phases()
        if not phases:
            return

        phase = np.random.choice(phases)
        phase_size = algo.get_phase_size(phase)

        # Insert when empty or coin flip, otherwise delete
        if phase_size == 0 or np.random.rand() < 0.5:
            self._add_random_instruction(algo, phase)
        elif phase_size > 0:
            idx = np.random.randint(phase_size)
            self._remove_instruction(algo, phase, idx)

    def _randomize_component(self, algo: AlgorithmArray) -> None:
        phases = algo.get_phases()
        if not phases:
            return

        phase = np.random.choice(phases)
        algo.phase_sizes[phase] = 0

        max_size = algo.phase_max_sizes.get(phase, 1)
        target_size = max(1, np.random.randint(1, max_size + 1))
        for _ in range(target_size):
            self._add_random_instruction(algo, phase)

    def _tweak_instruction(self, algo: AlgorithmArray) -> None:
        phases = [phase for phase in algo.get_phases() if algo.get_phase_size(phase) > 0]
        if not phases:
            # No instructions yet, fall back to insertion
            self._mutate_insert_or_remove(algo)
            return

        phase = np.random.choice(phases)
        idx = np.random.randint(algo.get_phase_size(phase))
        self._mutate_instruction_components(algo, phase, idx)

    # ------------------------------------------------------------------
    # Instruction-level helpers (adapted from legacy GA engine)
    # ------------------------------------------------------------------
    def _add_random_instruction(self, algo: AlgorithmArray, phase: str) -> None:
        from core.algorithm_array import (
            OPCODE_METADATA,
            ADDR_SCALARS,
            ADDR_VECTORS,
            ADDR_MATRICES,
        )

        max_size = algo.phase_max_sizes.get(phase, 0)
        current_size = algo.get_phase_size(phase)
        if current_size >= max_size > 0:
            return  # Phase is full

        op_name = np.random.choice(list(OPCODE_METADATA.keys()))
        op_meta = OPCODE_METADATA[op_name]

        s_start = ADDR_SCALARS
        s_end = ADDR_SCALARS + algo.scalar_count
        v_start = ADDR_VECTORS
        v_end = ADDR_VECTORS + algo.vector_count
        m_start = ADDR_MATRICES
        m_end = ADDR_MATRICES + algo.matrix_count

        def sample_addr(kind: str) -> int:
            if kind == "s":
                return np.random.randint(s_start, s_end) if algo.scalar_count > 0 else s_start
            if kind == "v":
                return np.random.randint(v_start, v_end) if algo.vector_count > 0 else v_start
            if kind == "m":
                return np.random.randint(m_start, m_end) if algo.matrix_count > 0 else m_start
            return 0

        arg1 = sample_addr(op_meta["arg1"])
        arg2 = sample_addr(op_meta["arg2"])
        dest = sample_addr(op_meta["dest"])

        const1 = np.random.uniform(-2.0, 2.0)
        const2 = np.random.uniform(-2.0, 2.0)

        algo.add_instruction(phase, op_name, arg1, arg2, dest, const1, const2)

    def _mutate_instruction_components(
        self, algo: AlgorithmArray, phase: str, idx: int
    ) -> None:
        from core.algorithm_array import (
            OPCODES,
            OPCODE_METADATA,
            ADDR_SCALARS,
            ADDR_VECTORS,
            ADDR_MATRICES,
        )

        arrays = algo.phase_arrays[phase]
        mutation_type = np.random.choice(["opcode", "address", "constant"])

        if mutation_type == "opcode":
            op_name = np.random.choice(list(OPCODE_METADATA.keys()))
            arrays["ops"][idx] = OPCODES[op_name]
            # Recursively tidy addresses/constants for new opcode
            self._mutate_instruction_components(algo, phase, idx)
            return

        if mutation_type == "address":
            op_code = arrays["ops"][idx]
            op_name = next((name for name, code in OPCODES.items() if code == op_code), None)
            if not op_name or op_name not in OPCODE_METADATA:
                return

            op_meta = OPCODE_METADATA[op_name]
            addr_field = np.random.choice(["arg1", "arg2", "dest"])

            s_start = ADDR_SCALARS
            s_end = ADDR_SCALARS + algo.scalar_count
            v_start = ADDR_VECTORS
            v_end = ADDR_VECTORS + algo.vector_count
            m_start = ADDR_MATRICES
            m_end = ADDR_MATRICES + algo.matrix_count

            def sample_addr(kind: str) -> int:
                if kind == "s":
                    return np.random.randint(s_start, s_end) if algo.scalar_count > 0 else s_start
                if kind == "v":
                    return np.random.randint(v_start, v_end) if algo.vector_count > 0 else v_start
                if kind == "m":
                    return np.random.randint(m_start, m_end) if algo.matrix_count > 0 else m_start
                return 0

            if op_meta[addr_field] != "none":
                arrays[addr_field][idx] = sample_addr(op_meta[addr_field])
        else:
            const_field = np.random.choice(["const1", "const2"])
            arrays[const_field][idx] = np.random.uniform(-2.0, 2.0)

    def _remove_instruction(self, algo: AlgorithmArray, phase: str, idx: int) -> None:
        arrays = algo.phase_arrays[phase]
        current_size = algo.phase_sizes[phase]
        if idx < 0 or idx >= current_size:
            return

        for field in ["ops", "arg1", "arg2", "dest", "const1", "const2"]:
            arr = arrays[field]
            arr[idx: current_size - 1] = arr[idx + 1 : current_size]
            arr[current_size - 1] = 0

        algo.phase_sizes[phase] -= 1

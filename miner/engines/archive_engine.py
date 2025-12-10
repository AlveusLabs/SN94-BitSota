from collections import deque
from copy import deepcopy
from typing import List, Tuple, Optional, Dict

import numpy as np

from core.algorithm_array import AlgorithmArray
from core.tasks.base import Task
from .base_engine import BaseEvolutionEngine


class ArchiveAwareBaselineEvolution(BaseEvolutionEngine):
    """
    Extends baseline evolution with an archive of all discovered algorithms.
    Transitions from simple population-based to open-ended evolution by sampling
    parents from the entire history weighted by fitness.
    """

    def __init__(
        self,
        task: Task,
        pop_size: int = 8,
        archive_size: int = 1000,
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
        super().__init__(
            task,
            pop_size,
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
        self.archive = deque(maxlen=archive_size)
        self.generation_counter = 0
        self.archive_diversity = {}
        self.min_archive_for_sampling = 20

    def add_to_archive(self, algo: AlgorithmArray, fitness: float):
        """
        Add algorithm to archive with metadata.

        Args:
            algo: Algorithm to archive
            fitness: Fitness score of the algorithm
        """
        # Calculate complexity as total instruction count
        complexity = sum(algo.get_phase_size(p) for p in algo.get_phases())

        # Create a signature for diversity tracking
        signature = self._get_algorithm_signature(algo)

        # Store in archive
        self.archive.append(
            {
                "algorithm": deepcopy(algo),
                "fitness": fitness,
                "generation": self.generation_counter,
                "complexity": complexity,
                "signature": signature,
            }
        )

        # Update diversity tracking
        if signature not in self.archive_diversity:
            self.archive_diversity[signature] = 0
        self.archive_diversity[signature] += 1

    def _get_algorithm_signature(self, algo: AlgorithmArray) -> str:
        """
        Generate a simplified signature for diversity tracking using direct array analysis.

        Args:
            algo: Algorithm to analyze

        Returns:
            String signature representing algorithm structure
        """
        from core.algorithm_array import OPCODES

        # Reverse OPCODES mapping for opcode to name
        opcode_to_name = {v: k for k, v in OPCODES.items()}

        sig_parts = []

        for phase_name in algo.get_phases():
            ops, _, _, _, _, _ = algo.get_phase_ops(phase_name)
            op_counts = {}

            for op_code in ops:
                if op_code in opcode_to_name:
                    op_name = opcode_to_name[op_code]
                    op_counts[op_name] = op_counts.get(op_name, 0) + 1

            # Create sorted signature for this phase
            phase_sig = f"{phase_name}:" + ",".join(
                f"{k}{v}" for k, v in sorted(op_counts.items())
            )
            sig_parts.append(phase_sig)

        return "|".join(sig_parts)

    def _random_mutate(self, algo: AlgorithmArray) -> AlgorithmArray:
        """Apply mutations directly to AlgorithmArray arrays without DSL conversion"""
        import copy

        new_algo = copy.deepcopy(algo)

        # Determine available phases for mutation
        phases = algo.get_phases()
        if not phases:
            return new_algo  # Cannot mutate an algorithm with no phases

        # Pick phase to mutate
        phase = np.random.choice(phases)

        # Determine mutation type
        phase_size = new_algo.get_phase_size(phase)
        mutation_type = np.random.choice(["add", "modify", "remove"], p=[0.5, 0.4, 0.1])

        if mutation_type == "add" or phase_size == 0:
            # Add new instruction
            self._add_random_instruction(new_algo, phase)

        elif mutation_type == "modify" and phase_size > 0:
            # Modify existing instruction
            idx = np.random.randint(phase_size)
            self._mutate_instruction_components(new_algo, phase, idx)

        elif mutation_type == "remove" and phase_size > 1:
            # Remove instruction
            idx = np.random.randint(phase_size)
            self._remove_instruction(new_algo, phase, idx)

        return new_algo

    def _add_random_instruction(self, algo: AlgorithmArray, phase: str) -> None:
        """Smarter instruction addition/replacement with type constraints"""
        from core.algorithm_array import (
            OPCODES,
            OPCODE_METADATA,
            ADDR_SCALARS,
            ADDR_VECTORS,
            ADDR_MATRICES,
        )

        op_name = np.random.choice(list(OPCODE_METADATA.keys()))
        op_meta = OPCODE_METADATA[op_name]

        s_start = ADDR_SCALARS
        s_end = ADDR_SCALARS + algo.scalar_count
        v_start = ADDR_VECTORS
        v_end = ADDR_VECTORS + algo.vector_count
        m_start = ADDR_MATRICES
        m_end = ADDR_MATRICES + algo.matrix_count

        def get_addr(addr_type):
            if addr_type == "s":
                return np.random.randint(s_start, s_end) if algo.scalar_count > 0 else s_start
            if addr_type == "v":
                return np.random.randint(v_start, v_end) if algo.vector_count > 0 else v_start
            if addr_type == "m":
                return np.random.randint(m_start, m_end) if algo.matrix_count > 0 else m_start
            return 0

        arg1 = get_addr(op_meta["arg1"])
        arg2 = get_addr(op_meta["arg2"])
        dest = get_addr(op_meta["dest"])

        const1 = np.random.uniform(-2.0, 2.0)
        const2 = np.random.uniform(-2.0, 2.0)

        # Check if phase is full
        current_size = algo.get_phase_size(phase)
        max_size = algo.get_phase_max_size(phase)

        if current_size < max_size:
            # Phase has space, add new instruction
            algo.add_instruction(phase, op_name, arg1, arg2, dest, const1, const2)
        elif current_size > 0:
            # Phase is full, replace a random existing instruction
            idx = np.random.randint(0, current_size)
            arrays = algo.phase_arrays[phase]

            arrays["ops"][idx] = OPCODES[op_name]
            arrays["arg1"][idx] = arg1
            arrays["arg2"][idx] = arg2
            arrays["dest"][idx] = dest
            arrays["const1"][idx] = const1
            arrays["const2"][idx] = const2

    def _mutate_instruction_components(
        self, algo: AlgorithmArray, phase: str, idx: int
    ) -> None:
        """Smarter component mutation with type constraints"""
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
            # After changing opcode, it's good practice to also fix addresses
            self._mutate_instruction_components(algo, phase, idx)

        elif mutation_type == "address":
            op_code = arrays["ops"][idx]
            op_name = next(
                (name for name, code in OPCODES.items() if code == op_code), None
            )
            if not op_name or op_name not in OPCODE_METADATA:
                return

            op_meta = OPCODE_METADATA[op_name]
            address_type = np.random.choice(["arg1", "arg2", "dest"])

            s_start = ADDR_SCALARS
            s_end = ADDR_SCALARS + algo.scalar_count
            v_start = ADDR_VECTORS
            v_end = ADDR_VECTORS + algo.vector_count
            m_start = ADDR_MATRICES
            m_end = ADDR_MATRICES + algo.matrix_count

            def get_addr(addr_type_code):
                if addr_type_code == "s":
                    return np.random.randint(s_start, s_end) if algo.scalar_count > 0 else s_start
                if addr_type_code == "v":
                    return np.random.randint(v_start, v_end) if algo.vector_count > 0 else v_start
                if addr_type_code == "m":
                    return np.random.randint(m_start, m_end) if algo.matrix_count > 0 else m_start
                return 0

            if op_meta[address_type] != "none":
                new_addr = get_addr(op_meta[address_type])
                arrays[address_type][idx] = new_addr

        else:  # constant
            constant_type = np.random.choice(["const1", "const2"])
            new_value = np.random.uniform(-2.0, 2.0)
            arrays[constant_type][idx] = new_value

    def _remove_instruction(self, algo: AlgorithmArray, phase: str, idx: int) -> None:
        """Remove instruction and shift arrays"""
        arrays = algo.phase_arrays[phase]
        current_size = algo.phase_sizes[phase]

        # Shift arrays left by 1 position
        for array_name in ["ops", "arg1", "arg2", "dest", "const1", "const2"]:
            arr = arrays[array_name]
            arr[idx:-1] = arr[idx + 1 :]
            arr[-1] = 0  # Clear last element

        algo.phase_sizes[phase] -= 1

    def sample_parent_from_archive(self) -> AlgorithmArray:
        """
        Sample a parent algorithm from archive weighted by fitness.

        Returns:
            Selected parent algorithm
        """
        if len(self.archive) == 0:
            # Archive is empty, create a new random algorithm
            return self.create_initial_algorithm()

        if len(self.archive) < self.min_archive_for_sampling:
            # Not enough history, use uniform sampling
            idx = np.random.randint(len(self.archive))
            return deepcopy(self.archive[idx]["algorithm"])

        # Calculate sampling weights based on fitness
        fitnesses = np.array([entry["fitness"] for entry in self.archive])

        # Handle case where all fitnesses are -inf
        if np.all(np.isinf(fitnesses)):
            # All algorithms failed, use uniform sampling
            idx = np.random.randint(len(self.archive))
            return deepcopy(self.archive[idx]["algorithm"])

        # Apply temperature scaling to balance exploration/exploitation
        temperature = 2.0  # Higher = more uniform, lower = more greedy

        # Handle NaN/inf values in fitnesses
        valid_fitnesses = fitnesses[~np.isinf(fitnesses)]
        if len(valid_fitnesses) > 0:
            min_fitness = np.min(valid_fitnesses)
            max_fitness = np.max(valid_fitnesses)

            # Normalize fitnesses to avoid extreme values
            if max_fitness > min_fitness:
                normalized_fitnesses = (fitnesses - min_fitness) / (
                    max_fitness - min_fitness
                )
            else:
                normalized_fitnesses = np.zeros_like(fitnesses)
        else:
            normalized_fitnesses = np.zeros_like(fitnesses)

        weights = np.exp((normalized_fitnesses) / temperature)

        # Handle any remaining NaN/inf values
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=0.0)

        # Bonus weight for diversity (favor less common signatures)
        diversity_bonus = np.array(
            [
                1.0 / (1 + self.archive_diversity.get(entry["signature"], 1))
                for entry in self.archive
            ]
        )

        # Combine fitness and diversity
        weights = weights * (1 + 0.3 * diversity_bonus)

        # Ensure all weights are positive
        weights = np.maximum(weights, 1e-8)

        # Normalize
        weights = weights / weights.sum()

        # Sample
        idx = np.random.choice(len(self.archive), p=weights)
        return deepcopy(self.archive[idx]["algorithm"])

    def crossover(
        self, parent1: AlgorithmArray, parent2: AlgorithmArray
    ) -> AlgorithmArray:
        """
        Perform crossover between two parent algorithms using direct array manipulation.

        Args:
            parent1: First parent algorithm
            parent2: Second parent algorithm

        Returns:
            Child algorithm combining elements from both parents
        """
        import numpy as np

        # Create child as a deep copy of parent1
        child = deepcopy(parent1)

        # Perform crossover for each phase
        for phase in child.get_phases():
            if phase not in parent2.phase_arrays:
                continue

            p1_size = parent1.get_phase_size(phase)
            p2_size = parent2.get_phase_size(phase)

            if p1_size == 0 or p2_size == 0:
                continue

            # Ensure we have a valid range for crossover
            min_size = min(p1_size, p2_size)
            if min_size <= 1:
                # Not enough instructions for meaningful crossover
                # Randomly choose to keep parent1 or switch to parent2
                if np.random.rand() < 0.5:
                    # Keep parent1 (already copied)
                    continue
                else:
                    # Switch to parent2 for this phase
                    for array_name in [
                        "ops",
                        "arg1",
                        "arg2",
                        "dest",
                        "const1",
                        "const2",
                    ]:
                        child.phase_arrays[phase][array_name][:p2_size] = (
                            parent2.phase_arrays[phase][array_name][:p2_size]
                        )
                    child.phase_sizes[phase] = p2_size
                continue

            # One-point crossover with valid range
            crossover_point = np.random.randint(1, min_size)

            # Copy instructions from parent2 after crossover point
            new_size = max(crossover_point, p2_size)
            for array_name in ["ops", "arg1", "arg2", "dest", "const1", "const2"]:
                child.phase_arrays[phase][array_name][crossover_point:new_size] = (
                    parent2.phase_arrays[phase][array_name][crossover_point:p2_size]
                )

            # Adjust size of the child's phase
            child.phase_sizes[phase] = new_size

        return child

    def initialize_population(self) -> List[AlgorithmArray]:
        """Initialize population with empty programs by default."""
        return super().initialize_population()

    def evolve_generation(
        self,
    ) -> Tuple[AlgorithmArray, float, List[AlgorithmArray], List[float]]:
        """Evolve a single generation with archive updates"""
        # Initialize if first generation
        if self.population is None:
            self.population = self.initialize_population()
            self.best_fitness = self.task.get_baseline_fitness()

        # Evaluate population
        pop_fitness = []
        scores = []
        success_count = 0
        failure_count = 0

        for algo in self.population:
            if not algo.validate_semantics():
                try:
                    fitness = self._evaluate_on_miner_tasks(algo)
                    pop_fitness.append((algo, fitness))
                    scores.append(fitness)
                    success_count += 1
                except Exception:
                    pop_fitness.append((algo, -np.inf))
                    scores.append(-np.inf)
                    failure_count += 1
            else:
                pop_fitness.append((algo, -np.inf))
                scores.append(-np.inf)
                failure_count += 1

        pop_fitness.sort(key=lambda x: x[1], reverse=True)

        # Add all to archive
        for algo, fit in pop_fitness:
            self.add_to_archive(algo, fit)

        # Track best
        if pop_fitness and pop_fitness[0][1] > self.best_fitness:
            self.best_fitness = pop_fitness[0][1]
            self.best_algo = pop_fitness[0][0]

        # Print progress
        fitnesses = [f for _, f in pop_fitness if f != -np.inf]
        avg_fitness = np.mean(fitnesses) if fitnesses else -np.inf
        unique_sigs = len(set(entry["signature"] for entry in self.archive))

        if self.verbose:
            print(
                f"archive gen {self.generation}: best={self.best_fitness:.4f}, avg={avg_fitness:.4f}, "
                f"archive_size={len(self.archive)}, unique_designs={unique_sigs}, "
                f"success={success_count}, failed={failure_count}"
            )

        # Build next generation
        new_population = [pop_fitness[0][0]]  # Elitism

        while len(new_population) < self.pop_size:
            # Choose evolution strategy
            strategy = np.random.choice(
                ["mutate_archive", "mutate_current", "crossover"], p=[0.4, 0.3, 0.3]
            )

            if strategy == "mutate_archive" and len(self.archive) > 0:
                parent = self.sample_parent_from_archive()
                child = self._random_mutate(parent)

            elif strategy == "mutate_current":
                parent = pop_fitness[np.random.randint(self.pop_size // 2)][0]
                child = self._random_mutate(parent)

            else:  # strategy == 'crossover' and len(self.archive) >= 2:
                parent1 = self.sample_parent_from_archive()
                parent2 = self.sample_parent_from_archive()
                child = self.crossover(parent1, parent2)
                if np.random.rand() < 0.3:
                    child = self._random_mutate(child)

            new_population.append(child)

        self.population = new_population
        self.generation += 1
        self.generation_counter = self.generation

        return self.best_algo, self.best_fitness, self.population, scores

    def evolve(self, generations: int = 15) -> Tuple[AlgorithmArray, float]:
        """
        Evolution with archive-based parent selection and crossover.

        Args:
            generations: Number of generations to evolve

        Returns:
            Best algorithm and its fitness
        """
        # Initialize population
        population = []
        for _ in range(self.pop_size):
            algo = self.create_initial_algorithm()
            # Heavily mutate initial population
            for _ in range(np.random.randint(2, 6)):
                algo = self._random_mutate(algo)
            population.append(algo)

        best_fitness = self.task.get_baseline_fitness()
        best_algo = self.create_initial_algorithm()

        for gen in range(generations):
            self.generation_counter = gen

            # Evaluate population
            pop_fitness = []
            success_count = 0
            failure_count = 0
            for algo in population:
                if not algo.validate_semantics():
                    try:
                        fitness = self._evaluate_on_miner_tasks(algo)
                        pop_fitness.append((algo, fitness))
                        success_count += 1
                    except Exception:
                        pop_fitness.append((algo, -np.inf))
                        failure_count += 1
                else:
                    pop_fitness.append((algo, -np.inf))
                    failure_count += 1

            pop_fitness.sort(key=lambda x: x[1], reverse=True)

            # Add all to archive
            for algo, fit in pop_fitness:
                self.add_to_archive(algo, fit)

            # Track best
            if pop_fitness and pop_fitness[0][1] > best_fitness:
                best_fitness = pop_fitness[0][1]
                best_algo = pop_fitness[0][0]

            # Print progress
            fitnesses = [f for _, f in pop_fitness if f != -np.inf]
            avg_fitness = np.mean(fitnesses) if fitnesses else -np.inf
            unique_sigs = len(set(entry["signature"] for entry in self.archive))
            print(
                f"archive gen {gen}: best={best_fitness:.4f}, avg={avg_fitness:.4f}, "
                f"archive_size={len(self.archive)}, unique_designs={unique_sigs}, "
                f"success={success_count}, failed={failure_count}"
            )

            # Build next generation
            new_population = [pop_fitness[0][0]]  # Elitism

            while len(new_population) < self.pop_size:
                # Choose evolution strategy
                strategy = np.random.choice(
                    ["mutate_archive", "mutate_current", "crossover", "fresh"],
                    p=[0.4, 0.3, 0.2, 0.1],
                )

                if strategy == "mutate_archive" and len(self.archive) > 0:
                    # Mutate from archive
                    parent = self.sample_parent_from_archive()
                    child = self._random_mutate(parent)

                elif strategy == "mutate_current":
                    # Mutate from current population
                    parent = pop_fitness[np.random.randint(self.pop_size // 2)][0]
                    child = self._random_mutate(parent)

                elif strategy == "crossover" and len(self.archive) >= 2:
                    # Crossover between archive members
                    parent1 = self.sample_parent_from_archive()
                    parent2 = self.sample_parent_from_archive()
                    child = self.crossover(parent1, parent2)
                    # Add small mutation
                    if np.random.rand() < 0.3:
                        child = self._random_mutate(child)

                else:  # fresh
                    # Create new random algorithm
                    child = self.create_initial_algorithm()
                    for _ in range(np.random.randint(2, 5)):
                        child = self._random_mutate(child)

                new_population.append(child)

            population = new_population

        # Final report on archive
        print(f"\n📊 Archive Analysis:")
        print(f"Total algorithms explored: {len(self.archive)}")
        print(f"Unique algorithm types: {len(self.archive_diversity)}")
        print(f"Best fitness in archive: {max(e['fitness'] for e in self.archive):.4f}")

        # Find most successful algorithm type
        type_fitness = {}
        for entry in self.archive:
            sig = entry["signature"]
            if sig not in type_fitness:
                type_fitness[sig] = []
            type_fitness[sig].append(entry["fitness"])

        best_type = max(
            type_fitness.items(), key=lambda x: np.mean(x[1]) if x[1] else 0
        )
        print(f"Most successful type avg fitness: {np.mean(best_type[1]):.4f}")

        return best_algo, best_fitness

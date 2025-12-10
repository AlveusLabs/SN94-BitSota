from typing import Optional, List

import numpy as np

from .algorithm_array import AlgorithmArray, OPCODES, ADDR_VECTORS, ADDR_MATRICES


class ArrayExecutor:
    """Vectorized executor for AlgorithmArray format"""

    def __init__(self, algorithm: AlgorithmArray):
        self.algorithm = algorithm
        self.input_dim = algorithm.input_dim

        # Memory layout
        self.scalar_count = algorithm.scalar_count
        self.vector_count = algorithm.vector_count
        self.matrix_count = algorithm.matrix_count
        self.vector_dim = algorithm.vector_dim
        self._rng = np.random.default_rng()

        # Internal state buffers (initialized lazily)
        self._scalars: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._matrices: Optional[np.ndarray] = None
        self._batch_size: Optional[int] = None

    def _initialize_state(self, batch_size: int) -> None:
        """Allocate buffers for the requested batch size."""
        self._batch_size = batch_size
        self._scalars = np.zeros((batch_size, self.scalar_count), dtype=np.float32)
        self._vectors = np.zeros(
            (batch_size, self.vector_count, self.vector_dim), dtype=np.float32,
        )
        self._matrices = np.zeros(
            (batch_size, self.matrix_count, self.vector_dim, self.vector_dim),
            dtype=np.float32,
        )

    def reset_state(self, batch_size: int) -> None:
        """Reset execution buffers for a new task or genome evaluation."""
        self._initialize_state(batch_size)

    def execute_batch(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        phases: Optional[List[str]] = None,
        reset_state: bool = True,
    ) -> np.ndarray:
        """
        Execute algorithm on entire batch at once

        Args:
            X: Input data of shape (batch_size, input_dim)
            y: Optional labels of shape (batch_size,)

        Returns:
            Predictions of shape (batch_size,)
        """
        if X is None:
            raise ValueError("Input array X is required for execution")

        batch_size = X.shape[0]
        needs_reinit = (
            reset_state
            or self._scalars is None
            or self._batch_size != batch_size
        )
        if needs_reinit:
            self._initialize_state(batch_size)
        

        scalars = self._scalars
        vectors = self._vectors
        matrices = self._matrices

        if scalars is None or vectors is None or matrices is None:
            raise RuntimeError("Executor state is uninitialized")

        phase_sequence = phases or self.algorithm.get_phases()
        if not phase_sequence:
            return scalars[:, 0].copy()

        # Load inputs into v0 when predict/learn phases are executed.
        if any(phase in ("predict", "learn") for phase in phase_sequence):
            vectors[:, 0, : self.input_dim] = X

        # Execute phases
        for phase in phase_sequence:
            if phase == "learn" and y is None:
                continue

            if phase == "learn":
                # Add labels to s1 for learning
                scalars[:, 1] = y

            self._execute_phase(phase, scalars, vectors, matrices, X, y)

        # Return predictions from s0
        return scalars[:, 0].copy()

    def _get_mem(
        self, addr: int, scalars: np.ndarray, vectors: np.ndarray, matrices: np.ndarray
    ):
        """Helper to get a reference to the correct memory array and index."""
        if addr < ADDR_VECTORS:
            return scalars, addr
        elif addr < ADDR_MATRICES:
            return vectors, addr - ADDR_VECTORS
        else:
            return matrices, addr - ADDR_MATRICES

    def _execute_phase(
        self,
        phase: str,
        scalars: np.ndarray,
        vectors: np.ndarray,
        matrices: np.ndarray,
        X: np.ndarray,
        y: Optional[np.ndarray],
    ) -> None:
        """Execute a single phase on the entire batch"""

        ops, arg1, arg2, dest, const1, const2 = self.algorithm.get_phase_ops(phase)
        batch_size = scalars.shape[0]

        for i in range(len(ops)):
            op = ops[i]
            a1, a2, d = arg1[i], arg2[i], dest[i]
            c1, c2 = const1[i], const2[i]

            if op == OPCODES["NOOP"]:
                continue

            # Get memory references
            mem_d, idx_d = self._get_mem(d, scalars, vectors, matrices)
            mem_a1, idx_a1 = self._get_mem(a1, scalars, vectors, matrices)
            mem_a2, idx_a2 = self._get_mem(a2, scalars, vectors, matrices)

            # --- Universal Operations ---
            if op == OPCODES["ADD"]:
                mem_d[:, idx_d] = mem_a1[:, idx_a1] + mem_a2[:, idx_a2]
            elif op == OPCODES["SUB"]:
                mem_d[:, idx_d] = mem_a1[:, idx_a1] - mem_a2[:, idx_a2]
            elif op == OPCODES["MUL"]:
                mem_d[:, idx_d] = mem_a1[:, idx_a1] * mem_a2[:, idx_a2]
            elif op == OPCODES["DIV"]:
                mem_d[:, idx_d] = mem_a1[:, idx_a1] / (mem_a2[:, idx_a2] + 1e-8)

            # --- Unary Operations ---
            elif op == OPCODES["ABS"]:
                mem_d[:, idx_d] = np.abs(mem_a1[:, idx_a1])
            elif op == OPCODES["EXP"]:
                mem_d[:, idx_d] = np.exp(np.clip(mem_a1[:, idx_a1], -10, 10))
            elif op == OPCODES["LOG"]:
                mem_d[:, idx_d] = np.log(np.abs(mem_a1[:, idx_a1]) + 1e-8)
            elif op == OPCODES["SIN"]:
                mem_d[:, idx_d] = np.sin(mem_a1[:, idx_a1])
            elif op == OPCODES["COS"]:
                mem_d[:, idx_d] = np.cos(mem_a1[:, idx_a1])
            elif op == OPCODES["TAN"]:
                mem_d[:, idx_d] = np.tan(mem_a1[:, idx_a1])
            elif op == OPCODES["HEAVISIDE"]:
                mem_d[:, idx_d] = (mem_a1[:, idx_a1] > 0).astype(np.float32)

            # --- Constant Loading ---
            elif op == OPCODES["CONST"]:
                mem_d[:, idx_d] = c1
            elif op == OPCODES["GAUSSIAN"]:
                dest = mem_d[:, idx_d]
                self._rng.standard_normal(dest.shape, dtype=dest.dtype, out=dest)
                dest *= c2
                dest += c1
            elif op == OPCODES["UNIFORM"]:
                dest = mem_d[:, idx_d]
                self._rng.random(dest.shape, out=dest)
                dest *= (c2 - c1)
                dest += c1

            elif op == OPCODES["COPY"]:
                if d < ADDR_VECTORS:  # Scalar destination
                    if a1 < ADDR_VECTORS:  # Scalar source
                        scalars[:, d] = scalars[:, a1]
                    elif a1 < ADDR_MATRICES:  # Vector source (take norm)
                        vectors_norm = np.linalg.norm(
                            vectors[:, a1 - ADDR_VECTORS], axis=1
                        )
                        scalars[:, d] = vectors_norm
                    else:  # Matrix source (take norm)
                        matrices_norm = np.linalg.norm(
                            matrices[:, a1 - ADDR_MATRICES], axis=(1, 2)
                        )
                        scalars[:, d] = matrices_norm

            # --- Specialized Operations ---
            elif op == OPCODES["CONST_VEC"]:
                vectors[:, idx_d, int(c1)] = c2

            elif op == OPCODES["DOT"]:
                # v_a1 . v_a2 -> s_d
                mem_d[:, idx_d] = np.einsum(
                    "bi,bi->b", vectors[:, idx_a1], vectors[:, idx_a2]
                )

            elif op == OPCODES["MATMUL"]:
                # m_a1 @ v_a2 -> v_d
                mem_d[:, idx_d] = np.einsum(
                    "bij,bj->bi", matrices[:, idx_a1], vectors[:, idx_a2]
                )

            elif op == OPCODES["OUTER"]:
                # v_a1 outer v_a2 -> m_d
                mem_d[:, idx_d] = np.einsum(
                    "bi,bj->bij", vectors[:, idx_a1], vectors[:, idx_a2]
                )

            elif op == OPCODES["NORM"]:
                # norm(v_a1) -> v_d (element-wise norm) or s_d (scalar norm)
                if d < ADDR_VECTORS:  # scalar destination
                    mem_d[:, idx_d] = np.linalg.norm(vectors[:, idx_a1], axis=1)
                else:  # vector destination
                    mem_d[:, idx_d] = np.linalg.norm(
                        vectors[:, idx_a1], axis=1, keepdims=True
                    )

            elif op == OPCODES["MEAN"]:
                if d < ADDR_VECTORS:
                    mem_d[:, idx_d] = np.mean(vectors[:, idx_a1], axis=1)
                else:
                    mem_d[:, idx_d] = np.mean(vectors[:, idx_a1], axis=1, keepdims=True)

            elif op == OPCODES["STD"]:
                if d < ADDR_VECTORS:
                    mem_d[:, idx_d] = np.std(vectors[:, idx_a1], axis=1)
                else:
                    mem_d[:, idx_d] = np.std(vectors[:, idx_a1], axis=1, keepdims=True)

    def execute_single(self, x: np.ndarray) -> float:
        """Execute on single sample for compatibility"""
        X = x.reshape(1, -1)
        result = self.execute_batch(X)
        return float(result[0])

from typing import Optional, List, Callable, Any

import numpy as np

from .algorithm_array import AlgorithmArray, OPCODES, ADDR_VECTORS, ADDR_MATRICES


TraceCallback = Callable[[dict[str, Any]], None]


class ArrayExecutor:
    """Vectorized executor for AlgorithmArray format"""

    def __init__(self, algorithm: AlgorithmArray, rng_seed: Optional[int] = None):
        self.algorithm = algorithm
        self.input_dim = algorithm.input_dim

        # Memory layout
        self.scalar_count = algorithm.scalar_count
        self.vector_count = algorithm.vector_count
        self.matrix_count = algorithm.matrix_count
        self.vector_dim = algorithm.vector_dim
        self._rng = np.random.default_rng(rng_seed)

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
        trace: Optional[TraceCallback] = None,
        trace_sample_idx: int = 0,
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

        trace_sample = 0
        if trace is not None:
            trace_sample = int(trace_sample_idx)
            if trace_sample < 0 or trace_sample >= batch_size:
                trace_sample = 0

        phase_sequence = phases or self.algorithm.get_phases()
        if not phase_sequence:
            return scalars[:, 0].copy()

        # Load inputs into v0 when predict/learn phases are executed.
        if any(phase in ("predict", "learn") for phase in phase_sequence):
            if trace is None:
                vectors[:, 0, : self.input_dim] = X
            else:
                before = vectors[trace_sample, 0, : self.input_dim].copy()
                vectors[:, 0, : self.input_dim] = X
                after = vectors[trace_sample, 0, : self.input_dim].copy()
                trace(
                    {
                        "event": "input_loaded",
                        "phase": None,
                        "ip": None,
                        "before": before,
                        "after": after,
                        "sample_idx": trace_sample,
                        "scalars_row": scalars[trace_sample].copy(),
                    }
                )

        # Execute phases
        for phase in phase_sequence:
            if phase == "learn" and y is None:
                continue

            if phase == "learn":
                # Add labels to s1 for learning
                if trace is None:
                    scalars[:, 1] = y
                else:
                    before = float(scalars[trace_sample, 1])
                    scalars[:, 1] = y
                    after = float(scalars[trace_sample, 1])
                    trace(
                        {
                            "event": "labels_loaded",
                            "phase": phase,
                            "ip": None,
                            "before": before,
                            "after": after,
                            "sample_idx": trace_sample,
                            "scalars_row": scalars[trace_sample].copy(),
                        }
                    )

            self._execute_phase(
                phase,
                scalars,
                vectors,
                matrices,
                X,
                y,
                trace=trace,
                trace_sample_idx=trace_sample,
            )
            if phase == "learn":
                # Clear labels after learning to avoid leakage into later predicts.
                scalars[:, 1] = 0.0

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
        *,
        trace: Optional[TraceCallback] = None,
        trace_sample_idx: int = 0,
    ) -> None:
        """Execute a single phase on the entire batch"""

        ops, arg1, arg2, dest, const1, const2 = self.algorithm.get_phase_ops(phase)

        opcode_map = None
        if trace is not None:
            opcode_map = {v: k for k, v in OPCODES.items()}

        def emit(
            *,
            ip: int,
            op: int,
            a1: int,
            a2: int,
            d: int,
            c1: float,
            c2: float,
            executed: bool,
            reason: Optional[str] = None,
            dest_kind: Optional[str] = None,
            dest_index: Optional[int] = None,
            before: Optional[np.ndarray] = None,
            after: Optional[np.ndarray] = None,
        ) -> None:
            if trace is None:
                return
            op_name = opcode_map.get(op, f"OP_{op}") if opcode_map is not None else str(op)
            trace(
                {
                    "event": "step",
                    "phase": phase,
                    "ip": int(ip),
                    "op": op_name,
                    "arg1": int(a1),
                    "arg2": int(a2),
                    "dest": int(d),
                    "const1": float(c1),
                    "const2": float(c2),
                    "executed": bool(executed),
                    "reason": reason,
                    "dest_kind": dest_kind,
                    "dest_index": dest_index,
                    "before": before,
                    "after": after,
                    "sample_idx": int(trace_sample_idx),
                    "scalars_row": scalars[trace_sample_idx].copy(),
                }
            )

        def addr_type(addr: int) -> str:
            if addr < 0:
                return "none"
            if addr < ADDR_VECTORS:
                return "s"
            if addr < ADDR_MATRICES:
                return "v"
            return "m"

        def pad_after_batch(arr: np.ndarray, target_ndim: int) -> np.ndarray:
            while arr.ndim < target_ndim:
                arr = arr[:, None, ...]
            return arr

        def is_broadcastable(src_shape: tuple[int, ...], dst_shape: tuple[int, ...]) -> bool:
            if len(src_shape) != len(dst_shape):
                return False
            if src_shape[0] != dst_shape[0]:
                return False
            for src_dim, dst_dim in zip(src_shape[1:], dst_shape[1:]):
                if src_dim != 1 and src_dim != dst_dim:
                    return False
            return True

        def are_broadcast_compatible(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> bool:
            if len(shape_a) != len(shape_b):
                return False
            if shape_a[0] != shape_b[0]:
                return False
            for dim_a, dim_b in zip(shape_a[1:], shape_b[1:]):
                if dim_a != dim_b and dim_a != 1 and dim_b != 1:
                    return False
            return True

        def assign(dest_view: np.ndarray, value: np.ndarray) -> bool:
            if value.ndim > dest_view.ndim:
                return False
            if value.ndim < dest_view.ndim:
                value = pad_after_batch(value, dest_view.ndim)
            if not is_broadcastable(value.shape, dest_view.shape):
                return False
            dest_view[...] = value
            return True

        def valid_reg(mem: np.ndarray, idx: int) -> bool:
            return 0 <= int(idx) < int(mem.shape[1])

        binary_ops = {
            OPCODES["ADD"],
            OPCODES["SUB"],
            OPCODES["MUL"],
            OPCODES["DIV"],
            OPCODES["DOT"],
            OPCODES["MATMUL"],
            OPCODES["OUTER"],
        }

        for i in range(len(ops)):
            op = int(ops[i])
            a1, a2, d = int(arg1[i]), int(arg2[i]), int(dest[i])
            c1, c2 = float(const1[i]), float(const2[i])

            if op == OPCODES["NOOP"]:
                emit(
                    ip=i,
                    op=op,
                    a1=a1,
                    a2=a2,
                    d=d,
                    c1=c1,
                    c2=c2,
                    executed=False,
                    reason="NOOP",
                )
                continue

            # Guard against sentinel "-1" ("none") operands.
            if d < 0:
                emit(
                    ip=i,
                    op=op,
                    a1=a1,
                    a2=a2,
                    d=d,
                    c1=c1,
                    c2=c2,
                    executed=False,
                    reason="dest is none",
                )
                continue
            if op not in (OPCODES["CONST"], OPCODES["CONST_VEC"], OPCODES["GAUSSIAN"], OPCODES["UNIFORM"]):
                if a1 < 0:
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="arg1 is none",
                    )
                    continue
                if op in binary_ops and a2 < 0:
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="arg2 is none",
                    )
                    continue

            mem_d, idx_d = self._get_mem(d, scalars, vectors, matrices)
            if not valid_reg(mem_d, idx_d):
                emit(
                    ip=i,
                    op=op,
                    a1=a1,
                    a2=a2,
                    d=d,
                    c1=c1,
                    c2=c2,
                    executed=False,
                    reason="dest register out of bounds",
                    dest_kind=addr_type(d),
                    dest_index=int(idx_d),
                )
                continue
            dest_view = mem_d[:, idx_d]

            mem_a1 = idx_a1 = None
            mem_a2 = idx_a2 = None
            if a1 >= 0:
                mem_a1, idx_a1 = self._get_mem(a1, scalars, vectors, matrices)
                if mem_a1 is not None and not valid_reg(mem_a1, idx_a1):
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="arg1 register out of bounds",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                    )
                    continue
            if a2 >= 0:
                mem_a2, idx_a2 = self._get_mem(a2, scalars, vectors, matrices)
                if mem_a2 is not None and not valid_reg(mem_a2, idx_a2):
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="arg2 register out of bounds",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                    )
                    continue

            dest_before = None
            if trace is not None:
                dest_before = np.array(dest_view[trace_sample_idx], copy=True)

            # --- Universal Operations (broadcasts per-sample across vectors/matrices) ---
            if op in (OPCODES["ADD"], OPCODES["SUB"], OPCODES["MUL"], OPCODES["DIV"]):
                if mem_a1 is None or mem_a2 is None:
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="missing operands",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                    )
                    continue
                lhs = mem_a1[:, idx_a1]
                rhs = mem_a2[:, idx_a2]
                target_ndim = max(lhs.ndim, rhs.ndim)
                lhs = pad_after_batch(lhs, target_ndim)
                rhs = pad_after_batch(rhs, target_ndim)
                if not are_broadcast_compatible(lhs.shape, rhs.shape):
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="operands not broadcast compatible",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                        before=dest_before,
                        after=dest_before,
                    )
                    continue
                if op == OPCODES["ADD"]:
                    result = lhs + rhs
                elif op == OPCODES["SUB"]:
                    result = lhs - rhs
                elif op == OPCODES["MUL"]:
                    result = lhs * rhs
                else:
                    result = lhs / (rhs + 1e-8)
                assign(dest_view, result)

            # --- Unary Operations ---
            elif op in (
                OPCODES["ABS"],
                OPCODES["EXP"],
                OPCODES["LOG"],
                OPCODES["SIN"],
                OPCODES["COS"],
                OPCODES["TAN"],
                OPCODES["HEAVISIDE"],
            ):
                if mem_a1 is None:
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="missing operand",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                    )
                    continue
                src = mem_a1[:, idx_a1]
                if op == OPCODES["ABS"]:
                    result = np.abs(src)
                elif op == OPCODES["EXP"]:
                    result = np.exp(np.clip(src, -10, 10))
                elif op == OPCODES["LOG"]:
                    result = np.log(np.abs(src) + 1e-8)
                elif op == OPCODES["SIN"]:
                    result = np.sin(src)
                elif op == OPCODES["COS"]:
                    result = np.cos(src)
                elif op == OPCODES["TAN"]:
                    result = np.tan(src)
                else:  # HEAVISIDE
                    result = (src > 0).astype(np.float32)
                assign(dest_view, result)

            # --- Constant Loading ---
            elif op == OPCODES["CONST"]:
                dest_view[...] = c1
            elif op == OPCODES["GAUSSIAN"]:
                dest_mem = dest_view
                self._rng.standard_normal(dest_mem.shape, dtype=dest_mem.dtype, out=dest_mem)
                dest_mem *= c2
                dest_mem += c1
            elif op == OPCODES["UNIFORM"]:
                dest_mem = dest_view
                self._rng.random(dest_mem.shape, dtype=dest_mem.dtype, out=dest_mem)
                dest_mem *= (c2 - c1)
                dest_mem += c1

            elif op == OPCODES["COPY"]:
                if mem_a1 is None:
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="missing operand",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                    )
                    continue
                dst_kind = addr_type(d)
                src_kind = addr_type(a1)

                if dst_kind == "s":
                    if src_kind == "s":
                        scalars[:, idx_d] = scalars[:, idx_a1]
                    elif src_kind == "v":
                        scalars[:, idx_d] = np.linalg.norm(vectors[:, idx_a1], axis=1)
                    elif src_kind == "m":
                        scalars[:, idx_d] = np.linalg.norm(matrices[:, idx_a1], axis=(1, 2))
                else:
                    src_view = mem_a1[:, idx_a1]
                    assign(dest_view, src_view)

            # --- Specialized Operations ---
            elif op == OPCODES["CONST_VEC"]:
                if addr_type(d) != "v":
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="dest is not a vector",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                        before=dest_before,
                        after=dest_before,
                    )
                    continue
                elem_idx = int(c1)
                if elem_idx < 0 or elem_idx >= int(self.vector_dim):
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="vector element index out of bounds",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                        before=dest_before,
                        after=dest_before,
                    )
                    continue
                vectors[:, idx_d, elem_idx] = c2

            elif op == OPCODES["DOT"]:
                if addr_type(a1) != "v" or addr_type(a2) != "v":
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="operands are not vectors",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                        before=dest_before,
                        after=dest_before,
                    )
                    continue
                v1 = vectors[:, idx_a1]
                v2 = vectors[:, idx_a2]
                result = np.einsum("bi,bi->b", v1, v2)
                assign(dest_view, result)

            elif op == OPCODES["MATMUL"]:
                if addr_type(a1) != "m" or addr_type(a2) != "v":
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="operands are not matrix/vector",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                        before=dest_before,
                        after=dest_before,
                    )
                    continue
                mat = matrices[:, idx_a1]
                vec = vectors[:, idx_a2]
                result = np.einsum("bij,bj->bi", mat, vec)
                assign(dest_view, result)

            elif op == OPCODES["OUTER"]:
                if addr_type(a1) != "v" or addr_type(a2) != "v":
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="operands are not vectors",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                        before=dest_before,
                        after=dest_before,
                    )
                    continue
                v1 = vectors[:, idx_a1]
                v2 = vectors[:, idx_a2]
                result = np.einsum("bi,bj->bij", v1, v2)
                assign(dest_view, result)

            elif op == OPCODES["NORM"]:
                if addr_type(a1) != "v":
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="operand is not a vector",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                        before=dest_before,
                        after=dest_before,
                    )
                    continue
                result = np.linalg.norm(vectors[:, idx_a1], axis=1)
                assign(dest_view, result)

            elif op == OPCODES["MEAN"]:
                if addr_type(a1) != "v":
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="operand is not a vector",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                        before=dest_before,
                        after=dest_before,
                    )
                    continue
                result = np.mean(vectors[:, idx_a1], axis=1)
                assign(dest_view, result)

            elif op == OPCODES["STD"]:
                if addr_type(a1) != "v":
                    emit(
                        ip=i,
                        op=op,
                        a1=a1,
                        a2=a2,
                        d=d,
                        c1=c1,
                        c2=c2,
                        executed=False,
                        reason="operand is not a vector",
                        dest_kind=addr_type(d),
                        dest_index=int(idx_d),
                        before=dest_before,
                        after=dest_before,
                    )
                    continue
                result = np.std(vectors[:, idx_a1], axis=1)
                assign(dest_view, result)

            if trace is not None:
                dest_after = np.array(dest_view[trace_sample_idx], copy=True)
                emit(
                    ip=i,
                    op=op,
                    a1=a1,
                    a2=a2,
                    d=d,
                    c1=c1,
                    c2=c2,
                    executed=True,
                    dest_kind=addr_type(d),
                    dest_index=int(idx_d),
                    before=dest_before,
                    after=dest_after,
                )

    def execute_single(self, x: np.ndarray) -> float:
        """Execute on single sample for compatibility"""
        X = x.reshape(1, -1)
        result = self.execute_batch(X)
        return float(result[0])

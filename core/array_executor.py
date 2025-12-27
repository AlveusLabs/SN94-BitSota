import math
import os
from typing import Optional, List, Tuple

import numpy as np

from .algorithm_array import AlgorithmArray, OPCODES, ADDR_VECTORS, ADDR_MATRICES

try:
    import numba as nb
except Exception:  # pragma: no cover - optional dependency
    nb = None

_NUMBA_AVAILABLE = nb is not None
_EXEC_STATS = {
    "shared_numba": 0,
    "shared_python": 0,
    "batch_numba": 0,
    "batch_numpy": 0,
}


def _bump_exec_stat(key: str, count: int = 1) -> None:
    try:
        _EXEC_STATS[key] = int(_EXEC_STATS.get(key, 0)) + int(count)
    except Exception:
        return


def get_exec_stats(reset: bool = False) -> dict:
    stats = {k: int(v) for k, v in _EXEC_STATS.items()}
    if reset:
        for key in _EXEC_STATS:
            _EXEC_STATS[key] = 0
    return stats


def _numba_enabled() -> bool:
    if not _NUMBA_AVAILABLE:
        return False
    flag = os.getenv("AUTOML_ZERO_USE_NUMBA", "").strip().lower()
    if not flag:
        return True
    return flag in {"1", "true", "yes", "on"}


def _numba_batch_enabled() -> bool:
    if not _NUMBA_AVAILABLE:
        return False
    flag = os.getenv("AUTOML_ZERO_USE_NUMBA_BATCH", "").strip().lower()
    if not flag:
        return False
    return flag in {"1", "true", "yes", "on"}


def _numba_batch_parallel_min() -> int:
    value = os.getenv("AUTOML_ZERO_NUMBA_BATCH_PARALLEL_MIN", "").strip().lower()
    if not value:
        return 128
    try:
        return max(1, int(value))
    except Exception:
        return 128


_OP_NOOP = OPCODES["NOOP"]
_OP_CONST = OPCODES["CONST"]
_OP_CONST_VEC = OPCODES["CONST_VEC"]
_OP_ADD = OPCODES["ADD"]
_OP_SUB = OPCODES["SUB"]
_OP_MUL = OPCODES["MUL"]
_OP_DIV = OPCODES["DIV"]
_OP_ABS = OPCODES["ABS"]
_OP_EXP = OPCODES["EXP"]
_OP_LOG = OPCODES["LOG"]
_OP_SIN = OPCODES["SIN"]
_OP_COS = OPCODES["COS"]
_OP_TAN = OPCODES["TAN"]
_OP_HEAVISIDE = OPCODES["HEAVISIDE"]
_OP_GAUSSIAN = OPCODES["GAUSSIAN"]
_OP_UNIFORM = OPCODES["UNIFORM"]
_OP_DOT = OPCODES["DOT"]
_OP_MATMUL = OPCODES["MATMUL"]
_OP_OUTER = OPCODES["OUTER"]
_OP_NORM = OPCODES["NORM"]
_OP_MEAN = OPCODES["MEAN"]
_OP_STD = OPCODES["STD"]
_OP_COPY = OPCODES["COPY"]


if _NUMBA_AVAILABLE:
    prange = nb.prange

    @nb.njit(cache=True)
    def _execute_phase_numba(
        ops: np.ndarray,
        arg1: np.ndarray,
        arg2: np.ndarray,
        dest: np.ndarray,
        const1: np.ndarray,
        const2: np.ndarray,
        scalars: np.ndarray,
        vectors: np.ndarray,
        matrices: np.ndarray,
        rand_offsets: np.ndarray,
        rand_sizes: np.ndarray,
        rand_vals: np.ndarray,
        vector_dim: int,
    ) -> None:
        for i in range(ops.shape[0]):
            op = ops[i]
            if op == _OP_NOOP:
                continue

            a1 = int(arg1[i])
            a2 = int(arg2[i])
            d = int(dest[i])
            c1 = float(const1[i])
            c2 = float(const2[i])

            if op == _OP_ADD:
                if d < ADDR_VECTORS:
                    scalars[d] = scalars[a1] + scalars[a2]
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = v1[j] + v2[j]
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    m2 = matrices[a2 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = m1[r, c] + m2[r, c]

            elif op == _OP_SUB:
                if d < ADDR_VECTORS:
                    scalars[d] = scalars[a1] - scalars[a2]
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = v1[j] - v2[j]
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    m2 = matrices[a2 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = m1[r, c] - m2[r, c]

            elif op == _OP_MUL:
                if d < ADDR_VECTORS:
                    scalars[d] = scalars[a1] * scalars[a2]
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = v1[j] * v2[j]
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    m2 = matrices[a2 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = m1[r, c] * m2[r, c]

            elif op == _OP_DIV:
                if d < ADDR_VECTORS:
                    scalars[d] = scalars[a1] / (scalars[a2] + 1e-8)
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = v1[j] / (v2[j] + 1e-8)
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    m2 = matrices[a2 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = m1[r, c] / (m2[r, c] + 1e-8)

            elif op == _OP_ABS:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    scalars[d] = val if val >= 0 else -val

            elif op == _OP_EXP:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    if val < -10.0:
                        val = -10.0
                    elif val > 10.0:
                        val = 10.0
                    scalars[d] = math.exp(val)

            elif op == _OP_LOG:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    if val < 0:
                        val = -val
                    scalars[d] = math.log(val + 1e-8)

            elif op == _OP_SIN:
                if d < ADDR_VECTORS:
                    scalars[d] = math.sin(scalars[a1])

            elif op == _OP_COS:
                if d < ADDR_VECTORS:
                    scalars[d] = math.cos(scalars[a1])

            elif op == _OP_TAN:
                if d < ADDR_VECTORS:
                    scalars[d] = math.tan(scalars[a1])

            elif op == _OP_HEAVISIDE:
                if d < ADDR_VECTORS:
                    scalars[d] = 1.0 if scalars[a1] > 0 else 0.0

            elif op == _OP_CONST:
                if d < ADDR_VECTORS:
                    scalars[d] = c1

            elif op == _OP_GAUSSIAN or op == _OP_UNIFORM:
                offset = rand_offsets[i]
                size = rand_sizes[i]
                if offset >= 0 and size > 0:
                    if d < ADDR_VECTORS:
                        val = rand_vals[offset]
                        if op == _OP_GAUSSIAN:
                            scalars[d] = val * c2 + c1
                        else:
                            scalars[d] = val * (c2 - c1) + c1
                    elif d < ADDR_MATRICES:
                        vd = vectors[d - ADDR_VECTORS]
                        for j in range(vector_dim):
                            val = rand_vals[offset + j]
                            if op == _OP_GAUSSIAN:
                                vd[j] = val * c2 + c1
                            else:
                                vd[j] = val * (c2 - c1) + c1
                    else:
                        md = matrices[d - ADDR_MATRICES]
                        idx = offset
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                val = rand_vals[idx]
                                idx += 1
                                if op == _OP_GAUSSIAN:
                                    md[r, c] = val * c2 + c1
                                else:
                                    md[r, c] = val * (c2 - c1) + c1

            elif op == _OP_COPY:
                if d < ADDR_VECTORS:
                    if a1 < ADDR_VECTORS:
                        scalars[d] = scalars[a1]
                    elif a1 < ADDR_MATRICES:
                        v = vectors[a1 - ADDR_VECTORS]
                        acc = 0.0
                        for j in range(vector_dim):
                            acc += v[j] * v[j]
                        scalars[d] = math.sqrt(acc)
                    else:
                        m = matrices[a1 - ADDR_MATRICES]
                        acc = 0.0
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                val = m[r, c]
                                acc += val * val
                        scalars[d] = math.sqrt(acc)

            elif op == _OP_CONST_VEC:
                if d >= ADDR_VECTORS and d < ADDR_MATRICES:
                    idx = int(c1)
                    if idx >= 0 and idx < vector_dim:
                        vectors[d - ADDR_VECTORS, idx] = c2

            elif op == _OP_DOT:
                if d < ADDR_VECTORS:
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    acc = 0.0
                    for j in range(vector_dim):
                        acc += v1[j] * v2[j]
                    scalars[d] = acc

            elif op == _OP_MATMUL:
                if d >= ADDR_VECTORS and d < ADDR_MATRICES:
                    m = matrices[a1 - ADDR_MATRICES]
                    v = vectors[a2 - ADDR_VECTORS]
                    vd = vectors[d - ADDR_VECTORS]
                    for r in range(vector_dim):
                        acc = 0.0
                        for c in range(vector_dim):
                            acc += m[r, c] * v[c]
                        vd[r] = acc

            elif op == _OP_OUTER:
                if d >= ADDR_MATRICES:
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    md = matrices[d - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = v1[r] * v2[c]

            elif op == _OP_NORM:
                v = vectors[a1 - ADDR_VECTORS]
                acc = 0.0
                for j in range(vector_dim):
                    acc += v[j] * v[j]
                norm = math.sqrt(acc)
                if d < ADDR_VECTORS:
                    scalars[d] = norm
                else:
                    vd = vectors[d - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = norm

            elif op == _OP_MEAN:
                v = vectors[a1 - ADDR_VECTORS]
                acc = 0.0
                for j in range(vector_dim):
                    acc += v[j]
                mean = acc / float(vector_dim)
                if d < ADDR_VECTORS:
                    scalars[d] = mean
                else:
                    vd = vectors[d - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = mean

            elif op == _OP_STD:
                v = vectors[a1 - ADDR_VECTORS]
                acc = 0.0
                for j in range(vector_dim):
                    acc += v[j]
                mean = acc / float(vector_dim)
                var = 0.0
                for j in range(vector_dim):
                    diff = v[j] - mean
                    var += diff * diff
                std = math.sqrt(var / float(vector_dim))
                if d < ADDR_VECTORS:
                    scalars[d] = std
                else:
                    vd = vectors[d - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = std

    @nb.njit(cache=True)
    def _execute_phase_numba_rng(
        ops: np.ndarray,
        arg1: np.ndarray,
        arg2: np.ndarray,
        dest: np.ndarray,
        const1: np.ndarray,
        const2: np.ndarray,
        scalars: np.ndarray,
        vectors: np.ndarray,
        matrices: np.ndarray,
        vector_dim: int,
    ) -> None:
        for i in range(ops.shape[0]):
            op = ops[i]
            if op == _OP_NOOP:
                continue

            a1 = int(arg1[i])
            a2 = int(arg2[i])
            d = int(dest[i])
            c1 = float(const1[i])
            c2 = float(const2[i])

            if op == _OP_ADD:
                if d < ADDR_VECTORS:
                    scalars[d] = scalars[a1] + scalars[a2]
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = v1[j] + v2[j]
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    m2 = matrices[a2 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = m1[r, c] + m2[r, c]

            elif op == _OP_SUB:
                if d < ADDR_VECTORS:
                    scalars[d] = scalars[a1] - scalars[a2]
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = v1[j] - v2[j]
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    m2 = matrices[a2 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = m1[r, c] - m2[r, c]

            elif op == _OP_MUL:
                if d < ADDR_VECTORS:
                    scalars[d] = scalars[a1] * scalars[a2]
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = v1[j] * v2[j]
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    m2 = matrices[a2 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = m1[r, c] * m2[r, c]

            elif op == _OP_DIV:
                if d < ADDR_VECTORS:
                    scalars[d] = scalars[a1] / (scalars[a2] + 1e-8)
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = v1[j] / (v2[j] + 1e-8)
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    m2 = matrices[a2 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = m1[r, c] / (m2[r, c] + 1e-8)

            elif op == _OP_ABS:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    scalars[d] = val if val >= 0 else -val

            elif op == _OP_EXP:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    if val < -10.0:
                        val = -10.0
                    elif val > 10.0:
                        val = 10.0
                    scalars[d] = math.exp(val)

            elif op == _OP_LOG:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    if val < 0:
                        val = -val
                    scalars[d] = math.log(val + 1e-8)

            elif op == _OP_SIN:
                if d < ADDR_VECTORS:
                    scalars[d] = math.sin(scalars[a1])

            elif op == _OP_COS:
                if d < ADDR_VECTORS:
                    scalars[d] = math.cos(scalars[a1])

            elif op == _OP_TAN:
                if d < ADDR_VECTORS:
                    scalars[d] = math.tan(scalars[a1])

            elif op == _OP_HEAVISIDE:
                if d < ADDR_VECTORS:
                    scalars[d] = 1.0 if scalars[a1] > 0 else 0.0

            elif op == _OP_CONST:
                if d < ADDR_VECTORS:
                    scalars[d] = c1

            elif op == _OP_GAUSSIAN:
                if d < ADDR_VECTORS:
                    scalars[d] = np.random.standard_normal() * c2 + c1
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = np.random.standard_normal() * c2 + c1
                else:
                    md = matrices[d - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = np.random.standard_normal() * c2 + c1

            elif op == _OP_UNIFORM:
                if d < ADDR_VECTORS:
                    scalars[d] = np.random.random() * (c2 - c1) + c1
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = np.random.random() * (c2 - c1) + c1
                else:
                    md = matrices[d - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = np.random.random() * (c2 - c1) + c1

            elif op == _OP_COPY:
                if d < ADDR_VECTORS:
                    if a1 < ADDR_VECTORS:
                        scalars[d] = scalars[a1]
                    elif a1 < ADDR_MATRICES:
                        v = vectors[a1 - ADDR_VECTORS]
                        acc = 0.0
                        for j in range(vector_dim):
                            acc += v[j] * v[j]
                        scalars[d] = math.sqrt(acc)
                    else:
                        m = matrices[a1 - ADDR_MATRICES]
                        acc = 0.0
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                val = m[r, c]
                                acc += val * val
                        scalars[d] = math.sqrt(acc)

            elif op == _OP_CONST_VEC:
                if d >= ADDR_VECTORS and d < ADDR_MATRICES:
                    idx = int(c1)
                    if idx >= 0 and idx < vector_dim:
                        vectors[d - ADDR_VECTORS, idx] = c2

            elif op == _OP_DOT:
                if d < ADDR_VECTORS:
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    acc = 0.0
                    for j in range(vector_dim):
                        acc += v1[j] * v2[j]
                    scalars[d] = acc

            elif op == _OP_MATMUL:
                if d >= ADDR_VECTORS and d < ADDR_MATRICES:
                    m = matrices[a1 - ADDR_MATRICES]
                    v = vectors[a2 - ADDR_VECTORS]
                    vd = vectors[d - ADDR_VECTORS]
                    for r in range(vector_dim):
                        acc = 0.0
                        for c in range(vector_dim):
                            acc += m[r, c] * v[c]
                        vd[r] = acc

            elif op == _OP_OUTER:
                if d >= ADDR_MATRICES:
                    v1 = vectors[a1 - ADDR_VECTORS]
                    v2 = vectors[a2 - ADDR_VECTORS]
                    md = matrices[d - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = v1[r] * v2[c]

            elif op == _OP_NORM:
                v = vectors[a1 - ADDR_VECTORS]
                acc = 0.0
                for j in range(vector_dim):
                    acc += v[j] * v[j]
                norm = math.sqrt(acc)
                if d < ADDR_VECTORS:
                    scalars[d] = norm
                else:
                    vd = vectors[d - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = norm

            elif op == _OP_MEAN:
                v = vectors[a1 - ADDR_VECTORS]
                acc = 0.0
                for j in range(vector_dim):
                    acc += v[j]
                mean = acc / float(vector_dim)
                if d < ADDR_VECTORS:
                    scalars[d] = mean
                else:
                    vd = vectors[d - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = mean

            elif op == _OP_STD:
                v = vectors[a1 - ADDR_VECTORS]
                acc = 0.0
                for j in range(vector_dim):
                    acc += v[j]
                mean = acc / float(vector_dim)
                var = 0.0
                for j in range(vector_dim):
                    diff = v[j] - mean
                    var += diff * diff
                std = math.sqrt(var / float(vector_dim))
                if d < ADDR_VECTORS:
                    scalars[d] = std
                else:
                    vd = vectors[d - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = std

    @nb.njit(cache=True, parallel=True, fastmath=True)
    def _execute_phase_numba_batch(
        ops: np.ndarray,
        arg1: np.ndarray,
        arg2: np.ndarray,
        dest: np.ndarray,
        const1: np.ndarray,
        const2: np.ndarray,
        scalars: np.ndarray,
        vectors: np.ndarray,
        matrices: np.ndarray,
        rand_offsets: np.ndarray,
        rand_sizes: np.ndarray,
        rand_vals: np.ndarray,
        vector_dim: int,
        parallel_threshold: int,
    ) -> None:
        batch_size = scalars.shape[0]
        use_parallel = batch_size >= parallel_threshold
        for i in range(ops.shape[0]):
            op = ops[i]
            if op == _OP_NOOP:
                continue

            a1 = int(arg1[i])
            a2 = int(arg2[i])
            d = int(dest[i])
            c1 = float(const1[i])
            c2 = float(const2[i])

            if op == _OP_ADD:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = scalars[b, a1] + scalars[b, a2]
                else:
                    for b in range(batch_size):
                        scalars[b, d] = scalars[b, a1] + scalars[b, a2]

            elif op == _OP_SUB:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = scalars[b, a1] - scalars[b, a2]
                else:
                    for b in range(batch_size):
                        scalars[b, d] = scalars[b, a1] - scalars[b, a2]

            elif op == _OP_MUL:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = scalars[b, a1] * scalars[b, a2]
                else:
                    for b in range(batch_size):
                        scalars[b, d] = scalars[b, a1] * scalars[b, a2]

            elif op == _OP_DIV:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = scalars[b, a1] / (scalars[b, a2] + 1e-8)
                else:
                    for b in range(batch_size):
                        scalars[b, d] = scalars[b, a1] / (scalars[b, a2] + 1e-8)

            elif op == _OP_ABS:
                if use_parallel:
                    for b in prange(batch_size):
                        val = scalars[b, a1]
                        scalars[b, d] = val if val >= 0 else -val
                else:
                    for b in range(batch_size):
                        val = scalars[b, a1]
                        scalars[b, d] = val if val >= 0 else -val

            elif op == _OP_EXP:
                if use_parallel:
                    for b in prange(batch_size):
                        val = scalars[b, a1]
                        if val < -10.0:
                            val = -10.0
                        elif val > 10.0:
                            val = 10.0
                        scalars[b, d] = math.exp(val)
                else:
                    for b in range(batch_size):
                        val = scalars[b, a1]
                        if val < -10.0:
                            val = -10.0
                        elif val > 10.0:
                            val = 10.0
                        scalars[b, d] = math.exp(val)

            elif op == _OP_LOG:
                if use_parallel:
                    for b in prange(batch_size):
                        val = scalars[b, a1]
                        if val < 0:
                            val = -val
                        scalars[b, d] = math.log(val + 1e-8)
                else:
                    for b in range(batch_size):
                        val = scalars[b, a1]
                        if val < 0:
                            val = -val
                        scalars[b, d] = math.log(val + 1e-8)

            elif op == _OP_SIN:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = math.sin(scalars[b, a1])
                else:
                    for b in range(batch_size):
                        scalars[b, d] = math.sin(scalars[b, a1])

            elif op == _OP_COS:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = math.cos(scalars[b, a1])
                else:
                    for b in range(batch_size):
                        scalars[b, d] = math.cos(scalars[b, a1])

            elif op == _OP_TAN:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = math.tan(scalars[b, a1])
                else:
                    for b in range(batch_size):
                        scalars[b, d] = math.tan(scalars[b, a1])

            elif op == _OP_HEAVISIDE:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = 1.0 if scalars[b, a1] > 0 else 0.0
                else:
                    for b in range(batch_size):
                        scalars[b, d] = 1.0 if scalars[b, a1] > 0 else 0.0

            elif op == _OP_CONST:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = c1
                else:
                    for b in range(batch_size):
                        scalars[b, d] = c1

            elif op == _OP_GAUSSIAN or op == _OP_UNIFORM:
                offset = rand_offsets[i]
                size = rand_sizes[i]
                if offset >= 0 and size >= batch_size:
                    if use_parallel:
                        for b in prange(batch_size):
                            val = rand_vals[offset + b]
                            if op == _OP_GAUSSIAN:
                                scalars[b, d] = val * c2 + c1
                            else:
                                scalars[b, d] = val * (c2 - c1) + c1
                    else:
                        for b in range(batch_size):
                            val = rand_vals[offset + b]
                            if op == _OP_GAUSSIAN:
                                scalars[b, d] = val * c2 + c1
                            else:
                                scalars[b, d] = val * (c2 - c1) + c1

            elif op == _OP_COPY:
                if use_parallel:
                    for b in prange(batch_size):
                        scalars[b, d] = scalars[b, a1]
                else:
                    for b in range(batch_size):
                        scalars[b, d] = scalars[b, a1]

            elif op == _OP_CONST_VEC:
                if d >= ADDR_VECTORS and d < ADDR_MATRICES:
                    idx = int(c1)
                    if idx >= 0 and idx < vector_dim:
                        if use_parallel:
                            for b in prange(batch_size):
                                vectors[b, d - ADDR_VECTORS, idx] = c2
                        else:
                            for b in range(batch_size):
                                vectors[b, d - ADDR_VECTORS, idx] = c2

            elif op == _OP_DOT:
                if d < ADDR_VECTORS:
                    v1 = a1 - ADDR_VECTORS
                    v2 = a2 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j] * vectors[b, v2, j]
                            scalars[b, d] = acc
                    else:
                        for b in range(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j] * vectors[b, v2, j]
                            scalars[b, d] = acc

            elif op == _OP_MATMUL:
                if d >= ADDR_VECTORS and d < ADDR_MATRICES:
                    m1 = a1 - ADDR_MATRICES
                    v2 = a2 - ADDR_VECTORS
                    vd = d - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            for r in range(vector_dim):
                                acc = 0.0
                                for c in range(vector_dim):
                                    acc += matrices[b, m1, r, c] * vectors[b, v2, c]
                                vectors[b, vd, r] = acc
                    else:
                        for b in range(batch_size):
                            for r in range(vector_dim):
                                acc = 0.0
                                for c in range(vector_dim):
                                    acc += matrices[b, m1, r, c] * vectors[b, v2, c]
                                vectors[b, vd, r] = acc

            elif op == _OP_OUTER:
                if d >= ADDR_MATRICES:
                    v1 = a1 - ADDR_VECTORS
                    v2 = a2 - ADDR_VECTORS
                    md = d - ADDR_MATRICES
                    if use_parallel:
                        for b in prange(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = (
                                        vectors[b, v1, r] * vectors[b, v2, c]
                                    )
                    else:
                        for b in range(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = (
                                        vectors[b, v1, r] * vectors[b, v2, c]
                                    )

            elif op == _OP_NORM:
                if d < ADDR_VECTORS:
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j] * vectors[b, v1, j]
                            scalars[b, d] = math.sqrt(acc)
                    else:
                        for b in range(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j] * vectors[b, v1, j]
                            scalars[b, d] = math.sqrt(acc)

            elif op == _OP_MEAN:
                if d < ADDR_VECTORS:
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j]
                            scalars[b, d] = acc / float(vector_dim)
                    else:
                        for b in range(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j]
                            scalars[b, d] = acc / float(vector_dim)

            elif op == _OP_STD:
                if d < ADDR_VECTORS:
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j]
                            mean = acc / float(vector_dim)
                            var = 0.0
                            for j in range(vector_dim):
                                diff = vectors[b, v1, j] - mean
                                var += diff * diff
                            scalars[b, d] = math.sqrt(var / float(vector_dim))
                    else:
                        for b in range(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j]
                            mean = acc / float(vector_dim)
                            var = 0.0
                            for j in range(vector_dim):
                                diff = vectors[b, v1, j] - mean
                                var += diff * diff
                            scalars[b, d] = math.sqrt(var / float(vector_dim))

    @nb.njit(cache=True)
    def _execute_phase_numba_batch_rng(
        ops: np.ndarray,
        arg1: np.ndarray,
        arg2: np.ndarray,
        dest: np.ndarray,
        const1: np.ndarray,
        const2: np.ndarray,
        scalars: np.ndarray,
        vectors: np.ndarray,
        matrices: np.ndarray,
        vector_dim: int,
    ) -> None:
        batch_size = scalars.shape[0]
        for b in range(batch_size):
            _execute_phase_numba_rng(
                ops,
                arg1,
                arg2,
                dest,
                const1,
                const2,
                scalars[b],
                vectors[b],
                matrices[b],
                vector_dim,
            )
    @nb.njit(cache=True)
    def _run_shared_memory_numba(
        setup_ops: np.ndarray,
        setup_arg1: np.ndarray,
        setup_arg2: np.ndarray,
        setup_dest: np.ndarray,
        setup_const1: np.ndarray,
        setup_const2: np.ndarray,
        predict_ops: np.ndarray,
        predict_arg1: np.ndarray,
        predict_arg2: np.ndarray,
        predict_dest: np.ndarray,
        predict_const1: np.ndarray,
        predict_const2: np.ndarray,
        learn_ops: np.ndarray,
        learn_arg1: np.ndarray,
        learn_arg2: np.ndarray,
        learn_dest: np.ndarray,
        learn_const1: np.ndarray,
        learn_const2: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        epochs: int,
        scalar_count: int,
        vector_count: int,
        matrix_count: int,
        vector_dim: int,
        input_dim: int,
        seed: int,
    ) -> np.ndarray:
        if seed >= 0:
            np.random.seed(seed)

        scalars = np.zeros((scalar_count,), dtype=np.float32)
        vectors = np.zeros((vector_count, vector_dim), dtype=np.float32)
        matrices = np.zeros((matrix_count, vector_dim, vector_dim), dtype=np.float32)

        if setup_ops.shape[0] > 0:
            _execute_phase_numba_rng(
                setup_ops,
                setup_arg1,
                setup_arg2,
                setup_dest,
                setup_const1,
                setup_const2,
                scalars,
                vectors,
                matrices,
                vector_dim,
            )

        for _ in range(epochs):
            for i in range(X_train.shape[0]):
                for j in range(input_dim):
                    vectors[0, j] = X_train[i, j]
                if predict_ops.shape[0] > 0:
                    _execute_phase_numba_rng(
                        predict_ops,
                        predict_arg1,
                        predict_arg2,
                        predict_dest,
                        predict_const1,
                        predict_const2,
                        scalars,
                        vectors,
                        matrices,
                        vector_dim,
                    )
                scalars[1] = y_train[i]
                if learn_ops.shape[0] > 0:
                    _execute_phase_numba_rng(
                        learn_ops,
                        learn_arg1,
                        learn_arg2,
                        learn_dest,
                        learn_const1,
                        learn_const2,
                        scalars,
                        vectors,
                        matrices,
                        vector_dim,
                    )
                scalars[1] = 0.0

        preds = np.zeros((X_val.shape[0],), dtype=np.float32)
        for i in range(X_val.shape[0]):
            for j in range(input_dim):
                vectors[0, j] = X_val[i, j]
            if predict_ops.shape[0] > 0:
                _execute_phase_numba_rng(
                    predict_ops,
                    predict_arg1,
                    predict_arg2,
                    predict_dest,
                    predict_const1,
                    predict_const2,
                    scalars,
                    vectors,
                    matrices,
                    vector_dim,
                )
            preds[i] = scalars[0]
        return preds


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
        self._use_numba = _numba_enabled()
        self._use_numba_batch = _numba_batch_enabled()
        self._numba_batch_parallel_min = _numba_batch_parallel_min()
        self._rng_seed = rng_seed
        self._numba_safe = False
        if self._use_numba and _NUMBA_AVAILABLE:
            try:
                self._numba_safe = (
                    not self.algorithm.validate_addresses()
                    and not self.algorithm.validate_semantics()
                )
            except Exception:
                self._numba_safe = False

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

    def record_shared_fallback(self) -> None:
        _bump_exec_stat("shared_python")

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

        used_numba = False

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

            used_numba |= self._execute_phase(phase, scalars, vectors, matrices, X, y)
            if phase == "learn":
                # Clear labels after learning to avoid leakage into later predicts.
                scalars[:, 1] = 0.0

        if used_numba:
            _bump_exec_stat("batch_numba")
        else:
            _bump_exec_stat("batch_numpy")

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

    def _prepare_random_buffers(
        self, ops: np.ndarray, dest: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rand_offsets = np.full(len(ops), -1, dtype=np.int32)
        rand_sizes = np.zeros(len(ops), dtype=np.int32)
        if len(ops) == 0:
            return rand_offsets, rand_sizes, np.empty(0, dtype=np.float32)

        rand_values = []
        offset = 0
        for i, op in enumerate(ops):
            if op != OPCODES["GAUSSIAN"] and op != OPCODES["UNIFORM"]:
                continue
            d = int(dest[i])
            if d < ADDR_VECTORS:
                size = 1
            elif d < ADDR_MATRICES:
                size = int(self.vector_dim)
            else:
                size = int(self.vector_dim) * int(self.vector_dim)
            if size <= 0:
                continue

            if op == OPCODES["GAUSSIAN"]:
                values = self._rng.standard_normal(size).astype(np.float32, copy=False)
            else:
                values = self._rng.random(size).astype(np.float32, copy=False)

            rand_offsets[i] = offset
            rand_sizes[i] = size
            rand_values.append(values)
            offset += size

        if not rand_values:
            return rand_offsets, rand_sizes, np.empty(0, dtype=np.float32)
        return rand_offsets, rand_sizes, np.concatenate(rand_values).astype(
            np.float32, copy=False
        )

    def _prepare_random_buffers_batch(
        self, ops: np.ndarray, dest: np.ndarray, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rand_offsets = np.full(len(ops), -1, dtype=np.int32)
        rand_sizes = np.zeros(len(ops), dtype=np.int32)
        if len(ops) == 0 or batch_size <= 0:
            return rand_offsets, rand_sizes, np.empty(0, dtype=np.float32)

        rand_values = []
        offset = 0
        for i, op in enumerate(ops):
            if op != OPCODES["GAUSSIAN"] and op != OPCODES["UNIFORM"]:
                continue
            size = int(batch_size)
            if size <= 0:
                continue

            if op == OPCODES["GAUSSIAN"]:
                values = self._rng.standard_normal(size).astype(np.float32, copy=False)
            else:
                values = self._rng.random(size).astype(np.float32, copy=False)

            rand_offsets[i] = offset
            rand_sizes[i] = size
            rand_values.append(values)
            offset += size

        if not rand_values:
            return rand_offsets, rand_sizes, np.empty(0, dtype=np.float32)
        return rand_offsets, rand_sizes, np.concatenate(rand_values).astype(
            np.float32, copy=False
        )

    def execute_shared_memory_numba(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        *,
        epochs: int = 1,
        rng_seed: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if not self._use_numba or not _NUMBA_AVAILABLE or not self._numba_safe:
            return None
        if X_train is None or y_train is None or X_val is None:
            return None
        if X_train.shape[0] <= 0 or X_val.shape[0] <= 0:
            return None
        try:
            setup_ops, setup_arg1, setup_arg2, setup_dest, setup_const1, setup_const2 = (
                self.algorithm.get_phase_ops("setup")
            )
            (
                predict_ops,
                predict_arg1,
                predict_arg2,
                predict_dest,
                predict_const1,
                predict_const2,
            ) = self.algorithm.get_phase_ops("predict")
            learn_ops, learn_arg1, learn_arg2, learn_dest, learn_const1, learn_const2 = (
                self.algorithm.get_phase_ops("learn")
            )
        except Exception:
            return None

        try:
            epochs = max(1, int(epochs))
        except Exception:
            epochs = 1

        seed = -1 if rng_seed is None else int(rng_seed)
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        y_train = np.ascontiguousarray(y_train, dtype=np.float32)
        X_val = np.ascontiguousarray(X_val, dtype=np.float32)

        preds = _run_shared_memory_numba(
            setup_ops,
            setup_arg1,
            setup_arg2,
            setup_dest,
            setup_const1,
            setup_const2,
            predict_ops,
            predict_arg1,
            predict_arg2,
            predict_dest,
            predict_const1,
            predict_const2,
            learn_ops,
            learn_arg1,
            learn_arg2,
            learn_dest,
            learn_const1,
            learn_const2,
            X_train,
            y_train,
            X_val,
            epochs,
            int(self.scalar_count),
            int(self.vector_count),
            int(self.matrix_count),
            int(self.vector_dim),
            int(self.input_dim),
            seed,
        )
        if preds is None or len(preds) == 0:
            return None
        _bump_exec_stat("shared_numba")
        return preds

    def _execute_phase(
        self,
        phase: str,
        scalars: np.ndarray,
        vectors: np.ndarray,
        matrices: np.ndarray,
        X: np.ndarray,
        y: Optional[np.ndarray],
    ) -> bool:
        """Execute a single phase on the entire batch"""

        ops, arg1, arg2, dest, const1, const2 = self.algorithm.get_phase_ops(phase)
        batch_size = scalars.shape[0]
        if self._use_numba and _NUMBA_AVAILABLE and self._numba_safe:
            if batch_size == 1:
                rand_offsets, rand_sizes, rand_vals = self._prepare_random_buffers(ops, dest)
                _execute_phase_numba(
                    ops,
                    arg1,
                    arg2,
                    dest,
                    const1,
                    const2,
                    scalars[0],
                    vectors[0],
                    matrices[0],
                    rand_offsets,
                    rand_sizes,
                    rand_vals,
                    int(self.vector_dim),
                )
                return True
            if self._use_numba_batch:
                rand_offsets, rand_sizes, rand_vals = self._prepare_random_buffers_batch(
                    ops, dest, batch_size
                )
                _execute_phase_numba_batch(
                    ops,
                    arg1,
                    arg2,
                    dest,
                    const1,
                    const2,
                    scalars,
                    vectors,
                    matrices,
                    rand_offsets,
                    rand_sizes,
                    rand_vals,
                    int(self.vector_dim),
                    int(self._numba_batch_parallel_min),
                )
                return True

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

        return False

    def execute_single(self, x: np.ndarray) -> float:
        """Execute on single sample for compatibility"""
        X = x.reshape(1, -1)
        result = self.execute_batch(X)
        return float(result[0])

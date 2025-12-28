import math
import os
from typing import Optional, List, Tuple

import numpy as np

from .algorithm_array import (
    AlgorithmArray,
    OPCODES,
    ADDR_VECTORS,
    ADDR_MATRICES,
    addr_type,
    binary_result_type,
)

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

            if op == _OP_ADD or op == _OP_SUB or op == _OP_MUL or op == _OP_DIV:
                t1 = 0
                if a1 >= ADDR_MATRICES:
                    t1 = 2
                elif a1 >= ADDR_VECTORS:
                    t1 = 1
                t2 = 0
                if a2 >= ADDR_MATRICES:
                    t2 = 2
                elif a2 >= ADDR_VECTORS:
                    t2 = 1
                td = 0
                if d >= ADDR_MATRICES:
                    td = 2
                elif d >= ADDR_VECTORS:
                    td = 1

                res = -1
                if t1 == t2:
                    res = t1
                elif (t1 == 0 and t2 == 1) or (t1 == 1 and t2 == 0):
                    res = 1
                elif (t1 == 0 and t2 == 2) or (t1 == 2 and t2 == 0):
                    res = 2
                elif (t1 == 1 and t2 == 2) or (t1 == 2 and t2 == 1):
                    res = 2

                if res != td:
                    raise ValueError("Arithmetic type mismatch")

                if res == 0:
                    if op == _OP_ADD:
                        scalars[d] = scalars[a1] + scalars[a2]
                    elif op == _OP_SUB:
                        scalars[d] = scalars[a1] - scalars[a2]
                    elif op == _OP_MUL:
                        scalars[d] = scalars[a1] * scalars[a2]
                    else:
                        scalars[d] = scalars[a1] / (scalars[a2] + 1e-8)

                elif res == 1:
                    vd = vectors[d - ADDR_VECTORS]
                    if t1 == 0:
                        s = scalars[a1]
                        v2 = vectors[a2 - ADDR_VECTORS]
                        for j in range(vector_dim):
                            if op == _OP_ADD:
                                vd[j] = s + v2[j]
                            elif op == _OP_SUB:
                                vd[j] = s - v2[j]
                            elif op == _OP_MUL:
                                vd[j] = s * v2[j]
                            else:
                                vd[j] = s / (v2[j] + 1e-8)
                    elif t2 == 0:
                        v1 = vectors[a1 - ADDR_VECTORS]
                        s = scalars[a2]
                        for j in range(vector_dim):
                            if op == _OP_ADD:
                                vd[j] = v1[j] + s
                            elif op == _OP_SUB:
                                vd[j] = v1[j] - s
                            elif op == _OP_MUL:
                                vd[j] = v1[j] * s
                            else:
                                vd[j] = v1[j] / (s + 1e-8)
                    else:
                        v1 = vectors[a1 - ADDR_VECTORS]
                        v2 = vectors[a2 - ADDR_VECTORS]
                        for j in range(vector_dim):
                            if op == _OP_ADD:
                                vd[j] = v1[j] + v2[j]
                            elif op == _OP_SUB:
                                vd[j] = v1[j] - v2[j]
                            elif op == _OP_MUL:
                                vd[j] = v1[j] * v2[j]
                            else:
                                vd[j] = v1[j] / (v2[j] + 1e-8)

                else:
                    md = matrices[d - ADDR_MATRICES]
                    if t1 == 0:
                        s = scalars[a1]
                        m2 = matrices[a2 - ADDR_MATRICES]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = s + m2[r, c]
                                elif op == _OP_SUB:
                                    md[r, c] = s - m2[r, c]
                                elif op == _OP_MUL:
                                    md[r, c] = s * m2[r, c]
                                else:
                                    md[r, c] = s / (m2[r, c] + 1e-8)
                    elif t2 == 0:
                        m1 = matrices[a1 - ADDR_MATRICES]
                        s = scalars[a2]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = m1[r, c] + s
                                elif op == _OP_SUB:
                                    md[r, c] = m1[r, c] - s
                                elif op == _OP_MUL:
                                    md[r, c] = m1[r, c] * s
                                else:
                                    md[r, c] = m1[r, c] / (s + 1e-8)
                    elif t1 == 1 and t2 == 2:
                        v1 = vectors[a1 - ADDR_VECTORS]
                        m2 = matrices[a2 - ADDR_MATRICES]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = v1[c] + m2[r, c]
                                elif op == _OP_SUB:
                                    md[r, c] = v1[c] - m2[r, c]
                                elif op == _OP_MUL:
                                    md[r, c] = v1[c] * m2[r, c]
                                else:
                                    md[r, c] = v1[c] / (m2[r, c] + 1e-8)
                    elif t1 == 2 and t2 == 1:
                        m1 = matrices[a1 - ADDR_MATRICES]
                        v2 = vectors[a2 - ADDR_VECTORS]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = m1[r, c] + v2[c]
                                elif op == _OP_SUB:
                                    md[r, c] = m1[r, c] - v2[c]
                                elif op == _OP_MUL:
                                    md[r, c] = m1[r, c] * v2[c]
                                else:
                                    md[r, c] = m1[r, c] / (v2[c] + 1e-8)
                    else:
                        m1 = matrices[a1 - ADDR_MATRICES]
                        m2 = matrices[a2 - ADDR_MATRICES]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = m1[r, c] + m2[r, c]
                                elif op == _OP_SUB:
                                    md[r, c] = m1[r, c] - m2[r, c]
                                elif op == _OP_MUL:
                                    md[r, c] = m1[r, c] * m2[r, c]
                                else:
                                    md[r, c] = m1[r, c] / (m2[r, c] + 1e-8)
                continue

            elif op == _OP_ABS:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    scalars[d] = val if val >= 0 else -val
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        val = v1[j]
                        vd[j] = val if val >= 0 else -val
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            val = m1[r, c]
                            md[r, c] = val if val >= 0 else -val

            elif op == _OP_EXP:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    if val < -10.0:
                        val = -10.0
                    elif val > 10.0:
                        val = 10.0
                    scalars[d] = math.exp(val)
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        val = v1[j]
                        if val < -10.0:
                            val = -10.0
                        elif val > 10.0:
                            val = 10.0
                        vd[j] = math.exp(val)
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            val = m1[r, c]
                            if val < -10.0:
                                val = -10.0
                            elif val > 10.0:
                                val = 10.0
                            md[r, c] = math.exp(val)

            elif op == _OP_LOG:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    if val < 0:
                        val = -val
                    scalars[d] = math.log(val + 1e-8)
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        val = v1[j]
                        if val < 0:
                            val = -val
                        vd[j] = math.log(val + 1e-8)
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            val = m1[r, c]
                            if val < 0:
                                val = -val
                            md[r, c] = math.log(val + 1e-8)

            elif op == _OP_SIN:
                if d < ADDR_VECTORS:
                    scalars[d] = math.sin(scalars[a1])
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = math.sin(v1[j])
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = math.sin(m1[r, c])

            elif op == _OP_COS:
                if d < ADDR_VECTORS:
                    scalars[d] = math.cos(scalars[a1])
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = math.cos(v1[j])
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = math.cos(m1[r, c])

            elif op == _OP_TAN:
                if d < ADDR_VECTORS:
                    scalars[d] = math.tan(scalars[a1])
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = math.tan(v1[j])
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = math.tan(m1[r, c])

            elif op == _OP_HEAVISIDE:
                if d < ADDR_VECTORS:
                    scalars[d] = 1.0 if scalars[a1] > 0 else 0.0
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = 1.0 if v1[j] > 0 else 0.0
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = 1.0 if m1[r, c] > 0 else 0.0

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
                    if d == a2:
                        tmp = np.empty(vector_dim, dtype=vd.dtype)
                        for r in range(vector_dim):
                            acc = 0.0
                            for c in range(vector_dim):
                                acc += m[r, c] * v[c]
                            tmp[r] = acc
                        for r in range(vector_dim):
                            vd[r] = tmp[r]
                    else:
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

            if op == _OP_ADD or op == _OP_SUB or op == _OP_MUL or op == _OP_DIV:
                t1 = 0
                if a1 >= ADDR_MATRICES:
                    t1 = 2
                elif a1 >= ADDR_VECTORS:
                    t1 = 1
                t2 = 0
                if a2 >= ADDR_MATRICES:
                    t2 = 2
                elif a2 >= ADDR_VECTORS:
                    t2 = 1
                td = 0
                if d >= ADDR_MATRICES:
                    td = 2
                elif d >= ADDR_VECTORS:
                    td = 1

                res = -1
                if t1 == t2:
                    res = t1
                elif (t1 == 0 and t2 == 1) or (t1 == 1 and t2 == 0):
                    res = 1
                elif (t1 == 0 and t2 == 2) or (t1 == 2 and t2 == 0):
                    res = 2
                elif (t1 == 1 and t2 == 2) or (t1 == 2 and t2 == 1):
                    res = 2

                if res != td:
                    raise ValueError("Arithmetic type mismatch")

                if res == 0:
                    if op == _OP_ADD:
                        scalars[d] = scalars[a1] + scalars[a2]
                    elif op == _OP_SUB:
                        scalars[d] = scalars[a1] - scalars[a2]
                    elif op == _OP_MUL:
                        scalars[d] = scalars[a1] * scalars[a2]
                    else:
                        scalars[d] = scalars[a1] / (scalars[a2] + 1e-8)

                elif res == 1:
                    vd = vectors[d - ADDR_VECTORS]
                    if t1 == 0:
                        s = scalars[a1]
                        v2 = vectors[a2 - ADDR_VECTORS]
                        for j in range(vector_dim):
                            if op == _OP_ADD:
                                vd[j] = s + v2[j]
                            elif op == _OP_SUB:
                                vd[j] = s - v2[j]
                            elif op == _OP_MUL:
                                vd[j] = s * v2[j]
                            else:
                                vd[j] = s / (v2[j] + 1e-8)
                    elif t2 == 0:
                        v1 = vectors[a1 - ADDR_VECTORS]
                        s = scalars[a2]
                        for j in range(vector_dim):
                            if op == _OP_ADD:
                                vd[j] = v1[j] + s
                            elif op == _OP_SUB:
                                vd[j] = v1[j] - s
                            elif op == _OP_MUL:
                                vd[j] = v1[j] * s
                            else:
                                vd[j] = v1[j] / (s + 1e-8)
                    else:
                        v1 = vectors[a1 - ADDR_VECTORS]
                        v2 = vectors[a2 - ADDR_VECTORS]
                        for j in range(vector_dim):
                            if op == _OP_ADD:
                                vd[j] = v1[j] + v2[j]
                            elif op == _OP_SUB:
                                vd[j] = v1[j] - v2[j]
                            elif op == _OP_MUL:
                                vd[j] = v1[j] * v2[j]
                            else:
                                vd[j] = v1[j] / (v2[j] + 1e-8)

                else:
                    md = matrices[d - ADDR_MATRICES]
                    if t1 == 0:
                        s = scalars[a1]
                        m2 = matrices[a2 - ADDR_MATRICES]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = s + m2[r, c]
                                elif op == _OP_SUB:
                                    md[r, c] = s - m2[r, c]
                                elif op == _OP_MUL:
                                    md[r, c] = s * m2[r, c]
                                else:
                                    md[r, c] = s / (m2[r, c] + 1e-8)
                    elif t2 == 0:
                        m1 = matrices[a1 - ADDR_MATRICES]
                        s = scalars[a2]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = m1[r, c] + s
                                elif op == _OP_SUB:
                                    md[r, c] = m1[r, c] - s
                                elif op == _OP_MUL:
                                    md[r, c] = m1[r, c] * s
                                else:
                                    md[r, c] = m1[r, c] / (s + 1e-8)
                    elif t1 == 1 and t2 == 2:
                        v1 = vectors[a1 - ADDR_VECTORS]
                        m2 = matrices[a2 - ADDR_MATRICES]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = v1[c] + m2[r, c]
                                elif op == _OP_SUB:
                                    md[r, c] = v1[c] - m2[r, c]
                                elif op == _OP_MUL:
                                    md[r, c] = v1[c] * m2[r, c]
                                else:
                                    md[r, c] = v1[c] / (m2[r, c] + 1e-8)
                    elif t1 == 2 and t2 == 1:
                        m1 = matrices[a1 - ADDR_MATRICES]
                        v2 = vectors[a2 - ADDR_VECTORS]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = m1[r, c] + v2[c]
                                elif op == _OP_SUB:
                                    md[r, c] = m1[r, c] - v2[c]
                                elif op == _OP_MUL:
                                    md[r, c] = m1[r, c] * v2[c]
                                else:
                                    md[r, c] = m1[r, c] / (v2[c] + 1e-8)
                    else:
                        m1 = matrices[a1 - ADDR_MATRICES]
                        m2 = matrices[a2 - ADDR_MATRICES]
                        for r in range(vector_dim):
                            for c in range(vector_dim):
                                if op == _OP_ADD:
                                    md[r, c] = m1[r, c] + m2[r, c]
                                elif op == _OP_SUB:
                                    md[r, c] = m1[r, c] - m2[r, c]
                                elif op == _OP_MUL:
                                    md[r, c] = m1[r, c] * m2[r, c]
                                else:
                                    md[r, c] = m1[r, c] / (m2[r, c] + 1e-8)
                continue

            elif op == _OP_ABS:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    scalars[d] = val if val >= 0 else -val
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        val = v1[j]
                        vd[j] = val if val >= 0 else -val
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            val = m1[r, c]
                            md[r, c] = val if val >= 0 else -val

            elif op == _OP_EXP:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    if val < -10.0:
                        val = -10.0
                    elif val > 10.0:
                        val = 10.0
                    scalars[d] = math.exp(val)
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        val = v1[j]
                        if val < -10.0:
                            val = -10.0
                        elif val > 10.0:
                            val = 10.0
                        vd[j] = math.exp(val)
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            val = m1[r, c]
                            if val < -10.0:
                                val = -10.0
                            elif val > 10.0:
                                val = 10.0
                            md[r, c] = math.exp(val)

            elif op == _OP_LOG:
                if d < ADDR_VECTORS:
                    val = scalars[a1]
                    if val < 0:
                        val = -val
                    scalars[d] = math.log(val + 1e-8)
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        val = v1[j]
                        if val < 0:
                            val = -val
                        vd[j] = math.log(val + 1e-8)
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            val = m1[r, c]
                            if val < 0:
                                val = -val
                            md[r, c] = math.log(val + 1e-8)

            elif op == _OP_SIN:
                if d < ADDR_VECTORS:
                    scalars[d] = math.sin(scalars[a1])
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = math.sin(v1[j])
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = math.sin(m1[r, c])

            elif op == _OP_COS:
                if d < ADDR_VECTORS:
                    scalars[d] = math.cos(scalars[a1])
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = math.cos(v1[j])
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = math.cos(m1[r, c])

            elif op == _OP_TAN:
                if d < ADDR_VECTORS:
                    scalars[d] = math.tan(scalars[a1])
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = math.tan(v1[j])
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = math.tan(m1[r, c])

            elif op == _OP_HEAVISIDE:
                if d < ADDR_VECTORS:
                    scalars[d] = 1.0 if scalars[a1] > 0 else 0.0
                elif d < ADDR_MATRICES:
                    vd = vectors[d - ADDR_VECTORS]
                    v1 = vectors[a1 - ADDR_VECTORS]
                    for j in range(vector_dim):
                        vd[j] = 1.0 if v1[j] > 0 else 0.0
                else:
                    md = matrices[d - ADDR_MATRICES]
                    m1 = matrices[a1 - ADDR_MATRICES]
                    for r in range(vector_dim):
                        for c in range(vector_dim):
                            md[r, c] = 1.0 if m1[r, c] > 0 else 0.0

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
                    if d == a2:
                        tmp = np.empty(vector_dim, dtype=vd.dtype)
                        for r in range(vector_dim):
                            acc = 0.0
                            for c in range(vector_dim):
                                acc += m[r, c] * v[c]
                            tmp[r] = acc
                        for r in range(vector_dim):
                            vd[r] = tmp[r]
                    else:
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

            if op == _OP_ADD or op == _OP_SUB or op == _OP_MUL or op == _OP_DIV:
                t1 = 0
                if a1 >= ADDR_MATRICES:
                    t1 = 2
                elif a1 >= ADDR_VECTORS:
                    t1 = 1
                t2 = 0
                if a2 >= ADDR_MATRICES:
                    t2 = 2
                elif a2 >= ADDR_VECTORS:
                    t2 = 1
                td = 0
                if d >= ADDR_MATRICES:
                    td = 2
                elif d >= ADDR_VECTORS:
                    td = 1

                res = -1
                if t1 == t2:
                    res = t1
                elif (t1 == 0 and t2 == 1) or (t1 == 1 and t2 == 0):
                    res = 1
                elif (t1 == 0 and t2 == 2) or (t1 == 2 and t2 == 0):
                    res = 2
                elif (t1 == 1 and t2 == 2) or (t1 == 2 and t2 == 1):
                    res = 2

                if res != td:
                    raise ValueError("Arithmetic type mismatch")

                if res == 0:
                    if use_parallel:
                        for b in prange(batch_size):
                            if op == _OP_ADD:
                                scalars[b, d] = scalars[b, a1] + scalars[b, a2]
                            elif op == _OP_SUB:
                                scalars[b, d] = scalars[b, a1] - scalars[b, a2]
                            elif op == _OP_MUL:
                                scalars[b, d] = scalars[b, a1] * scalars[b, a2]
                            else:
                                scalars[b, d] = scalars[b, a1] / (scalars[b, a2] + 1e-8)
                    else:
                        for b in range(batch_size):
                            if op == _OP_ADD:
                                scalars[b, d] = scalars[b, a1] + scalars[b, a2]
                            elif op == _OP_SUB:
                                scalars[b, d] = scalars[b, a1] - scalars[b, a2]
                            elif op == _OP_MUL:
                                scalars[b, d] = scalars[b, a1] * scalars[b, a2]
                            else:
                                scalars[b, d] = scalars[b, a1] / (scalars[b, a2] + 1e-8)

                elif res == 1:
                    vd = d - ADDR_VECTORS
                    if t1 == 0:
                        v2 = a2 - ADDR_VECTORS
                        if use_parallel:
                            for b in prange(batch_size):
                                s = scalars[b, a1]
                                for j in range(vector_dim):
                                    if op == _OP_ADD:
                                        vectors[b, vd, j] = s + vectors[b, v2, j]
                                    elif op == _OP_SUB:
                                        vectors[b, vd, j] = s - vectors[b, v2, j]
                                    elif op == _OP_MUL:
                                        vectors[b, vd, j] = s * vectors[b, v2, j]
                                    else:
                                        vectors[b, vd, j] = s / (vectors[b, v2, j] + 1e-8)
                        else:
                            for b in range(batch_size):
                                s = scalars[b, a1]
                                for j in range(vector_dim):
                                    if op == _OP_ADD:
                                        vectors[b, vd, j] = s + vectors[b, v2, j]
                                    elif op == _OP_SUB:
                                        vectors[b, vd, j] = s - vectors[b, v2, j]
                                    elif op == _OP_MUL:
                                        vectors[b, vd, j] = s * vectors[b, v2, j]
                                    else:
                                        vectors[b, vd, j] = s / (vectors[b, v2, j] + 1e-8)
                    elif t2 == 0:
                        v1 = a1 - ADDR_VECTORS
                        if use_parallel:
                            for b in prange(batch_size):
                                s = scalars[b, a2]
                                for j in range(vector_dim):
                                    if op == _OP_ADD:
                                        vectors[b, vd, j] = vectors[b, v1, j] + s
                                    elif op == _OP_SUB:
                                        vectors[b, vd, j] = vectors[b, v1, j] - s
                                    elif op == _OP_MUL:
                                        vectors[b, vd, j] = vectors[b, v1, j] * s
                                    else:
                                        vectors[b, vd, j] = vectors[b, v1, j] / (s + 1e-8)
                        else:
                            for b in range(batch_size):
                                s = scalars[b, a2]
                                for j in range(vector_dim):
                                    if op == _OP_ADD:
                                        vectors[b, vd, j] = vectors[b, v1, j] + s
                                    elif op == _OP_SUB:
                                        vectors[b, vd, j] = vectors[b, v1, j] - s
                                    elif op == _OP_MUL:
                                        vectors[b, vd, j] = vectors[b, v1, j] * s
                                    else:
                                        vectors[b, vd, j] = vectors[b, v1, j] / (s + 1e-8)
                    else:
                        v1 = a1 - ADDR_VECTORS
                        v2 = a2 - ADDR_VECTORS
                        if use_parallel:
                            for b in prange(batch_size):
                                for j in range(vector_dim):
                                    if op == _OP_ADD:
                                        vectors[b, vd, j] = (
                                            vectors[b, v1, j] + vectors[b, v2, j]
                                        )
                                    elif op == _OP_SUB:
                                        vectors[b, vd, j] = (
                                            vectors[b, v1, j] - vectors[b, v2, j]
                                        )
                                    elif op == _OP_MUL:
                                        vectors[b, vd, j] = (
                                            vectors[b, v1, j] * vectors[b, v2, j]
                                        )
                                    else:
                                        vectors[b, vd, j] = vectors[b, v1, j] / (
                                            vectors[b, v2, j] + 1e-8
                                        )
                        else:
                            for b in range(batch_size):
                                for j in range(vector_dim):
                                    if op == _OP_ADD:
                                        vectors[b, vd, j] = (
                                            vectors[b, v1, j] + vectors[b, v2, j]
                                        )
                                    elif op == _OP_SUB:
                                        vectors[b, vd, j] = (
                                            vectors[b, v1, j] - vectors[b, v2, j]
                                        )
                                    elif op == _OP_MUL:
                                        vectors[b, vd, j] = (
                                            vectors[b, v1, j] * vectors[b, v2, j]
                                        )
                                    else:
                                        vectors[b, vd, j] = vectors[b, v1, j] / (
                                            vectors[b, v2, j] + 1e-8
                                        )

                else:
                    md = d - ADDR_MATRICES
                    if t1 == 0:
                        m2 = a2 - ADDR_MATRICES
                        if use_parallel:
                            for b in prange(batch_size):
                                s = scalars[b, a1]
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = s + matrices[b, m2, r, c]
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = s - matrices[b, m2, r, c]
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = s * matrices[b, m2, r, c]
                                        else:
                                            matrices[b, md, r, c] = s / (matrices[b, m2, r, c] + 1e-8)
                        else:
                            for b in range(batch_size):
                                s = scalars[b, a1]
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = s + matrices[b, m2, r, c]
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = s - matrices[b, m2, r, c]
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = s * matrices[b, m2, r, c]
                                        else:
                                            matrices[b, md, r, c] = s / (matrices[b, m2, r, c] + 1e-8)
                    elif t2 == 0:
                        m1 = a1 - ADDR_MATRICES
                        if use_parallel:
                            for b in prange(batch_size):
                                s = scalars[b, a2]
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] + s
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] - s
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] * s
                                        else:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] / (s + 1e-8)
                        else:
                            for b in range(batch_size):
                                s = scalars[b, a2]
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] + s
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] - s
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] * s
                                        else:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] / (s + 1e-8)
                    elif t1 == 1 and t2 == 2:
                        v1 = a1 - ADDR_VECTORS
                        m2 = a2 - ADDR_MATRICES
                        if use_parallel:
                            for b in prange(batch_size):
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = vectors[b, v1, c] + matrices[b, m2, r, c]
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = vectors[b, v1, c] - matrices[b, m2, r, c]
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = vectors[b, v1, c] * matrices[b, m2, r, c]
                                        else:
                                            matrices[b, md, r, c] = vectors[b, v1, c] / (matrices[b, m2, r, c] + 1e-8)
                        else:
                            for b in range(batch_size):
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = vectors[b, v1, c] + matrices[b, m2, r, c]
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = vectors[b, v1, c] - matrices[b, m2, r, c]
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = vectors[b, v1, c] * matrices[b, m2, r, c]
                                        else:
                                            matrices[b, md, r, c] = vectors[b, v1, c] / (matrices[b, m2, r, c] + 1e-8)
                    elif t1 == 2 and t2 == 1:
                        m1 = a1 - ADDR_MATRICES
                        v2 = a2 - ADDR_VECTORS
                        if use_parallel:
                            for b in prange(batch_size):
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] + vectors[b, v2, c]
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] - vectors[b, v2, c]
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] * vectors[b, v2, c]
                                        else:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] / (vectors[b, v2, c] + 1e-8)
                        else:
                            for b in range(batch_size):
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] + vectors[b, v2, c]
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] - vectors[b, v2, c]
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] * vectors[b, v2, c]
                                        else:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] / (vectors[b, v2, c] + 1e-8)
                    else:
                        m1 = a1 - ADDR_MATRICES
                        m2 = a2 - ADDR_MATRICES
                        if use_parallel:
                            for b in prange(batch_size):
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] + matrices[b, m2, r, c]
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] - matrices[b, m2, r, c]
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] * matrices[b, m2, r, c]
                                        else:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] / (matrices[b, m2, r, c] + 1e-8)
                        else:
                            for b in range(batch_size):
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        if op == _OP_ADD:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] + matrices[b, m2, r, c]
                                        elif op == _OP_SUB:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] - matrices[b, m2, r, c]
                                        elif op == _OP_MUL:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] * matrices[b, m2, r, c]
                                        else:
                                            matrices[b, md, r, c] = matrices[b, m1, r, c] / (matrices[b, m2, r, c] + 1e-8)
                continue

            elif op == _OP_ABS:
                if d < ADDR_VECTORS:
                    if use_parallel:
                        for b in prange(batch_size):
                            val = scalars[b, a1]
                            scalars[b, d] = val if val >= 0 else -val
                    else:
                        for b in range(batch_size):
                            val = scalars[b, a1]
                            scalars[b, d] = val if val >= 0 else -val
                elif d < ADDR_MATRICES:
                    vd = d - ADDR_VECTORS
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            for j in range(vector_dim):
                                val = vectors[b, v1, j]
                                vectors[b, vd, j] = val if val >= 0 else -val
                    else:
                        for b in range(batch_size):
                            for j in range(vector_dim):
                                val = vectors[b, v1, j]
                                vectors[b, vd, j] = val if val >= 0 else -val
                else:
                    md = d - ADDR_MATRICES
                    m1 = a1 - ADDR_MATRICES
                    if use_parallel:
                        for b in prange(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    val = matrices[b, m1, r, c]
                                    matrices[b, md, r, c] = val if val >= 0 else -val
                    else:
                        for b in range(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    val = matrices[b, m1, r, c]
                                    matrices[b, md, r, c] = val if val >= 0 else -val

            elif op == _OP_EXP:
                if d < ADDR_VECTORS:
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
                elif d < ADDR_MATRICES:
                    vd = d - ADDR_VECTORS
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            for j in range(vector_dim):
                                val = vectors[b, v1, j]
                                if val < -10.0:
                                    val = -10.0
                                elif val > 10.0:
                                    val = 10.0
                                vectors[b, vd, j] = math.exp(val)
                    else:
                        for b in range(batch_size):
                            for j in range(vector_dim):
                                val = vectors[b, v1, j]
                                if val < -10.0:
                                    val = -10.0
                                elif val > 10.0:
                                    val = 10.0
                                vectors[b, vd, j] = math.exp(val)
                else:
                    md = d - ADDR_MATRICES
                    m1 = a1 - ADDR_MATRICES
                    if use_parallel:
                        for b in prange(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    val = matrices[b, m1, r, c]
                                    if val < -10.0:
                                        val = -10.0
                                    elif val > 10.0:
                                        val = 10.0
                                    matrices[b, md, r, c] = math.exp(val)
                    else:
                        for b in range(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    val = matrices[b, m1, r, c]
                                    if val < -10.0:
                                        val = -10.0
                                    elif val > 10.0:
                                        val = 10.0
                                    matrices[b, md, r, c] = math.exp(val)

            elif op == _OP_LOG:
                if d < ADDR_VECTORS:
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
                elif d < ADDR_MATRICES:
                    vd = d - ADDR_VECTORS
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            for j in range(vector_dim):
                                val = vectors[b, v1, j]
                                if val < 0:
                                    val = -val
                                vectors[b, vd, j] = math.log(val + 1e-8)
                    else:
                        for b in range(batch_size):
                            for j in range(vector_dim):
                                val = vectors[b, v1, j]
                                if val < 0:
                                    val = -val
                                vectors[b, vd, j] = math.log(val + 1e-8)
                else:
                    md = d - ADDR_MATRICES
                    m1 = a1 - ADDR_MATRICES
                    if use_parallel:
                        for b in prange(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    val = matrices[b, m1, r, c]
                                    if val < 0:
                                        val = -val
                                    matrices[b, md, r, c] = math.log(val + 1e-8)
                    else:
                        for b in range(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    val = matrices[b, m1, r, c]
                                    if val < 0:
                                        val = -val
                                    matrices[b, md, r, c] = math.log(val + 1e-8)

            elif op == _OP_SIN:
                if d < ADDR_VECTORS:
                    if use_parallel:
                        for b in prange(batch_size):
                            scalars[b, d] = math.sin(scalars[b, a1])
                    else:
                        for b in range(batch_size):
                            scalars[b, d] = math.sin(scalars[b, a1])
                elif d < ADDR_MATRICES:
                    vd = d - ADDR_VECTORS
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            for j in range(vector_dim):
                                vectors[b, vd, j] = math.sin(vectors[b, v1, j])
                    else:
                        for b in range(batch_size):
                            for j in range(vector_dim):
                                vectors[b, vd, j] = math.sin(vectors[b, v1, j])
                else:
                    md = d - ADDR_MATRICES
                    m1 = a1 - ADDR_MATRICES
                    if use_parallel:
                        for b in prange(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = math.sin(matrices[b, m1, r, c])
                    else:
                        for b in range(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = math.sin(matrices[b, m1, r, c])

            elif op == _OP_COS:
                if d < ADDR_VECTORS:
                    if use_parallel:
                        for b in prange(batch_size):
                            scalars[b, d] = math.cos(scalars[b, a1])
                    else:
                        for b in range(batch_size):
                            scalars[b, d] = math.cos(scalars[b, a1])
                elif d < ADDR_MATRICES:
                    vd = d - ADDR_VECTORS
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            for j in range(vector_dim):
                                vectors[b, vd, j] = math.cos(vectors[b, v1, j])
                    else:
                        for b in range(batch_size):
                            for j in range(vector_dim):
                                vectors[b, vd, j] = math.cos(vectors[b, v1, j])
                else:
                    md = d - ADDR_MATRICES
                    m1 = a1 - ADDR_MATRICES
                    if use_parallel:
                        for b in prange(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = math.cos(matrices[b, m1, r, c])
                    else:
                        for b in range(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = math.cos(matrices[b, m1, r, c])

            elif op == _OP_TAN:
                if d < ADDR_VECTORS:
                    if use_parallel:
                        for b in prange(batch_size):
                            scalars[b, d] = math.tan(scalars[b, a1])
                    else:
                        for b in range(batch_size):
                            scalars[b, d] = math.tan(scalars[b, a1])
                elif d < ADDR_MATRICES:
                    vd = d - ADDR_VECTORS
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            for j in range(vector_dim):
                                vectors[b, vd, j] = math.tan(vectors[b, v1, j])
                    else:
                        for b in range(batch_size):
                            for j in range(vector_dim):
                                vectors[b, vd, j] = math.tan(vectors[b, v1, j])
                else:
                    md = d - ADDR_MATRICES
                    m1 = a1 - ADDR_MATRICES
                    if use_parallel:
                        for b in prange(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = math.tan(matrices[b, m1, r, c])
                    else:
                        for b in range(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = math.tan(matrices[b, m1, r, c])

            elif op == _OP_HEAVISIDE:
                if d < ADDR_VECTORS:
                    if use_parallel:
                        for b in prange(batch_size):
                            scalars[b, d] = 1.0 if scalars[b, a1] > 0 else 0.0
                    else:
                        for b in range(batch_size):
                            scalars[b, d] = 1.0 if scalars[b, a1] > 0 else 0.0
                elif d < ADDR_MATRICES:
                    vd = d - ADDR_VECTORS
                    v1 = a1 - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            for j in range(vector_dim):
                                vectors[b, vd, j] = 1.0 if vectors[b, v1, j] > 0 else 0.0
                    else:
                        for b in range(batch_size):
                            for j in range(vector_dim):
                                vectors[b, vd, j] = 1.0 if vectors[b, v1, j] > 0 else 0.0
                else:
                    md = d - ADDR_MATRICES
                    m1 = a1 - ADDR_MATRICES
                    if use_parallel:
                        for b in prange(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = 1.0 if matrices[b, m1, r, c] > 0 else 0.0
                    else:
                        for b in range(batch_size):
                            for r in range(vector_dim):
                                for c in range(vector_dim):
                                    matrices[b, md, r, c] = 1.0 if matrices[b, m1, r, c] > 0 else 0.0

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
                if d < ADDR_VECTORS:
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
                elif d < ADDR_MATRICES:
                    expected = batch_size * vector_dim
                    if offset >= 0 and size >= expected:
                        vd = d - ADDR_VECTORS
                        if use_parallel:
                            for b in prange(batch_size):
                                base = offset + b * vector_dim
                                for j in range(vector_dim):
                                    val = rand_vals[base + j]
                                    if op == _OP_GAUSSIAN:
                                        vectors[b, vd, j] = val * c2 + c1
                                    else:
                                        vectors[b, vd, j] = val * (c2 - c1) + c1
                        else:
                            for b in range(batch_size):
                                base = offset + b * vector_dim
                                for j in range(vector_dim):
                                    val = rand_vals[base + j]
                                    if op == _OP_GAUSSIAN:
                                        vectors[b, vd, j] = val * c2 + c1
                                    else:
                                        vectors[b, vd, j] = val * (c2 - c1) + c1
                else:
                    expected = batch_size * vector_dim * vector_dim
                    if offset >= 0 and size >= expected:
                        md = d - ADDR_MATRICES
                        if use_parallel:
                            for b in prange(batch_size):
                                base = offset + b * vector_dim * vector_dim
                                idx = 0
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        val = rand_vals[base + idx]
                                        idx += 1
                                        if op == _OP_GAUSSIAN:
                                            matrices[b, md, r, c] = val * c2 + c1
                                        else:
                                            matrices[b, md, r, c] = val * (c2 - c1) + c1
                        else:
                            for b in range(batch_size):
                                base = offset + b * vector_dim * vector_dim
                                idx = 0
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        val = rand_vals[base + idx]
                                        idx += 1
                                        if op == _OP_GAUSSIAN:
                                            matrices[b, md, r, c] = val * c2 + c1
                                        else:
                                            matrices[b, md, r, c] = val * (c2 - c1) + c1

            elif op == _OP_COPY:
                if d < ADDR_VECTORS:
                    if a1 < ADDR_VECTORS:
                        if use_parallel:
                            for b in prange(batch_size):
                                scalars[b, d] = scalars[b, a1]
                        else:
                            for b in range(batch_size):
                                scalars[b, d] = scalars[b, a1]
                    elif a1 < ADDR_MATRICES:
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
                    else:
                        m1 = a1 - ADDR_MATRICES
                        if use_parallel:
                            for b in prange(batch_size):
                                acc = 0.0
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        val = matrices[b, m1, r, c]
                                        acc += val * val
                                scalars[b, d] = math.sqrt(acc)
                        else:
                            for b in range(batch_size):
                                acc = 0.0
                                for r in range(vector_dim):
                                    for c in range(vector_dim):
                                        val = matrices[b, m1, r, c]
                                        acc += val * val
                                scalars[b, d] = math.sqrt(acc)

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
                    if vd == v2:
                        if use_parallel:
                            for b in prange(batch_size):
                                tmp = np.empty(vector_dim, dtype=vectors.dtype)
                                for r in range(vector_dim):
                                    acc = 0.0
                                    for c in range(vector_dim):
                                        acc += matrices[b, m1, r, c] * vectors[b, v2, c]
                                    tmp[r] = acc
                                for r in range(vector_dim):
                                    vectors[b, vd, r] = tmp[r]
                        else:
                            for b in range(batch_size):
                                tmp = np.empty(vector_dim, dtype=vectors.dtype)
                                for r in range(vector_dim):
                                    acc = 0.0
                                    for c in range(vector_dim):
                                        acc += matrices[b, m1, r, c] * vectors[b, v2, c]
                                    tmp[r] = acc
                                for r in range(vector_dim):
                                    vectors[b, vd, r] = tmp[r]
                    else:
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
                v1 = a1 - ADDR_VECTORS
                if d < ADDR_VECTORS:
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
                else:
                    vd = d - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j] * vectors[b, v1, j]
                            norm = math.sqrt(acc)
                            for j in range(vector_dim):
                                vectors[b, vd, j] = norm
                    else:
                        for b in range(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j] * vectors[b, v1, j]
                            norm = math.sqrt(acc)
                            for j in range(vector_dim):
                                vectors[b, vd, j] = norm

            elif op == _OP_MEAN:
                v1 = a1 - ADDR_VECTORS
                if d < ADDR_VECTORS:
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
                else:
                    vd = d - ADDR_VECTORS
                    if use_parallel:
                        for b in prange(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j]
                            mean = acc / float(vector_dim)
                            for j in range(vector_dim):
                                vectors[b, vd, j] = mean
                    else:
                        for b in range(batch_size):
                            acc = 0.0
                            for j in range(vector_dim):
                                acc += vectors[b, v1, j]
                            mean = acc / float(vector_dim)
                            for j in range(vector_dim):
                                vectors[b, vd, j] = mean

            elif op == _OP_STD:
                v1 = a1 - ADDR_VECTORS
                if d < ADDR_VECTORS:
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
                else:
                    vd = d - ADDR_VECTORS
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
                            std = math.sqrt(var / float(vector_dim))
                            for j in range(vector_dim):
                                vectors[b, vd, j] = std
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
                            std = math.sqrt(var / float(vector_dim))
                            for j in range(vector_dim):
                                vectors[b, vd, j] = std

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
                values = self._rng.standard_normal(size, dtype=np.float32)
            else:
                values = self._rng.random(size, dtype=np.float32)

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
            d = int(dest[i])
            if d < ADDR_VECTORS:
                size = int(batch_size)
            elif d < ADDR_MATRICES:
                size = int(batch_size) * int(self.vector_dim)
            else:
                size = int(batch_size) * int(self.vector_dim) * int(self.vector_dim)
            if size <= 0:
                continue

            if op == OPCODES["GAUSSIAN"]:
                values = self._rng.standard_normal(size, dtype=np.float32)
            else:
                values = self._rng.random(size, dtype=np.float32)

            rand_offsets[i] = offset
            rand_sizes[i] = size
            rand_values.append(values)
            offset += size

        if not rand_values:
            return rand_offsets, rand_sizes, np.empty(0, dtype=np.float32)
        return rand_offsets, rand_sizes, np.concatenate(rand_values).astype(
            np.float32, copy=False
        )

    def _opcode_name(self, op: int) -> str:
        if op == _OP_ADD:
            return "ADD"
        if op == _OP_SUB:
            return "SUB"
        if op == _OP_MUL:
            return "MUL"
        if op == _OP_DIV:
            return "DIV"
        return f"OP_{int(op)}"

    def _ensure_vector_matrix_dims(
        self, vectors: np.ndarray, matrices: np.ndarray, op_name: str
    ) -> None:
        vec_dim = int(vectors.shape[-1])
        mat_rows = int(matrices.shape[-2])
        mat_cols = int(matrices.shape[-1])
        if mat_rows != mat_cols or vec_dim != mat_cols:
            raise ValueError(
                f"{op_name} vector/matrix shape mismatch: "
                f"vector_dim={vec_dim}, matrix_dim={mat_rows}x{mat_cols}"
            )

    def _apply_binary_arithmetic(
        self,
        op: int,
        a1: int,
        a2: int,
        d: int,
        scalars: np.ndarray,
        vectors: np.ndarray,
        matrices: np.ndarray,
    ) -> None:
        type1 = addr_type(a1)
        type2 = addr_type(a2)
        dest_type = addr_type(d)
        result_type = binary_result_type(type1, type2)
        if result_type is None:
            raise ValueError(
                f"{self._opcode_name(op)} unsupported types: {type1}, {type2}"
            )
        if dest_type != result_type:
            raise ValueError(
                f"{self._opcode_name(op)} dest type {dest_type} does not match {result_type}"
            )

        def apply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
            if op == _OP_ADD:
                return lhs + rhs
            if op == _OP_SUB:
                return lhs - rhs
            if op == _OP_MUL:
                return lhs * rhs
            return lhs / (rhs + 1e-8)

        if result_type == "s":
            scalars[:, d] = apply(scalars[:, a1], scalars[:, a2])
            return

        if result_type == "v":
            out = vectors[:, d - ADDR_VECTORS]
            if type1 == "s":
                left = scalars[:, a1][:, None]
                right = vectors[:, a2 - ADDR_VECTORS]
            elif type2 == "s":
                left = vectors[:, a1 - ADDR_VECTORS]
                right = scalars[:, a2][:, None]
            else:
                left = vectors[:, a1 - ADDR_VECTORS]
                right = vectors[:, a2 - ADDR_VECTORS]
            out[...] = apply(left, right)
            return

        out = matrices[:, d - ADDR_MATRICES]
        if type1 == "s":
            left = scalars[:, a1][:, None, None]
            right = matrices[:, a2 - ADDR_MATRICES]
        elif type2 == "s":
            left = matrices[:, a1 - ADDR_MATRICES]
            right = scalars[:, a2][:, None, None]
        elif type1 == "v" and type2 == "m":
            # Broadcast the vector across matrix rows.
            self._ensure_vector_matrix_dims(vectors, matrices, self._opcode_name(op))
            left = vectors[:, a1 - ADDR_VECTORS][:, None, :]
            right = matrices[:, a2 - ADDR_MATRICES]
        elif type1 == "m" and type2 == "v":
            # Broadcast the vector across matrix rows.
            self._ensure_vector_matrix_dims(vectors, matrices, self._opcode_name(op))
            left = matrices[:, a1 - ADDR_MATRICES]
            right = vectors[:, a2 - ADDR_VECTORS][:, None, :]
        else:
            left = matrices[:, a1 - ADDR_MATRICES]
            right = matrices[:, a2 - ADDR_MATRICES]
        out[...] = apply(left, right)

    def _phase_supports_numba_single(
        self,
        ops: np.ndarray,
        arg1: np.ndarray,
        arg2: np.ndarray,
        dest: np.ndarray,
    ) -> bool:
        for i in range(len(ops)):
            op = int(ops[i])
            if op == _OP_NOOP:
                continue
            t1 = addr_type(int(arg1[i]))
            t2 = addr_type(int(arg2[i]))
            td = addr_type(int(dest[i]))

            if op in (_OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV):
                result_type = binary_result_type(t1, t2)
                if result_type is None or td != result_type:
                    return False
            elif op in (
                _OP_ABS,
                _OP_EXP,
                _OP_LOG,
                _OP_SIN,
                _OP_COS,
                _OP_TAN,
                _OP_HEAVISIDE,
            ):
                if t1 != td or t1 not in ("s", "v", "m"):
                    return False
            elif op == _OP_CONST:
                if td != "s":
                    return False
            elif op in (_OP_GAUSSIAN, _OP_UNIFORM):
                if td not in ("s", "v", "m"):
                    return False
            elif op == _OP_COPY:
                if td != "s":
                    return False
            elif op == _OP_CONST_VEC:
                if td != "v":
                    return False
            elif op == _OP_DOT:
                if t1 != "v" or t2 != "v" or td != "s":
                    return False
            elif op == _OP_MATMUL:
                if t1 != "m" or t2 != "v" or td != "v":
                    return False
            elif op == _OP_OUTER:
                if t1 != "v" or t2 != "v" or td != "m":
                    return False
            elif op in (_OP_NORM, _OP_MEAN, _OP_STD):
                if t1 != "v" or td not in ("s", "v"):
                    return False
            else:
                return False

        return True

    def _phase_supports_numba_batch(
        self,
        ops: np.ndarray,
        arg1: np.ndarray,
        arg2: np.ndarray,
        dest: np.ndarray,
    ) -> bool:
        for i in range(len(ops)):
            op = int(ops[i])
            if op == _OP_NOOP:
                continue
            t1 = addr_type(int(arg1[i]))
            t2 = addr_type(int(arg2[i]))
            td = addr_type(int(dest[i]))

            if op in (_OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV):
                result_type = binary_result_type(t1, t2)
                if result_type is None or td != result_type:
                    return False
            elif op in (
                _OP_ABS,
                _OP_EXP,
                _OP_LOG,
                _OP_SIN,
                _OP_COS,
                _OP_TAN,
                _OP_HEAVISIDE,
            ):
                if t1 != td or t1 not in ("s", "v", "m"):
                    return False
            elif op == _OP_CONST:
                if td != "s":
                    return False
            elif op in (_OP_GAUSSIAN, _OP_UNIFORM):
                if td not in ("s", "v", "m"):
                    return False
            elif op == _OP_COPY:
                if td != "s":
                    return False
            elif op == _OP_CONST_VEC:
                if td != "v":
                    return False
            elif op == _OP_DOT:
                if t1 != "v" or t2 != "v" or td != "s":
                    return False
            elif op == _OP_MATMUL:
                if t1 != "m" or t2 != "v" or td != "v":
                    return False
            elif op == _OP_OUTER:
                if t1 != "v" or t2 != "v" or td != "m":
                    return False
            elif op in (_OP_NORM, _OP_MEAN, _OP_STD):
                if t1 != "v" or td not in ("s", "v"):
                    return False
            else:
                return False

        return True

    def _phase_has_vector_matrix_arithmetic(
        self,
        ops: np.ndarray,
        arg1: np.ndarray,
        arg2: np.ndarray,
    ) -> bool:
        for i in range(len(ops)):
            op = int(ops[i])
            if op in (_OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV):
                t1 = addr_type(int(arg1[i]))
                t2 = addr_type(int(arg2[i]))
                if (t1 == "v" and t2 == "m") or (t1 == "m" and t2 == "v"):
                    return True
        return False

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

        if not (
            self._phase_supports_numba_single(
                setup_ops, setup_arg1, setup_arg2, setup_dest
            )
            and self._phase_supports_numba_single(
                predict_ops, predict_arg1, predict_arg2, predict_dest
            )
            and self._phase_supports_numba_single(
                learn_ops, learn_arg1, learn_arg2, learn_dest
            )
        ):
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
            if self._phase_has_vector_matrix_arithmetic(ops, arg1, arg2):
                self._ensure_vector_matrix_dims(vectors, matrices, "ARITHMETIC")
            if batch_size == 1 and self._phase_supports_numba_single(
                ops, arg1, arg2, dest
            ):
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
            if self._use_numba_batch and self._phase_supports_numba_batch(
                ops, arg1, arg2, dest
            ):
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

            if op == _OP_NOOP:
                continue

            if op in (_OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV):
                self._apply_binary_arithmetic(op, int(a1), int(a2), int(d), scalars, vectors, matrices)
                continue

            # Get memory references
            mem_d, idx_d = self._get_mem(d, scalars, vectors, matrices)
            mem_a1, idx_a1 = self._get_mem(a1, scalars, vectors, matrices)
            mem_a2, idx_a2 = self._get_mem(a2, scalars, vectors, matrices)

            # --- Unary Operations ---
            if op == OPCODES["ABS"]:
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
                dest_buf = mem_d[:, idx_d]
                dest_buf[...] = self._rng.standard_normal(
                    dest_buf.shape, dtype=dest_buf.dtype
                )
                dest_buf *= c2
                dest_buf += c1
            elif op == OPCODES["UNIFORM"]:
                dest_buf = mem_d[:, idx_d]
                dest_buf[...] = self._rng.random(dest_buf.shape, dtype=dest_buf.dtype)
                dest_buf *= (c2 - c1)
                dest_buf += c1

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

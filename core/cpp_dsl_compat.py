from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from core.algorithm_array import (
    ADDR_MATRICES,
    ADDR_SCALARS,
    ADDR_VECTORS,
    OPCODE_METADATA,
    AlgorithmArray,
    OPCODES,
)
from core.dsl_parser import DSLParser


CPP_BACKEND_ENV_VALUES = {"cpp", "cpp_baseline", "automl_zero_cpp"}


def cpp_backend_enabled_from_env() -> bool:
    backend = str(os.environ.get("BITSOTA_MINER_BACKEND", "") or "").strip().lower()
    return backend in CPP_BACKEND_ENV_VALUES


@dataclass(frozen=True)
class CppDslProfile:
    scalar_count: int
    vector_count: int
    matrix_count: int
    vector_dim: int
    phase_max_sizes: Mapping[str, int]
    allowed_ops_by_phase: Mapping[str, Sequence[str]]


_CIFAR_BASELINE_PROFILE = CppDslProfile(
    scalar_count=5,
    vector_count=9,
    matrix_count=2,
    vector_dim=16,
    phase_max_sizes={"setup": 7, "predict": 11, "learn": 23},
    allowed_ops_by_phase={
        # Mirrors run_baseline.sh setup ops under pool op names.
        "setup": ("CONST", "GAUSSIAN", "UNIFORM"),
        # Includes arithmetic ops needed for algebraic max-rewrites.
        "predict": ("ADD", "SUB", "MUL", "ABS", "MATMUL", "DOT", "NORM", "MEAN", "STD", "HEAVISIDE"),
        "learn": ("ADD", "SUB", "MUL", "HEAVISIDE", "OUTER"),
    },
)

_SCALAR_DEMO_PROFILE = CppDslProfile(
    scalar_count=4,
    vector_count=3,
    matrix_count=1,
    vector_dim=4,
    phase_max_sizes={"setup": 10, "predict": 2, "learn": 8},
    allowed_ops_by_phase={
        "setup": ("CONST", "ADD", "SUB", "MUL", "DOT"),
        "predict": ("CONST", "ADD", "SUB", "MUL", "DOT"),
        "learn": ("CONST", "ADD", "SUB", "MUL", "DOT"),
    },
)


def profile_for_task(task_type: str, *, input_dim: int) -> CppDslProfile:
    task = str(task_type or "").strip().lower()
    if task == "scalar_linear":
        base = _SCALAR_DEMO_PROFILE
    else:
        base = _CIFAR_BASELINE_PROFILE
    return CppDslProfile(
        scalar_count=int(base.scalar_count),
        vector_count=int(base.vector_count),
        matrix_count=int(base.matrix_count),
        vector_dim=max(int(base.vector_dim), max(1, int(input_dim))),
        phase_max_sizes=dict(base.phase_max_sizes),
        allowed_ops_by_phase=dict(base.allowed_ops_by_phase),
    )


def _normalize_addr(addr: int, *, profile: CppDslProfile, expected_kind: str | None = None) -> int:
    if int(addr) < 0:
        return -1
    value = int(addr)
    kind = str(expected_kind or "").strip().lower()
    if kind in {"s", "v", "m"}:
        if kind == "s":
            base = ADDR_SCALARS
            count = max(1, int(profile.scalar_count))
        elif kind == "v":
            base = ADDR_VECTORS
            count = max(1, int(profile.vector_count))
        else:
            base = ADDR_MATRICES
            count = max(1, int(profile.matrix_count))

        idx = value - base
        if idx < 0:
            if value < ADDR_VECTORS:
                idx = value - ADDR_SCALARS
            elif value < ADDR_MATRICES:
                idx = value - ADDR_VECTORS
            else:
                idx = value - ADDR_MATRICES
        return base + (idx % count)

    if value < ADDR_VECTORS:
        idx = value - ADDR_SCALARS
        count = max(1, int(profile.scalar_count))
        return ADDR_SCALARS + (idx % count)
    if value < ADDR_MATRICES:
        idx = value - ADDR_VECTORS
        count = max(1, int(profile.vector_count))
        return ADDR_VECTORS + (idx % count)
    idx = value - ADDR_MATRICES
    count = max(1, int(profile.matrix_count))
    return ADDR_MATRICES + (idx % count)


def _is_scalar_addr(addr: int) -> bool:
    return 0 <= int(addr) < ADDR_VECTORS


def _build_cpp_style_empty(input_dim: int, profile: CppDslProfile) -> AlgorithmArray:
    phases = ["setup", "predict", "learn"]
    return AlgorithmArray.create_empty(
        input_dim=max(1, int(input_dim)),
        phases=phases,
        max_sizes={k: int(v) for k, v in profile.phase_max_sizes.items()},
        scalar_count=int(profile.scalar_count),
        vector_count=int(profile.vector_count),
        matrix_count=int(profile.matrix_count),
        vector_dim=int(profile.vector_dim),
    )


def normalize_algorithm_dsl_for_cpp(
    algorithm_dsl: str,
    *,
    input_dim: int,
    task_type: str = "cifar10_binary",
) -> str:
    profile = profile_for_task(task_type, input_dim=int(input_dim))
    try:
        parsed = DSLParser.from_dsl(str(algorithm_dsl or ""), max(1, int(input_dim)))
    except Exception:
        return DSLParser.to_dsl(_build_cpp_style_empty(max(1, int(input_dim)), profile))

    opcode_to_name = {int(code): str(name) for name, code in OPCODES.items()}
    zero_scalar_addr = ADDR_SCALARS + max(0, int(profile.scalar_count) - 1)
    phase_instr: Dict[str, List[tuple[str, int, int, int, float, float]]] = {
        "setup": [],
        "predict": [],
        "learn": [],
    }
    needs_zero_scalar = False

    for raw_phase in parsed.get_phases():
        phase = str(raw_phase).lower().strip()
        if phase not in phase_instr:
            phase = "predict"
        allowed = {str(name).upper() for name in profile.allowed_ops_by_phase.get(phase, ())}
        ops, arg1s, arg2s, dests, const1s, const2s = parsed.get_phase_ops(raw_phase)
        for i in range(len(ops)):
            op_name = opcode_to_name.get(int(ops[i]), "")
            if not op_name or op_name == "NOOP":
                continue

            metadata = OPCODE_METADATA.get(op_name, {})
            a1 = _normalize_addr(int(arg1s[i]), profile=profile, expected_kind=metadata.get("arg1"))
            a2 = _normalize_addr(int(arg2s[i]), profile=profile, expected_kind=metadata.get("arg2"))
            d = _normalize_addr(int(dests[i]), profile=profile, expected_kind=metadata.get("dest"))
            c1 = float(const1s[i])
            c2 = float(const2s[i])

            if op_name == "COPY":
                # C++ readable style does not have an explicit COPY op. Lower scalar copy
                # into an ADD with a dedicated zero scalar register.
                if not (_is_scalar_addr(a1) and _is_scalar_addr(d)):
                    continue
                needs_zero_scalar = True
                op_name = "ADD"
                a2 = int(zero_scalar_addr)

            if op_name == "CONST_VEC":
                # Drop element-wise vector constants; C++ baseline profile does not use them.
                continue
            if op_name not in allowed:
                continue

            phase_instr[phase].append((op_name, a1, a2, d, c1, c2))

    if needs_zero_scalar:
        setup_allowed = {str(name).upper() for name in profile.allowed_ops_by_phase.get("setup", ())}
        if "CONST" in setup_allowed:
            already_has_zero = any(
                op == "CONST" and int(dest) == int(zero_scalar_addr) and abs(float(c1)) <= 1e-12
                for (op, _a1, _a2, dest, c1, _c2) in phase_instr["setup"]
            )
            if not already_has_zero:
                phase_instr["setup"].insert(0, ("CONST", -1, -1, int(zero_scalar_addr), 0.0, 0.0))

    out = _build_cpp_style_empty(max(1, int(input_dim)), profile)
    for phase in ("setup", "predict", "learn"):
        cap = max(0, int(profile.phase_max_sizes.get(phase, 0)))
        for op_name, a1, a2, d, c1, c2 in phase_instr.get(phase, [])[:cap]:
            out.add_instruction(
                phase=phase,
                op=op_name,
                arg1=int(a1),
                arg2=int(a2),
                dest=int(d),
                const1=float(c1),
                const2=float(c2),
            )
    return DSLParser.to_dsl(out)


def normalize_algorithm_record_for_cpp(
    record: Mapping[str, Any],
    *,
    task_type: str = "cifar10_binary",
    fallback_input_dim: int = 16,
) -> Dict[str, Any]:
    out = dict(record or {})
    raw_dsl = str(out.get("algorithm_dsl") or "")
    if not raw_dsl.strip():
        return out
    try:
        dim = int(out.get("input_dim") or fallback_input_dim)
    except Exception:
        dim = int(fallback_input_dim)
    dim = max(1, int(dim))
    out["input_dim"] = dim
    out["algorithm_dsl"] = normalize_algorithm_dsl_for_cpp(
        raw_dsl,
        input_dim=dim,
        task_type=task_type,
    )
    return out


def normalize_algorithm_batch_for_cpp(
    records: Iterable[Mapping[str, Any]],
    *,
    task_type: str = "cifar10_binary",
    fallback_input_dim: int = 16,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        normalized.append(
            normalize_algorithm_record_for_cpp(
                record,
                task_type=task_type,
                fallback_input_dim=int(fallback_input_dim),
            )
        )
    return normalized

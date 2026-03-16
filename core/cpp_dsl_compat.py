from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from core.algorithm_array import (
    ADDR_MATRICES,
    ADDR_SCALARS,
    ADDR_VECTORS,
    AlgorithmArray,
    OPCODES,
)
from core.dsl_parser import DSLParser


CPP_BACKEND_ENV_VALUES = {"cpp", "cpp_baseline", "automl_zero_cpp"}


def cpp_backend_enabled_from_env() -> bool:
    backend = str(os.environ.get("BITSOTA_MINER_BACKEND", "") or "").strip().lower()
    return backend in CPP_BACKEND_ENV_VALUES


def cpp_allow_any_phase_ops_from_env() -> bool:
    raw = str(os.environ.get("BITSOTA_CPP_ALLOW_ANY_PHASE_OPS", "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CppDslProfile:
    scalar_count: int
    vector_count: int
    matrix_count: int
    vector_dim: int
    phase_max_sizes: Mapping[str, int]
    allowed_ops_by_phase: Mapping[str, Sequence[str]]


class CppDslComplianceError(ValueError):
    """Raised when a DSL program is not compliant with the C++ backend profile."""


_CIFAR_BASELINE_PROFILE = CppDslProfile(
    scalar_count=5,
    vector_count=9,
    matrix_count=2,
    vector_dim=16,
    phase_max_sizes={"setup": 7, "predict": 11, "learn": 23},
    allowed_ops_by_phase={
        # Mirrors run_baseline.sh setup ops under pool op names.
        "setup": ("CONST", "GAUSSIAN", "UNIFORM"),
        # run_baseline.sh predict ops:
        # SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MAX_OP, VECTOR_INNER_PRODUCT_OP, VECTOR_SUM_OP
        "predict": ("ADD", "MATMUL", "MAX", "DOT"),
        # run_baseline.sh learn ops:
        # SCALAR_SUM_OP, SCALAR_DIFF_OP, SCALAR_PRODUCT_OP, SCALAR_VECTOR_PRODUCT_OP,
        # VECTOR_SUM_OP, VECTOR_HEAVISIDE_OP, VECTOR_PRODUCT_OP, VECTOR_OUTER_PRODUCT_OP, MATRIX_SUM_OP
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
    allow_any_phase_ops = bool(cpp_allow_any_phase_ops_from_env())
    raw = str(algorithm_dsl or "").strip()
    if not raw:
        raise CppDslComplianceError("algorithm_dsl is empty")
    try:
        parsed = DSLParser.from_dsl(raw, max(1, int(input_dim)), strict=True)
    except Exception as e:
        raise CppDslComplianceError(f"DSL parse failed: {e}") from e

    allowed_by_phase = {
        phase: {str(name).upper() for name in profile.allowed_ops_by_phase.get(phase, ())}
        for phase in ("setup", "predict", "learn")
    }
    opcode_to_name = {int(code): str(name).upper() for name, code in OPCODES.items()}

    def _addr_kind(addr: int) -> str:
        value = int(addr)
        if value < 0:
            return "none"
        if value < ADDR_VECTORS:
            return "s"
        if value < ADDR_MATRICES:
            return "v"
        return "m"

    def _addr_in_profile(kind: str, addr: int) -> bool:
        value = int(addr)
        if kind == "s":
            return ADDR_SCALARS <= value < (ADDR_SCALARS + int(profile.scalar_count))
        if kind == "v":
            return ADDR_VECTORS <= value < (ADDR_VECTORS + int(profile.vector_count))
        if kind == "m":
            return ADDR_MATRICES <= value < (ADDR_MATRICES + int(profile.matrix_count))
        return value < 0

    out = _build_cpp_style_empty(max(1, int(input_dim)), profile)
    for raw_phase in parsed.get_phases():
        phase = str(raw_phase).lower().strip()
        if phase not in {"setup", "predict", "learn"}:
            raise CppDslComplianceError(f"unsupported phase '{raw_phase}'")
        cap = max(0, int(profile.phase_max_sizes.get(phase, 0)))
        allowed = allowed_by_phase.get(phase, set())

        ops, arg1s, arg2s, dests, const1s, const2s = parsed.get_phase_ops(raw_phase)
        if len(ops) > cap:
            raise CppDslComplianceError(
                f"phase '{phase}' has {len(ops)} ops but cap is {cap}"
            )

        for i in range(len(ops)):
            op_code = int(ops[i])
            op_name = opcode_to_name.get(op_code, "")
            if not op_name or op_name == "NOOP":
                continue
            if (not allow_any_phase_ops) and op_name not in allowed:
                raise CppDslComplianceError(
                    f"op '{op_name}' is not allowed in phase '{phase}' for C++ profile"
                )
            if op_name in {"COPY", "CONST_VEC"}:
                raise CppDslComplianceError(f"op '{op_name}' is not supported by C++ backend")

            arg1 = int(arg1s[i])
            arg2 = int(arg2s[i])
            dest = int(dests[i])
            def _require_addr(label: str, addr: int, expected_kind: str | None = None) -> str:
                if int(addr) < 0:
                    raise CppDslComplianceError(f"{phase}[{i}] {op_name}: {label} missing")
                actual_kind = _addr_kind(addr)
                if expected_kind is not None and actual_kind != expected_kind:
                    raise CppDslComplianceError(
                        f"{phase}[{i}] {op_name}: {label} kind '{actual_kind}' != '{expected_kind}'"
                    )
                if not _addr_in_profile(actual_kind, addr):
                    raise CppDslComplianceError(
                        f"{phase}[{i}] {op_name}: {label} address out of C++ profile bounds"
                    )
                return actual_kind

            if op_name in {"CONST", "GAUSSIAN", "UNIFORM"}:
                if arg1 != -1 or arg2 != -1:
                    raise CppDslComplianceError(
                        f"{phase}[{i}] {op_name}: constants/noise ops must not set arg1/arg2"
                    )
                _require_addr("dest", dest)
            elif op_name in {"ABS", "HEAVISIDE", "SIN", "COS", "TAN", "EXP", "LOG"}:
                k1 = _require_addr("arg1", arg1)
                if arg2 != -1:
                    raise CppDslComplianceError(f"{phase}[{i}] {op_name}: arg2 must be none")
                kd = _require_addr("dest", dest)
                if kd != k1:
                    raise CppDslComplianceError(
                        f"{phase}[{i}] {op_name}: dest kind '{kd}' != arg1 kind '{k1}'"
                    )
            elif op_name in {"ADD", "SUB", "DIV", "MIN", "MAX"}:
                k1 = _require_addr("arg1", arg1)
                k2 = _require_addr("arg2", arg2)
                kd = _require_addr("dest", dest)
                if not (k1 == k2 == kd):
                    raise CppDslComplianceError(
                        f"{phase}[{i}] {op_name}: requires same-kind arg1/arg2/dest"
                    )
            elif op_name == "MUL":
                k1 = _require_addr("arg1", arg1)
                k2 = _require_addr("arg2", arg2)
                kd = _require_addr("dest", dest)
                ok = (
                    (k1 == k2 == kd)
                    or (k1 == "s" and k2 == "v" and kd == "v")
                    or (k1 == "v" and k2 == "s" and kd == "v")
                    or (k1 == "s" and k2 == "m" and kd == "m")
                    or (k1 == "m" and k2 == "s" and kd == "m")
                )
                if not ok:
                    raise CppDslComplianceError(
                        f"{phase}[{i}] {op_name}: unsupported kind signature ({k1},{k2})->{kd}"
                    )
            elif op_name == "DOT":
                _require_addr("arg1", arg1, "v")
                _require_addr("arg2", arg2, "v")
                _require_addr("dest", dest, "s")
            elif op_name == "MATMUL":
                _require_addr("arg1", arg1, "m")
                k2 = _require_addr("arg2", arg2)
                kd = _require_addr("dest", dest)
                if not ((k2 == "v" and kd == "v") or (k2 == "m" and kd == "m")):
                    raise CppDslComplianceError(
                        f"{phase}[{i}] MATMUL: requires (m,v)->v or (m,m)->m"
                    )
            elif op_name == "OUTER":
                _require_addr("arg1", arg1, "v")
                _require_addr("arg2", arg2, "v")
                _require_addr("dest", dest, "m")
            elif op_name in {"NORM", "MEAN", "STD"}:
                k1 = _require_addr("arg1", arg1)
                if arg2 != -1:
                    raise CppDslComplianceError(f"{phase}[{i}] {op_name}: arg2 must be none")
                kd = _require_addr("dest", dest)
                if op_name == "NORM":
                    ok = (k1 in {"v", "m"}) and kd == "s"
                else:
                    ok = (k1 == "v" and kd == "s") or (k1 == "m" and kd in {"s", "v"})
                if not ok:
                    raise CppDslComplianceError(
                        f"{phase}[{i}] {op_name}: unsupported kind signature ({k1})->{kd}"
                    )
            else:
                raise CppDslComplianceError(f"{phase}[{i}] unsupported opcode '{op_name}'")

            try:
                out.add_instruction(
                    phase=phase,
                    op=op_name,
                    arg1=arg1,
                    arg2=arg2,
                    dest=dest,
                    const1=float(const1s[i]),
                    const2=float(const2s[i]),
                )
            except Exception as e:
                raise CppDslComplianceError(
                    f"{phase}[{i}] {op_name}: failed to add instruction: {e}"
                ) from e

    errors = out.validate_addresses()
    if errors:
        raise CppDslComplianceError(str(errors[0]))
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

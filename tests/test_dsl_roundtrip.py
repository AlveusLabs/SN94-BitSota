import numpy as np

from core.algorithm_array import (
    ADDR_MATRICES,
    ADDR_SCALARS,
    ADDR_VECTORS,
    OPCODE_METADATA,
    AlgorithmArray,
)
from core.dsl_parser import DSLParser


def _make_algorithm_with_small_constants() -> AlgorithmArray:
    algo = AlgorithmArray.create_empty(
        input_dim=4,
        phases=["setup", "predict", "learn"],
        max_sizes={"setup": 8, "predict": 8, "learn": 8},
    )

    algo.add_instruction("setup", "CONST", -1, -1, 5, 1e-8, 0.0)  # s5 = 1e-08
    algo.add_instruction("predict", "CONST", -1, -1, 6, -2.5e-4, 0.0)  # s6 = -2.5e-04
    algo.add_instruction("predict", "ADD", 5, 6, 0, 0.0, 0.0)  # s0 = s5 + s6

    return algo


def _sample_addr(kind: str, algo: AlgorithmArray, rng: np.random.Generator) -> int:
    s_start = ADDR_SCALARS
    s_end = ADDR_SCALARS + algo.scalar_count
    v_start = ADDR_VECTORS
    v_end = ADDR_VECTORS + algo.vector_count
    m_start = ADDR_MATRICES
    m_end = ADDR_MATRICES + algo.matrix_count

    if kind == "s":
        return int(rng.integers(s_start, s_end))
    if kind == "v":
        return int(rng.integers(v_start, v_end))
    if kind == "m":
        return int(rng.integers(m_start, m_end))
    return s_start


def _populate_random_algo(
    algo: AlgorithmArray, phase_sizes: dict, rng: np.random.Generator
) -> None:
    op_names = [name for name in OPCODE_METADATA.keys() if name != "NOOP"]
    const_choices = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    for phase, count in phase_sizes.items():
        for _ in range(count):
            op_name = str(rng.choice(op_names))
            op_meta = OPCODE_METADATA[op_name]

            arg1 = (
                -1
                if op_meta["arg1"] == "none"
                else _sample_addr(op_meta["arg1"], algo, rng)
            )
            arg2 = (
                -1
                if op_meta["arg2"] == "none"
                else _sample_addr(op_meta["arg2"], algo, rng)
            )
            dest = (
                -1
                if op_meta["dest"] == "none"
                else _sample_addr(op_meta["dest"], algo, rng)
            )

            if op_name == "CONST":
                const1 = float(rng.choice(const_choices))
                const2 = 0.0
            elif op_name == "CONST_VEC":
                const1 = int(rng.integers(0, max(1, algo.vector_dim)))
                const2 = float(rng.choice(const_choices))
            elif op_name in {"GAUSSIAN", "UNIFORM"}:
                const1 = float(rng.choice(const_choices))
                const2 = float(rng.choice(const_choices))
            else:
                const1 = 0.0
                const2 = 0.0

            algo.add_instruction(phase, op_name, arg1, arg2, dest, const1, const2)


def test_array_to_dsl_to_array_roundtrip_preserves_scientific_constants():
    algo = _make_algorithm_with_small_constants()
    dsl = DSLParser.to_dsl(algo)
    roundtripped = DSLParser.from_dsl(dsl, input_dim=algo.input_dim)

    assert algo.fingerprint() == roundtripped.fingerprint()
    assert roundtripped.validate_addresses() == []


def test_from_dsl_parses_scientific_notation_constants():
    dsl = """
# setup
s5 = 1e-08

# predict
s6 = -2.5e-4
s0 = s5 + s6
"""
    algo = DSLParser.from_dsl(dsl, input_dim=4)
    ops_setup, *_rest = algo.get_phase_ops("setup")
    ops_predict, *_rest2 = algo.get_phase_ops("predict")

    assert len(ops_setup) == 1
    assert len(ops_predict) == 2


def test_array_to_dsl_to_array_roundtrip_preserves_matrix_count_and_vector_dim():
    algo = AlgorithmArray.create_empty(
        input_dim=3,
        phases=["setup", "predict", "learn"],
        max_sizes={"setup": 16, "predict": 16, "learn": 16},
        matrix_count=8,
        vector_dim=23,
    )

    # Build a matrix in a high-index matrix register (m7).
    v1 = ADDR_VECTORS + 1
    m7 = ADDR_MATRICES + 7
    algo.add_instruction("setup", "CONST_VEC", -1, -1, v1, 0.0, 1.0)
    algo.add_instruction("setup", "CONST_VEC", -1, -1, v1, 1.0, -2.0)
    algo.add_instruction("setup", "OUTER", ADDR_VECTORS + 0, v1, m7, 0.0, 0.0)

    dsl = DSLParser.to_dsl(algo)
    roundtripped = DSLParser.from_dsl(dsl, input_dim=algo.input_dim)

    assert roundtripped.matrix_count == 8
    assert roundtripped.vector_dim == 23
    assert algo.fingerprint() == roundtripped.fingerprint()
    assert roundtripped.validate_addresses() == []


def test_from_dsl_handles_large_skewed_phase_sizes():
    setup_lines = ["s0 = s0 + s0" for _ in range(70)]
    predict_lines = ["s1 = s1 + s1", "s2 = s2 + s2"]
    learn_lines = ["s3 = s3 + s3", "s4 = s4 + s4"]

    dsl = "\n".join(
        ["# setup", *setup_lines, "", "# predict", *predict_lines, "", "# learn", *learn_lines]
    )
    algo = DSLParser.from_dsl(dsl, input_dim=4)

    assert algo.get_phase_size("setup") == len(setup_lines)
    assert algo.get_phase_size("predict") == len(predict_lines)
    assert algo.get_phase_size("learn") == len(learn_lines)
    assert algo.get_phase_max_size("setup") == 70


def test_roundtrip_preserves_skewed_phase_limits_from_meta():
    algo = AlgorithmArray.create_empty(
        input_dim=5,
        phases=["setup", "predict", "learn"],
        max_sizes={"setup": 96, "predict": 8, "learn": 4},
    )

    for _ in range(90):
        algo.add_instruction("setup", "ADD", 0, 0, 0, 0.0, 0.0)
    for _ in range(2):
        algo.add_instruction("predict", "ADD", 1, 1, 1, 0.0, 0.0)
    algo.add_instruction("learn", "ADD", 2, 2, 2, 0.0, 0.0)

    dsl = DSLParser.to_dsl(algo)
    roundtripped = DSLParser.from_dsl(dsl, input_dim=algo.input_dim)

    assert roundtripped.get_phase_size("setup") == 90
    assert roundtripped.get_phase_size("predict") == 2
    assert roundtripped.get_phase_size("learn") == 1
    assert roundtripped.get_phase_max_size("setup") == 96
    assert roundtripped.get_phase_max_size("predict") == 8
    assert roundtripped.get_phase_max_size("learn") == 4


def test_random_roundtrip_large():
    rng = np.random.default_rng(1337)
    algo = AlgorithmArray.create_empty(
        input_dim=6,
        phases=["setup", "predict", "learn"],
        max_sizes={"setup": 160, "predict": 160, "learn": 160},
    )
    _populate_random_algo(
        algo,
        {"setup": 120, "predict": 130, "learn": 110},
        rng,
    )

    dsl = DSLParser.to_dsl(algo)
    roundtripped = DSLParser.from_dsl(dsl, input_dim=algo.input_dim)

    assert algo.fingerprint() == roundtripped.fingerprint()
    assert roundtripped.get_phase_size("setup") == 120
    assert roundtripped.get_phase_size("predict") == 130
    assert roundtripped.get_phase_size("learn") == 110
    assert roundtripped.get_phase_max_size("setup") == 160
    assert roundtripped.get_phase_max_size("predict") == 160
    assert roundtripped.get_phase_max_size("learn") == 160


def test_random_roundtrip_skewed():
    rng = np.random.default_rng(2025)
    algo = AlgorithmArray.create_empty(
        input_dim=5,
        phases=["setup", "predict", "learn"],
        max_sizes={"setup": 200, "predict": 12, "learn": 6},
    )
    _populate_random_algo(
        algo,
        {"setup": 180, "predict": 6, "learn": 3},
        rng,
    )

    dsl = DSLParser.to_dsl(algo)
    roundtripped = DSLParser.from_dsl(dsl, input_dim=algo.input_dim)

    assert algo.fingerprint() == roundtripped.fingerprint()
    assert roundtripped.get_phase_size("setup") == 180
    assert roundtripped.get_phase_size("predict") == 6
    assert roundtripped.get_phase_size("learn") == 3
    assert roundtripped.get_phase_max_size("setup") == 200
    assert roundtripped.get_phase_max_size("predict") == 12
    assert roundtripped.get_phase_max_size("learn") == 6

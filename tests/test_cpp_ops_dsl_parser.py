from __future__ import annotations

import numpy as np
import pytest

from core.array_executor import ArrayExecutor
from core.dsl_parser import DSLParser


def test_cpp_style_ops_parse_and_execute():
    dsl = """
# meta: scalar_count=5 vector_count=5 matrix_count=2 vector_dim=4
def Setup():
  NoOp()
def Predict():
  v2 = dot(m0, v0)
  v3 = maximum(v2, v1)
  v4 = minimum(v3, v1)
  v1 = heaviside(v4, 1.0)
  s0 = dot(v1, v0)
def Learn():
  NoOp()
"""
    algo = DSLParser.from_dsl(dsl, input_dim=4)
    exe = ArrayExecutor(algo, rng_seed=123)
    x = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.4],
            [0.4, 0.3, -0.2, -0.1],
            [-0.7, 0.2, 0.5, -0.4],
        ],
        dtype=np.float32,
    )
    preds = exe.execute_batch(x, phases=["setup", "predict"], reset_state=True)
    assert preds.shape == (3,)
    assert np.all(np.isfinite(preds))


def test_cpp_style_ops_roundtrip_format():
    dsl = """
def Setup():
  NoOp()
def Predict():
  v1 = dot(m0, v0)
  v2 = maximum(v1, v0)
  v3 = minimum(v2, v0)
  v4 = heaviside(v3, 1.0)
def Learn():
  NoOp()
"""
    algo = DSLParser.from_dsl(dsl, input_dim=4)
    out = DSLParser.to_dsl(algo)
    assert "dot(m0, v0)" in out
    assert "maximum(v1, v0)" in out
    assert "minimum(v2, v0)" in out
    assert "heaviside(v3, 1.0)" in out


def test_dsl_parser_strict_mode_rejects_unparsed_lines():
    dsl = """
def Setup():
  NoOp()
def Predict():
  totally_invalid_syntax_here
def Learn():
  NoOp()
"""
    with pytest.raises(ValueError):
        DSLParser.from_dsl(dsl, input_dim=4, strict=True)

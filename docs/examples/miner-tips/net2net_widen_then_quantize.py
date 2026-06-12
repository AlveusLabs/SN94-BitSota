"""Net2Net-style widening before low-bit quantization.

This toy MLP shows the core trick: duplicate hidden units and split outgoing
weights so the wider model initially computes the same function.
"""

from __future__ import annotations

import numpy as np


def mlp(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return np.tanh(x @ w1.T) @ w2.T


def widen_hidden_by_duplication(
    w1: np.ndarray,
    w2: np.ndarray,
    copies: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate hidden units while preserving the MLP function."""
    w1_wide = np.repeat(w1, copies, axis=0)
    w2_wide = np.repeat(w2 / copies, copies, axis=1)
    return w1_wide, w2_wide


def ternary_np(weights: np.ndarray) -> np.ndarray:
    scale = np.mean(np.abs(weights)) + 1e-8
    threshold = 0.7 * scale
    return np.where(
        np.abs(weights) >= threshold,
        np.sign(weights) * scale,
        0.0,
    ).astype(np.float32)


def toy_check() -> None:
    rng = np.random.default_rng(11)
    w1 = rng.normal(size=(3, 4)).astype(np.float32)
    w2 = rng.normal(size=(2, 3)).astype(np.float32)
    x = rng.normal(size=(5, 4)).astype(np.float32)

    y = mlp(x, w1, w2)
    w1_wide, w2_wide = widen_hidden_by_duplication(w1, w2, copies=3)
    y_wide = mlp(x, w1_wide, w2_wide)
    y_low_bit = mlp(x, ternary_np(w1_wide), ternary_np(w2_wide))

    print("max diff after widening:", float(np.max(np.abs(y - y_wide))))
    print("mse after widening + toy ternary:", float(np.mean((y - y_low_bit) ** 2)))


if __name__ == "__main__":
    toy_check()

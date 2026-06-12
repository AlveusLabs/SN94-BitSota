"""Row sensitivity plus q4 rescue baseline.

Use this pattern for binary or ternary recipes:
1. quantize all rows to the target low-bit format;
2. measure which rows lose the most;
3. keep a small fraction of fragile rows in q4.
"""

from __future__ import annotations

import numpy as np


def quantize_binary_per_row(weights: np.ndarray) -> np.ndarray:
    scale = np.mean(np.abs(weights), axis=1, keepdims=True) + 1e-8
    return scale * np.sign(weights)


def quantize_ternary_per_row(weights: np.ndarray) -> np.ndarray:
    scale = np.mean(np.abs(weights), axis=1, keepdims=True) + 1e-8
    threshold = 0.7 * scale
    return np.where(np.abs(weights) >= threshold, np.sign(weights) * scale, 0.0)


def quantize_q4_per_row(weights: np.ndarray) -> np.ndarray:
    qmax = 7.0
    scale = np.max(np.abs(weights), axis=1, keepdims=True) / qmax + 1e-8
    q = np.clip(np.round(weights / scale), -qmax, qmax)
    return q * scale


def row_mse(original: np.ndarray, approx: np.ndarray) -> np.ndarray:
    return np.mean((original - approx) ** 2, axis=1)


def build_rowmix(
    weights: np.ndarray,
    base_mode: str = "ternary",
    rescue_fraction: float = 0.10,
) -> tuple[np.ndarray, list[int]]:
    low_bit = (
        quantize_binary_per_row(weights)
        if base_mode == "binary"
        else quantize_ternary_per_row(weights)
    )
    q4 = quantize_q4_per_row(weights)
    benefit = row_mse(weights, low_bit) - row_mse(weights, q4)
    num_rescue = max(1, int(round(weights.shape[0] * rescue_fraction)))
    rescue_rows = np.argsort(-benefit)[:num_rescue].tolist()

    mixed = low_bit.copy()
    mixed[rescue_rows] = q4[rescue_rows]
    return mixed, rescue_rows


def toy_check() -> None:
    rng = np.random.default_rng(7)
    weights = rng.normal(size=(12, 16)).astype(np.float32)
    weights[[2, 7, 10]] *= 4.0

    mixed, rescue_rows = build_rowmix(weights, base_mode="ternary")
    pure = quantize_ternary_per_row(weights)
    print("q4 rescue rows:", rescue_rows)
    print("pure ternary mse:", float(np.mean(row_mse(weights, pure))))
    print("mixed mse:", float(np.mean(row_mse(weights, mixed))))


if __name__ == "__main__":
    toy_check()

"""Toy layerwise distillation loop for binary/ternary compression.

The production-scale idea:
1. quantize one student layer;
2. match the teacher layer output on public calibration text;
3. freeze it;
4. move to the next layer;
5. only then run public/dev PPL.
"""

from __future__ import annotations

import numpy as np


def make_block(rng: np.random.Generator, width: int = 16) -> dict[str, np.ndarray]:
    scale = 1.0 / np.sqrt(width)
    return {
        "w1": rng.normal(scale=scale, size=(width, width)).astype(np.float64),
        "b1": np.zeros(width, dtype=np.float64),
        "w2": rng.normal(scale=scale, size=(width, width)).astype(np.float64),
        "b2": np.zeros(width, dtype=np.float64),
    }


def clone_block(block: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in block.items()}


def block_forward(block: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, dict]:
    z = x @ block["w1"].T + block["b1"]
    h = np.tanh(z)
    y = x + h @ block["w2"].T + block["b2"]
    return y, {"x": x, "z": z, "h": h}


def stack_forward(
    blocks: list[dict[str, np.ndarray]],
    x: np.ndarray,
    stop_after: int | None = None,
) -> np.ndarray:
    for i, block in enumerate(blocks):
        x, _ = block_forward(block, x)
        if stop_after is not None and i == stop_after:
            break
    return x


def ternary_matrix(weights: np.ndarray) -> np.ndarray:
    scale = np.mean(np.abs(weights)) + 1e-8
    threshold = 0.7 * scale
    return np.where(np.abs(weights) >= threshold, np.sign(weights) * scale, 0.0)


def quantize_one_block_ternary(block: dict[str, np.ndarray]) -> None:
    block["w1"] = ternary_matrix(block["w1"])
    block["w2"] = ternary_matrix(block["w2"])


def distill_one_block(
    teacher: list[dict[str, np.ndarray]],
    student: list[dict[str, np.ndarray]],
    block_index: int,
    rng: np.random.Generator,
    steps: int = 200,
    lr: float = 0.03,
) -> float:
    """Train one student block with direct MSE gradients."""
    block = student[block_index]
    for _ in range(steps):
        x0 = rng.normal(size=(64, 16)).astype(np.float64)
        prefix = x0 if block_index == 0 else stack_forward(student, x0, block_index - 1)
        target = stack_forward(teacher, x0, block_index)
        pred, cache = block_forward(block, prefix)

        err = pred - target
        loss = float(np.mean(err * err))
        grad_y = (2.0 / err.size) * err
        grad_w2 = grad_y.T @ cache["h"]
        grad_b2 = np.sum(grad_y, axis=0)
        grad_h = grad_y @ block["w2"]
        grad_z = grad_h * (1.0 - cache["h"] ** 2)
        grad_w1 = grad_z.T @ cache["x"]
        grad_b1 = np.sum(grad_z, axis=0)

        block["w2"] -= lr * grad_w2
        block["b2"] -= lr * grad_b2
        block["w1"] -= lr * grad_w1
        block["b1"] -= lr * grad_b1

    return loss


def toy_check() -> None:
    rng = np.random.default_rng(0)
    teacher = [make_block(rng) for _ in range(3)]
    student = [clone_block(block) for block in teacher]

    for layer in range(len(student)):
        quantize_one_block_ternary(student[layer])
        loss = distill_one_block(teacher, student, layer, rng)
        print(f"layer {layer} final fit mse: {loss:.6f}")

    x = rng.normal(size=(128, 16)).astype(np.float64)
    final_mse = np.mean((stack_forward(student, x) - stack_forward(teacher, x)) ** 2)
    print(f"full-stack teacher/student mse: {final_mse:.6f}")


if __name__ == "__main__":
    toy_check()

"""Public/dev PPL and artifact packaging helpers.

This file is intentionally small and public-safe. It uses toy tensors by
default. Replace the toy logits with your own public/dev text and local
candidate model when you are ready.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def shifted_cross_entropy_from_logits(logits, labels, ignore_index: int = -100):
    """Compute next-token cross entropy from logits with NumPy.

    logits: [batch, sequence, vocab]
    labels: [batch, sequence]
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)

    keep = flat_labels != ignore_index
    flat_logits = flat_logits[keep]
    flat_labels = flat_labels[keep]
    if flat_labels.size == 0:
        raise ValueError("No scored tokens after applying ignore_index.")

    # Stable log-softmax: logsumexp(logits) - logit_of_true_label.
    max_logits = np.max(flat_logits, axis=1, keepdims=True)
    logsumexp = max_logits[:, 0] + np.log(
        np.sum(np.exp(flat_logits - max_logits), axis=1)
    )
    nll = logsumexp - flat_logits[np.arange(flat_labels.size), flat_labels]
    return float(np.mean(nll))


def perplexity_from_loss(loss) -> float:
    value = float(loss)
    return float("inf") if value > 80 else math.exp(value)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_directory_deterministic(source_dir: str | Path, zip_path: str | Path) -> dict:
    """Zip a model directory while skipping caches and generated junk."""
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    skip_names = {".git", "__pycache__", ".ipynb_checkpoints"}
    skip_suffixes = {".pyc", ".log"}

    files = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_names for part in path.parts):
            continue
        if path.suffix in skip_suffixes:
            continue
        files.append(path)

    fixed_time = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            relpath = path.relative_to(source_dir).as_posix()
            info = zipfile.ZipInfo(relpath, date_time=fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())

    return {
        "zip_path": str(zip_path),
        "artifact_size_bytes": zip_path.stat().st_size,
        "artifact_sha256": sha256_file(zip_path),
    }


def toy_check() -> None:
    labels = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    logits = np.zeros((2, 4, 5), dtype=np.float64)
    for batch in range(labels.shape[0]):
        for pos in range(labels.shape[1] - 1):
            logits[batch, pos, labels[batch, pos + 1]] = 5.0

    loss = shifted_cross_entropy_from_logits(logits, labels)
    print("toy score:", {"loss": loss, "ppl": perplexity_from_loss(loss)})

    # Packaging smoke test on a tiny temporary model directory.
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "toy_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type":"toy"}\n')
        (model_dir / "weights.bin").write_bytes(b"toy weights")
        artifact = Path(tmp) / "toy_model.zip"
        print("toy package:", zip_directory_deterministic(model_dir, artifact))


if __name__ == "__main__":
    toy_check()

"""Public/dev PPL and artifact packaging helpers.

This file is intentionally small and public-safe. It uses toy tensors by
default. Replace the toy logits with your own public/dev text and local
candidate model when you are ready.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import zipfile

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # Keep importable on machines without torch.
    torch = None
    F = None


def shifted_cross_entropy_from_logits(logits, labels, ignore_index: int = -100):
    """Compute next-token cross entropy from model logits.

    logits: [batch, sequence, vocab]
    labels: [batch, sequence]
    """
    if torch is None:
        raise RuntimeError("Install torch to compute cross entropy.")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


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

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            zf.write(path, path.relative_to(source_dir).as_posix())

    return {
        "zip_path": str(zip_path),
        "artifact_size_bytes": zip_path.stat().st_size,
        "artifact_sha256": sha256_file(zip_path),
    }


def toy_check() -> None:
    if torch is None:
        print("Install torch to run the toy PPL check.")
        return

    labels = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    logits = torch.zeros((2, 4, 5))
    for batch in range(labels.size(0)):
        for pos in range(labels.size(1) - 1):
            logits[batch, pos, labels[batch, pos + 1]] = 5.0

    loss = shifted_cross_entropy_from_logits(logits, labels)
    print({"loss": float(loss), "ppl": perplexity_from_loss(loss)})


if __name__ == "__main__":
    toy_check()

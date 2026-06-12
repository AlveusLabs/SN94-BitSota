"""Simple diagnostics for compression experiment triage."""

from __future__ import annotations

import math


def is_dominated(candidate: dict, others: list[dict]) -> bool:
    """True if another valid experiment is both smaller and lower-PPL."""
    for other in others:
        if other is candidate:
            continue
        smaller_or_equal = other["size_gb"] <= candidate["size_gb"]
        better_or_equal = other["public_ppl"] <= candidate["public_ppl"]
        strictly_better = (
            other["size_gb"] < candidate["size_gb"]
            or other["public_ppl"] < candidate["public_ppl"]
        )
        if smaller_or_equal and better_or_equal and strictly_better:
            return True
    return False


def frontier(experiments: list[dict]) -> list[dict]:
    valid = [
        e
        for e in experiments
        if e.get("loads") and math.isfinite(float(e.get("public_ppl", math.inf)))
    ]
    return [e for e in valid if not is_dominated(e, valid)]


def toy_check() -> None:
    experiments = [
        {"name": "pure_binary", "size_gb": 6.2, "public_ppl": 95.0, "loads": True},
        {"name": "binary_q4_rescue", "size_gb": 7.1, "public_ppl": 38.0, "loads": True},
        {"name": "ternary_distilled", "size_gb": 9.4, "public_ppl": 27.0, "loads": True},
        {"name": "proxy_only", "size_gb": 5.8, "public_ppl": 180.0, "loads": True},
        {"name": "broken_packaging", "size_gb": 7.0, "public_ppl": 30.0, "loads": False},
    ]
    for row in frontier(experiments):
        print(row)


if __name__ == "__main__":
    toy_check()

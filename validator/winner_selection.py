from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

TIE_EPSILON = 1e-9


@dataclass(frozen=True)
class WinnerSelection:
    result: dict[str, Any]
    best_score: float
    tied_count: int
    reason: str


def _score_of(result: Mapping[str, Any]) -> float:
    return float(result.get("validator_score", float("-inf")))


def _timestamp_key(result: Mapping[str, Any]) -> tuple[bool, str, str]:
    return (
        result.get("timestamp") is None,
        str(result.get("timestamp") or ""),
        str(result.get("miner_hotkey") or ""),
    )


def select_best_result(
    results: Sequence[dict[str, Any]],
    sota_score: float,
    *,
    current_local_winner_hotkey: Optional[str] = None,
    latest_sota_event: Optional[Mapping[str, Any]] = None,
    tie_epsilon: float = TIE_EPSILON,
) -> WinnerSelection:
    """Choose the best validated relay result with stable tie-breaking."""
    if not results:
        raise ValueError("select_best_result() requires at least one result")

    best_score = max(_score_of(result) for result in results)
    tied_results = [
        result for result in results if abs(_score_of(result) - best_score) <= tie_epsilon
    ]

    if len(tied_results) == 1:
        return WinnerSelection(
            result=tied_results[0],
            best_score=best_score,
            tied_count=1,
            reason="unique_best",
        )

    local_hotkey = str(current_local_winner_hotkey or "").strip()
    if local_hotkey:
        for candidate in tied_results:
            if str(candidate.get("miner_hotkey") or "") == local_hotkey:
                return WinnerSelection(
                    result=candidate,
                    best_score=best_score,
                    tied_count=len(tied_results),
                    reason="keep_local_winner",
                )

    if latest_sota_event is not None and abs(best_score - float(sota_score)) <= tie_epsilon:
        try:
            event_hotkey = str(latest_sota_event.get("miner_hotkey") or "").strip()
            event_score = float(latest_sota_event.get("score"))
        except Exception:
            event_hotkey = ""
            event_score = None
        if event_hotkey and event_score is not None and abs(event_score - best_score) <= tie_epsilon:
            for candidate in tied_results:
                if str(candidate.get("miner_hotkey") or "") == event_hotkey:
                    return WinnerSelection(
                        result=candidate,
                        best_score=best_score,
                        tied_count=len(tied_results),
                        reason="relay_frontier_winner",
                    )

    chosen = min(tied_results, key=_timestamp_key)
    return WinnerSelection(
        result=chosen,
        best_score=best_score,
        tied_count=len(tied_results),
        reason="oldest_submission",
    )

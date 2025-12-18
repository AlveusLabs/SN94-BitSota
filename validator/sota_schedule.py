from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class SOTAEvent:
    event_id: int
    miner_hotkey: str
    start_block: int
    end_block: int


def ceil_to_mod(block: int, mod: int) -> int:
    if mod <= 0:
        raise ValueError("mod must be > 0")
    r = block % mod
    return block if r == 0 else (block + (mod - r))


def active_event(events: Iterable[SOTAEvent], current_block: int) -> Optional[SOTAEvent]:
    """Return the active event at current_block, applying 'cutoff at next start'."""
    normalized = [
        e
        for e in events
        if isinstance(e.start_block, int)
        and isinstance(e.end_block, int)
        and e.start_block > 0
        and e.end_block > e.start_block
    ]
    if not normalized:
        return None

    ordered = sorted(normalized, key=lambda e: (e.start_block, e.event_id))

    for i, event in enumerate(ordered):
        next_start = ordered[i + 1].start_block if i + 1 < len(ordered) else None
        effective_end = min(event.end_block, next_start) if next_start is not None else event.end_block
        if event.start_block <= current_block < effective_end:
            return event
    return None


def target_hotkey(
    events: Iterable[SOTAEvent], current_block: int, burn_hotkey: str
) -> str:
    """Return miner hotkey if rewarding, else burn_hotkey."""
    event = active_event(events, current_block)
    return event.miner_hotkey if event else burn_hotkey


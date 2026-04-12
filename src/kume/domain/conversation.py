from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ConversationEvent:
    """A single message in a user's conversation with Kume."""

    id: str
    user_id: str
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime


def resolve_session(
    events: list[ConversationEvent],
    now: datetime,
    gap_seconds: int = 3600,
) -> list[ConversationEvent]:
    """Return events belonging to the current session.

    Walks backward from the most recent event. If the gap between
    any two consecutive events (or between now and the latest event)
    exceeds gap_seconds, everything before that gap is dropped.
    Returns events in chronological order.
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.created_at)

    # Check if the session has expired (gap between now and latest event)
    last = sorted_events[-1]
    if (now - last.created_at).total_seconds() > gap_seconds:
        return []

    # Walk backward through pairs looking for the last gap
    start = 0
    for i in range(len(sorted_events) - 1, 0, -1):
        gap = (sorted_events[i].created_at - sorted_events[i - 1].created_at).total_seconds()
        if gap > gap_seconds:
            start = i
            break

    return sorted_events[start:]

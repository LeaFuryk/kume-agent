from __future__ import annotations

from datetime import UTC, datetime

from kume.domain.conversation import ConversationEvent, resolve_session


class SessionStore:
    """In-memory per-user conversation history with 1-hour session gaps."""

    def __init__(self, gap_seconds: int = 3600) -> None:
        self._history: dict[str, list[ConversationEvent]] = {}
        self._gap_seconds = gap_seconds

    def add(self, user_id: str, event: ConversationEvent) -> None:
        """Append an event to a user's history."""
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(event)

    def get_session(self, user_id: str) -> list[ConversationEvent]:
        """Return events from the current session. Prunes stale events."""
        events = self._history.get(user_id, [])
        if not events:
            return []
        now = datetime.now(UTC)
        session = resolve_session(events, now, self._gap_seconds)
        # Prune: keep only session events to avoid memory leaks
        self._history[user_id] = session
        return session

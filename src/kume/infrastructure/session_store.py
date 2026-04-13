from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from kume.domain.conversation import ConversationEvent, resolve_session

# Users with no activity for this long are evicted from memory
_EVICTION_SECONDS = 7200  # 2 hours


class SessionStore:
    """In-memory per-user conversation history with 1-hour session gaps.

    Thread-safe under concurrent async requests via per-user locks.
    Inactive users are evicted periodically to prevent memory leaks.
    """

    def __init__(self, gap_seconds: int = 3600) -> None:
        self._history: dict[str, list[ConversationEvent]] = {}
        self._gap_seconds = gap_seconds
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_access: dict[str, datetime] = {}

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def add(self, user_id: str, event: ConversationEvent) -> None:
        """Append an event to a user's history."""
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(event)
        self._last_access[user_id] = datetime.now(UTC)

    def get_session(self, user_id: str) -> list[ConversationEvent]:
        """Return events from the current session. Prunes stale events."""
        events = self._history.get(user_id, [])
        if not events:
            return []
        now = datetime.now(UTC)
        session = resolve_session(events, now, self._gap_seconds)
        # Prune: keep only session events to avoid unbounded growth
        self._history[user_id] = list(session)
        self._last_access[user_id] = now
        # Opportunistic cleanup of inactive users
        self._evict_stale(now)
        return session

    def _evict_stale(self, now: datetime) -> None:
        """Remove users who haven't been active recently.

        Skips users whose lock is currently held (active request in progress).
        """
        stale = [
            uid
            for uid, last in self._last_access.items()
            if (now - last).total_seconds() > _EVICTION_SECONDS
            and not (uid in self._locks and self._locks[uid].locked())
        ]
        for uid in stale:
            self._history.pop(uid, None)
            self._last_access.pop(uid, None)
            self._locks.pop(uid, None)

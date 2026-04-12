from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from kume.domain.conversation import ConversationEvent
from kume.infrastructure.session_store import SessionStore


def _event(user_id: str, content: str, created_at: datetime, role: str = "user") -> ConversationEvent:
    return ConversationEvent(
        id=f"{user_id}-{content}",
        user_id=user_id,
        role=role,
        content=content,
        created_at=created_at,
    )


class TestSessionStoreAddAndGet:
    def test_add_and_get_session_returns_events(self) -> None:
        store = SessionStore()
        now = datetime.now(UTC)
        e1 = _event("u1", "hello", now - timedelta(minutes=5))
        e2 = _event("u1", "world", now - timedelta(minutes=2))

        store.add("u1", e1)
        store.add("u1", e2)

        with patch("kume.infrastructure.session_store.datetime") as mock_dt:
            mock_dt.now.return_value = now
            session = store.get_session("u1")

        assert len(session) == 2
        assert session[0].content == "hello"
        assert session[1].content == "world"

    def test_get_session_unknown_user_returns_empty(self) -> None:
        store = SessionStore()
        assert store.get_session("nonexistent") == []


class TestSessionGapPruning:
    def test_session_gap_prunes_old_events(self) -> None:
        store = SessionStore(gap_seconds=3600)
        now = datetime.now(UTC)

        # Old events (>1h before the latest)
        old1 = _event("u1", "old1", now - timedelta(hours=3))
        old2 = _event("u1", "old2", now - timedelta(hours=2))
        # Recent events (within the current session)
        new1 = _event("u1", "new1", now - timedelta(minutes=30))
        new2 = _event("u1", "new2", now - timedelta(minutes=10))

        for e in [old1, old2, new1, new2]:
            store.add("u1", e)

        with patch("kume.infrastructure.session_store.datetime") as mock_dt:
            mock_dt.now.return_value = now
            session = store.get_session("u1")

        assert len(session) == 2
        assert session[0].content == "new1"
        assert session[1].content == "new2"


class TestUserIsolation:
    def test_user_a_events_dont_leak_to_user_b(self) -> None:
        store = SessionStore()
        now = datetime.now(UTC)

        e_a = _event("alice", "alice msg", now - timedelta(minutes=5))
        e_b = _event("bob", "bob msg", now - timedelta(minutes=5))

        store.add("alice", e_a)
        store.add("bob", e_b)

        with patch("kume.infrastructure.session_store.datetime") as mock_dt:
            mock_dt.now.return_value = now
            alice_session = store.get_session("alice")
            bob_session = store.get_session("bob")

        assert len(alice_session) == 1
        assert alice_session[0].content == "alice msg"
        assert len(bob_session) == 1
        assert bob_session[0].content == "bob msg"


class TestMemoryCleanup:
    def test_stale_events_pruned_after_get_session(self) -> None:
        store = SessionStore(gap_seconds=3600)
        now = datetime.now(UTC)

        old = _event("u1", "old", now - timedelta(hours=3))
        new = _event("u1", "new", now - timedelta(minutes=5))

        store.add("u1", old)
        store.add("u1", new)

        # Before get_session, both events are stored
        assert len(store._history["u1"]) == 2

        with patch("kume.infrastructure.session_store.datetime") as mock_dt:
            mock_dt.now.return_value = now
            store.get_session("u1")

        # After get_session, only the session events remain
        assert len(store._history["u1"]) == 1
        assert store._history["u1"][0].content == "new"

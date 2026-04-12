from datetime import datetime, timedelta

import pytest

from kume.domain.conversation import ConversationEvent, resolve_session

NOW = datetime(2026, 1, 15, 10, 30, 0)


class TestConversationEvent:
    def test_creation(self) -> None:
        event = ConversationEvent(id="e-1", user_id="u-1", role="user", content="Hello", created_at=NOW)
        assert event.id == "e-1"
        assert event.user_id == "u-1"
        assert event.role == "user"
        assert event.content == "Hello"
        assert event.created_at == NOW

    def test_immutability(self) -> None:
        event = ConversationEvent(id="e-1", user_id="u-1", role="user", content="Hello", created_at=NOW)
        with pytest.raises(AttributeError):
            event.content = "Changed"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            event.role = "assistant"  # type: ignore[misc]

    def test_field_types(self) -> None:
        event = ConversationEvent(id="e-1", user_id="u-1", role="assistant", content="Hi!", created_at=NOW)
        assert isinstance(event.role, str)
        assert isinstance(event.content, str)
        assert isinstance(event.created_at, datetime)


def _make_event(id: str, minutes_ago: float) -> ConversationEvent:
    """Helper: create an event at NOW - minutes_ago minutes."""
    return ConversationEvent(
        id=id,
        user_id="u-1",
        role="user",
        content=f"msg-{id}",
        created_at=NOW - timedelta(minutes=minutes_ago),
    )


class TestResolveSession:
    def test_empty_list(self) -> None:
        assert resolve_session([], NOW) == []

    def test_single_event_within_gap(self) -> None:
        event = _make_event("e-1", minutes_ago=30)
        result = resolve_session([event], NOW)
        assert result == [event]

    def test_single_event_outside_gap(self) -> None:
        event = _make_event("e-1", minutes_ago=61)  # > 60 min = > 3600s
        result = resolve_session([event], NOW)
        assert result == []

    def test_all_events_within_gap(self) -> None:
        events = [
            _make_event("e-1", minutes_ago=50),
            _make_event("e-2", minutes_ago=40),
            _make_event("e-3", minutes_ago=20),
            _make_event("e-4", minutes_ago=5),
        ]
        result = resolve_session(events, NOW)
        assert result == sorted(events, key=lambda e: e.created_at)

    def test_gap_in_the_middle(self) -> None:
        events = [
            _make_event("e-1", minutes_ago=150),  # old session
            _make_event("e-2", minutes_ago=140),  # old session
            # --- gap of 80 min here (140 - 60 = 80 min gap) ---
            _make_event("e-3", minutes_ago=40),  # current session
            _make_event("e-4", minutes_ago=10),  # current session
        ]
        result = resolve_session(events, NOW)
        assert len(result) == 2
        assert result[0].id == "e-3"
        assert result[1].id == "e-4"

    def test_gap_between_now_and_latest_event(self) -> None:
        events = [
            _make_event("e-1", minutes_ago=130),
            _make_event("e-2", minutes_ago=120),
            _make_event("e-3", minutes_ago=90),  # latest event is 90 min ago > 1h
        ]
        result = resolve_session(events, NOW)
        assert result == []

    def test_exact_boundary_not_a_gap(self) -> None:
        # Event at exactly 3600s (60 min) ago — should still be in session
        event = _make_event("e-1", minutes_ago=60)  # exactly 3600s
        result = resolve_session([event], NOW)
        assert result == [event]

    def test_exact_boundary_between_events_not_a_gap(self) -> None:
        # Two events exactly 3600s apart — not a gap
        events = [
            _make_event("e-1", minutes_ago=70),  # 70 min ago
            _make_event("e-2", minutes_ago=10),  # 10 min ago; gap = 60 min = 3600s
        ]
        result = resolve_session(events, NOW)
        assert len(result) == 2
        assert result[0].id == "e-1"
        assert result[1].id == "e-2"

    def test_multiple_gaps_keeps_only_after_last(self) -> None:
        events = [
            _make_event("e-1", minutes_ago=300),  # very old
            # --- gap ---
            _make_event("e-2", minutes_ago=180),  # old session
            _make_event("e-3", minutes_ago=170),  # old session
            # --- gap ---
            _make_event("e-4", minutes_ago=30),  # current session
            _make_event("e-5", minutes_ago=10),  # current session
        ]
        result = resolve_session(events, NOW)
        assert len(result) == 2
        assert result[0].id == "e-4"
        assert result[1].id == "e-5"

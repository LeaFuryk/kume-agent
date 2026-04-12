from datetime import datetime

import pytest

from kume.domain.conversation import ConversationEvent

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

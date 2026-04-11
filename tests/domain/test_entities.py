import pytest

from kume.domain.entities import User


class TestUser:
    def test_creation(self) -> None:
        user = User(id="u-123", telegram_id=456)
        assert user.id == "u-123"
        assert user.telegram_id == 456

    def test_immutability(self) -> None:
        user = User(id="u-1", telegram_id=1)
        with pytest.raises(AttributeError):
            user.id = "u-2"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            user.telegram_id = 2  # type: ignore[misc]

    def test_equality(self) -> None:
        a = User(id="u-1", telegram_id=1)
        b = User(id="u-1", telegram_id=1)
        assert a == b

    def test_inequality(self) -> None:
        a = User(id="u-1", telegram_id=1)
        b = User(id="u-2", telegram_id=1)
        assert a != b

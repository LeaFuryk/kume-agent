from datetime import datetime

import pytest

from kume.domain.entities import Document, Goal, LabMarker, Restriction, User
from kume.ports.output.database import DatabasePort


def test_database_port_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        DatabasePort()  # type: ignore[abstract]


class _FakeDatabase(DatabasePort):
    async def get_or_create_user(self, telegram_id: int, name: str | None = None, language: str = "en") -> User:
        return User(id="u1", telegram_id=telegram_id, name=name, language=language)

    async def update_user(self, user: User) -> None:
        self.updated_user = user

    async def save_document(self, doc: Document) -> None:
        self.saved_doc = doc

    async def save_lab_markers(self, markers: list[LabMarker]) -> None:
        self.saved_markers = markers

    async def get_lab_markers(
        self,
        user_id: str,
        name: str | None = None,
        since: datetime | None = None,
    ) -> list[LabMarker]:
        return []

    async def save_goal(self, goal: Goal) -> None:
        self.saved_goal = goal

    async def get_goals(self, user_id: str, active_only: bool = True) -> list[Goal]:
        return []

    async def save_restriction(self, restriction: Restriction) -> None:
        self.saved_restriction = restriction

    async def get_restrictions(self, user_id: str, active_only: bool = True) -> list[Restriction]:
        return []


async def test_concrete_subclass_can_be_instantiated() -> None:
    db = _FakeDatabase()
    user = await db.get_or_create_user(telegram_id=123)
    assert user.telegram_id == 123
    assert user.id == "u1"


async def test_save_and_get_goals() -> None:
    db = _FakeDatabase()
    goal = Goal(
        id="g1",
        user_id="u1",
        description="lose weight",
        created_at=datetime(2026, 1, 1),
    )
    await db.save_goal(goal)
    assert db.saved_goal == goal
    assert await db.get_goals("u1") == []


async def test_save_and_get_restrictions() -> None:
    db = _FakeDatabase()
    restriction = Restriction(
        id="r1",
        user_id="u1",
        type="allergy",
        description="peanuts",
        created_at=datetime(2026, 1, 1),
    )
    await db.save_restriction(restriction)
    assert db.saved_restriction == restriction
    assert await db.get_restrictions("u1") == []


def test_database_port_importable_from_ports_package() -> None:
    from kume.ports import DatabasePort as FromPorts
    from kume.ports.output import DatabasePort as FromOutput

    assert FromPorts is FromOutput

from datetime import datetime

import pytest

from kume.domain.entities import Document, Goal, LabMarker, Meal, Restriction, User
from kume.ports.output.repositories import (
    DocumentRepository,
    GoalRepository,
    LabMarkerRepository,
    MealRepository,
    RestrictionRepository,
    UserRepository,
)

# --- Abstract classes cannot be instantiated ---


@pytest.mark.parametrize(
    "cls",
    [UserRepository, GoalRepository, RestrictionRepository, DocumentRepository, LabMarkerRepository, MealRepository],
    ids=[
        "UserRepository",
        "GoalRepository",
        "RestrictionRepository",
        "DocumentRepository",
        "LabMarkerRepository",
        "MealRepository",
    ],
)
def test_abstract_repository_cannot_be_instantiated(cls: type) -> None:
    with pytest.raises(TypeError):
        cls()  # type: ignore[abstract]


# --- Concrete subclasses ---


class _FakeUserRepository(UserRepository):
    async def get_or_create(self, telegram_id: int, name: str | None = None, language: str = "en") -> User:
        return User(id="u1", telegram_id=telegram_id, name=name, language=language)

    async def update(self, user: User) -> None:
        self.updated_user = user


class _FakeGoalRepository(GoalRepository):
    async def save(self, goal: Goal) -> None:
        self.saved_goal = goal

    async def get_by_user(self, user_id: str, active_only: bool = True) -> list[Goal]:
        return []


class _FakeRestrictionRepository(RestrictionRepository):
    async def save(self, restriction: Restriction) -> None:
        self.saved_restriction = restriction

    async def get_by_user(self, user_id: str, active_only: bool = True) -> list[Restriction]:
        return []


class _FakeDocumentRepository(DocumentRepository):
    async def save(self, doc: Document) -> None:
        self.saved_doc = doc


class _FakeLabMarkerRepository(LabMarkerRepository):
    async def save_many(self, markers: list[LabMarker]) -> None:
        self.saved_markers = markers

    async def get_by_user(
        self,
        user_id: str,
        name: str | None = None,
        since: datetime | None = None,
    ) -> list[LabMarker]:
        return []


async def test_user_repository_concrete() -> None:
    repo = _FakeUserRepository()
    user = await repo.get_or_create(telegram_id=123)
    assert user.telegram_id == 123
    assert user.id == "u1"


async def test_goal_repository_concrete() -> None:
    repo = _FakeGoalRepository()
    goal = Goal(id="g1", user_id="u1", description="lose weight", created_at=datetime(2026, 1, 1))
    await repo.save(goal)
    assert repo.saved_goal == goal
    assert await repo.get_by_user("u1") == []


async def test_restriction_repository_concrete() -> None:
    repo = _FakeRestrictionRepository()
    restriction = Restriction(
        id="r1", user_id="u1", type="allergy", description="peanuts", created_at=datetime(2026, 1, 1)
    )
    await repo.save(restriction)
    assert repo.saved_restriction == restriction
    assert await repo.get_by_user("u1") == []


async def test_document_repository_concrete() -> None:
    repo = _FakeDocumentRepository()
    doc = Document(
        id="d1",
        user_id="u1",
        type="lab_report",
        filename="report.pdf",
        summary="test",
        ingested_at=datetime(2026, 1, 1),
    )
    await repo.save(doc)
    assert repo.saved_doc == doc


async def test_lab_marker_repository_concrete() -> None:
    repo = _FakeLabMarkerRepository()
    marker = LabMarker(
        id="m1",
        document_id="d1",
        user_id="u1",
        name="GLUCOSE",
        value=90.0,
        unit="mg/dL",
        reference_range="70-100 mg/dL",
        date=datetime(2026, 1, 1),
    )
    await repo.save_many([marker])
    assert repo.saved_markers == [marker]
    assert await repo.get_by_user("u1") == []


class _FakeMealRepository(MealRepository):
    async def save(self, meal: Meal) -> None:
        self.saved_meal = meal

    async def get_by_user(
        self,
        user_id: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[Meal]:
        return []


async def test_meal_repository_concrete() -> None:
    repo = _FakeMealRepository()
    meal = Meal(
        id="meal1",
        user_id="u1",
        description="Grilled chicken with rice",
        calories=450.0,
        protein_g=35.0,
        carbs_g=50.0,
        fat_g=10.0,
        fiber_g=3.0,
        sodium_mg=600.0,
        sugar_g=2.0,
        saturated_fat_g=2.5,
        cholesterol_mg=85.0,
        confidence=0.85,
        image_present=True,
        logged_at=datetime(2026, 4, 12),
    )
    await repo.save(meal)
    assert repo.saved_meal == meal
    assert await repo.get_by_user("u1") == []


def test_repositories_importable_from_ports_package() -> None:
    from kume.ports import GoalRepository as FromPorts
    from kume.ports.output import GoalRepository as FromOutput

    assert FromPorts is FromOutput

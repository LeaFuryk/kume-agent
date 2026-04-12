"""Tests for PostgreSQL repository implementations using in-memory SQLite via aiosqlite."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kume.adapters.output.postgres_db import (
    PostgresDocumentRepository,
    PostgresGoalRepository,
    PostgresLabMarkerRepository,
    PostgresMealRepository,
    PostgresRestrictionRepository,
    PostgresUserRepository,
)
from kume.adapters.output.postgres_models import Base
from kume.domain.entities import Document, Goal, LabMarker, Meal, Restriction
from kume.ports.output.repositories import (
    DocumentRepository,
    GoalRepository,
    LabMarkerRepository,
    MealRepository,
    RestrictionRepository,
    UserRepository,
)


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = async_sessionmaker(engine, expire_on_commit=False)
    yield sf
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# --- Interface compliance ---


def test_user_repo_implements_interface() -> None:
    assert issubclass(PostgresUserRepository, UserRepository)


def test_goal_repo_implements_interface() -> None:
    assert issubclass(PostgresGoalRepository, GoalRepository)


def test_restriction_repo_implements_interface() -> None:
    assert issubclass(PostgresRestrictionRepository, RestrictionRepository)


def test_document_repo_implements_interface() -> None:
    assert issubclass(PostgresDocumentRepository, DocumentRepository)


def test_lab_marker_repo_implements_interface() -> None:
    assert issubclass(PostgresLabMarkerRepository, LabMarkerRepository)


def test_meal_repo_implements_interface() -> None:
    assert issubclass(PostgresMealRepository, MealRepository)


# --- UserRepository ---


async def test_user_create_new(session_factory) -> None:
    repo = PostgresUserRepository(session_factory)
    user = await repo.get_or_create(telegram_id=12345, name="Leandro", language="es")
    assert user.telegram_id == 12345
    assert user.name == "Leandro"
    assert user.language == "es"


async def test_user_get_existing(session_factory) -> None:
    repo = PostgresUserRepository(session_factory)
    u1 = await repo.get_or_create(telegram_id=12345)
    u2 = await repo.get_or_create(telegram_id=12345)
    assert u1.id == u2.id


async def test_user_update(session_factory) -> None:
    repo = PostgresUserRepository(session_factory)
    user = await repo.get_or_create(telegram_id=12345, name="Old Name")
    from dataclasses import replace

    updated = replace(user, name="New Name", language="es")
    await repo.update(updated)
    fetched = await repo.get_or_create(telegram_id=12345)
    assert fetched.name == "New Name"
    assert fetched.language == "es"


# --- GoalRepository ---


async def test_goal_save_and_get(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    user = await user_repo.get_or_create(telegram_id=1)
    goal_repo = PostgresGoalRepository(session_factory)

    goal = Goal(id=str(uuid.uuid4()), user_id=user.id, description="Lose 5kg", created_at=datetime.now(tz=UTC))
    await goal_repo.save(goal)

    goals = await goal_repo.get_by_user(user.id)
    assert len(goals) == 1
    assert goals[0].description == "Lose 5kg"


async def test_goal_active_only_filter(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    user = await user_repo.get_or_create(telegram_id=1)
    goal_repo = PostgresGoalRepository(session_factory)

    active = Goal(id=str(uuid.uuid4()), user_id=user.id, description="Active", created_at=datetime.now(tz=UTC))
    completed = Goal(
        id=str(uuid.uuid4()),
        user_id=user.id,
        description="Done",
        created_at=datetime.now(tz=UTC),
        completed_at=datetime.now(tz=UTC),
    )
    await goal_repo.save(active)
    await goal_repo.save(completed)

    active_goals = await goal_repo.get_by_user(user.id, active_only=True)
    assert len(active_goals) == 1
    assert active_goals[0].description == "Active"

    all_goals = await goal_repo.get_by_user(user.id, active_only=False)
    assert len(all_goals) == 2


async def test_goal_user_isolation(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    u1 = await user_repo.get_or_create(telegram_id=1)
    u2 = await user_repo.get_or_create(telegram_id=2)
    goal_repo = PostgresGoalRepository(session_factory)

    await goal_repo.save(
        Goal(id=str(uuid.uuid4()), user_id=u1.id, description="U1 goal", created_at=datetime.now(tz=UTC))
    )
    await goal_repo.save(
        Goal(id=str(uuid.uuid4()), user_id=u2.id, description="U2 goal", created_at=datetime.now(tz=UTC))
    )

    assert len(await goal_repo.get_by_user(u1.id)) == 1
    assert len(await goal_repo.get_by_user(u2.id)) == 1


# --- RestrictionRepository ---


async def test_restriction_save_and_get(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    user = await user_repo.get_or_create(telegram_id=1)
    repo = PostgresRestrictionRepository(session_factory)

    r = Restriction(
        id=str(uuid.uuid4()), user_id=user.id, type="allergy", description="Peanuts", created_at=datetime.now(tz=UTC)
    )
    await repo.save(r)
    restrictions = await repo.get_by_user(user.id)
    assert len(restrictions) == 1
    assert restrictions[0].type == "allergy"


# --- DocumentRepository ---


async def test_document_save(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    user = await user_repo.get_or_create(telegram_id=1)
    repo = PostgresDocumentRepository(session_factory)

    doc = Document(
        id=str(uuid.uuid4()),
        user_id=user.id,
        type="lab_report",
        filename="test.pdf",
        summary="Test report",
        ingested_at=datetime.now(tz=UTC),
    )
    await repo.save(doc)


# --- LabMarkerRepository ---


async def test_lab_markers_save_and_get(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    user = await user_repo.get_or_create(telegram_id=1)
    doc_repo = PostgresDocumentRepository(session_factory)
    doc = Document(
        id=str(uuid.uuid4()),
        user_id=user.id,
        type="lab_report",
        filename="test.pdf",
        summary="Test",
        ingested_at=datetime.now(tz=UTC),
    )
    await doc_repo.save(doc)

    marker_repo = PostgresLabMarkerRepository(session_factory)
    marker = LabMarker(
        id=str(uuid.uuid4()),
        document_id=doc.id,
        user_id=user.id,
        name="CHOLESTEROL",
        value=200.0,
        unit="mg/dL",
        reference_range="< 200",
        date=datetime.now(tz=UTC),
    )
    await marker_repo.save_many([marker])
    markers = await marker_repo.get_by_user(user.id)
    assert len(markers) == 1
    assert markers[0].name == "CHOLESTEROL"


# --- MealRepository ---


def _make_meal(user_id: str, description: str = "Grilled chicken salad", logged_at: datetime | None = None) -> Meal:
    return Meal(
        id=str(uuid.uuid4()),
        user_id=user_id,
        description=description,
        calories=450.0,
        protein_g=35.0,
        carbs_g=20.0,
        fat_g=15.0,
        fiber_g=5.0,
        sodium_mg=600.0,
        sugar_g=3.0,
        saturated_fat_g=4.0,
        cholesterol_mg=85.0,
        confidence=0.85,
        image_present=True,
        logged_at=logged_at or datetime.now(tz=UTC),
    )


async def test_meal_save_and_get(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    user = await user_repo.get_or_create(telegram_id=1)
    meal_repo = PostgresMealRepository(session_factory)

    meal = _make_meal(user.id)
    await meal_repo.save(meal)

    meals = await meal_repo.get_by_user(user.id)
    assert len(meals) == 1
    assert meals[0].description == "Grilled chicken salad"
    assert meals[0].calories == 450.0
    assert meals[0].protein_g == 35.0
    assert meals[0].image_present is True
    assert meals[0].confidence == 0.85


async def test_meal_get_by_user_filters_by_date(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    user = await user_repo.get_or_create(telegram_id=1)
    meal_repo = PostgresMealRepository(session_factory)

    old_meal = _make_meal(user.id, description="Old meal", logged_at=datetime(2024, 1, 1, tzinfo=UTC))
    recent_meal = _make_meal(user.id, description="Recent meal", logged_at=datetime(2024, 6, 15, tzinfo=UTC))
    await meal_repo.save(old_meal)
    await meal_repo.save(recent_meal)

    since = datetime(2024, 6, 1, tzinfo=UTC)
    meals = await meal_repo.get_by_user(user.id, since=since)
    assert len(meals) == 1
    assert meals[0].description == "Recent meal"


async def test_meal_get_by_user_with_limit(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    user = await user_repo.get_or_create(telegram_id=1)
    meal_repo = PostgresMealRepository(session_factory)

    for i in range(5):
        meal = _make_meal(user.id, description=f"Meal {i}", logged_at=datetime(2024, 6, i + 1, tzinfo=UTC))
        await meal_repo.save(meal)

    meals = await meal_repo.get_by_user(user.id, limit=3)
    assert len(meals) == 3
    # Results ordered by logged_at desc, so most recent first
    assert meals[0].description == "Meal 4"


async def test_meal_user_isolation(session_factory) -> None:
    user_repo = PostgresUserRepository(session_factory)
    u1 = await user_repo.get_or_create(telegram_id=1)
    u2 = await user_repo.get_or_create(telegram_id=2)
    meal_repo = PostgresMealRepository(session_factory)

    await meal_repo.save(_make_meal(u1.id, description="U1 meal"))
    await meal_repo.save(_make_meal(u2.id, description="U2 meal"))

    u1_meals = await meal_repo.get_by_user(u1.id)
    u2_meals = await meal_repo.get_by_user(u2.id)

    assert len(u1_meals) == 1
    assert u1_meals[0].description == "U1 meal"
    assert len(u2_meals) == 1
    assert u2_meals[0].description == "U2 meal"


async def test_meal_empty_results(session_factory) -> None:
    meal_repo = PostgresMealRepository(session_factory)
    meals = await meal_repo.get_by_user("nonexistent-user-id")
    assert meals == []

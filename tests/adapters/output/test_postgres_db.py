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
    PostgresRestrictionRepository,
    PostgresUserRepository,
)
from kume.adapters.output.postgres_models import Base
from kume.domain.entities import Document, Goal, LabMarker, Restriction
from kume.ports.output.repositories import (
    DocumentRepository,
    GoalRepository,
    LabMarkerRepository,
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

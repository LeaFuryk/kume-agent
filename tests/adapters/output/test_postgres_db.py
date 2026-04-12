"""Tests for PostgresAdapter using in-memory SQLite via aiosqlite."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kume.adapters.output.postgres_db import PostgresAdapter
from kume.adapters.output.postgres_models import Base
from kume.domain.entities import Document, Goal, LabMarker, Restriction
from kume.ports.output.database import DatabasePort

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def adapter():
    """Create a PostgresAdapter backed by an in-memory SQLite database."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    db = PostgresAdapter.__new__(PostgresAdapter)
    db._engine = engine
    db._session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield db

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------


def test_implements_database_port():
    assert issubclass(PostgresAdapter, DatabasePort)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


async def test_get_or_create_user_creates_new_user(adapter: PostgresAdapter):
    user = await adapter.get_or_create_user(telegram_id=123456, name="Alice", language="es")
    assert user.telegram_id == 123456
    assert user.name == "Alice"
    assert user.language == "es"
    assert user.timezone == "UTC"
    assert user.id  # should have a UUID


async def test_get_or_create_user_returns_existing(adapter: PostgresAdapter):
    user1 = await adapter.get_or_create_user(telegram_id=123456, name="Alice")
    user2 = await adapter.get_or_create_user(telegram_id=123456, name="Bob")
    assert user1.id == user2.id
    # The name should remain as originally created
    assert user2.name == "Alice"


async def test_update_user(adapter: PostgresAdapter):
    user = await adapter.get_or_create_user(telegram_id=999, name="Carol", language="en")
    from dataclasses import replace

    updated_user = replace(user, name="Carolina", language="pt", timezone="America/Sao_Paulo")
    await adapter.update_user(updated_user)

    fetched = await adapter.get_or_create_user(telegram_id=999)
    assert fetched.name == "Carolina"
    assert fetched.language == "pt"
    assert fetched.timezone == "America/Sao_Paulo"


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


async def test_save_and_get_goals_active_only(adapter: PostgresAdapter):
    user = await adapter.get_or_create_user(telegram_id=1)
    now = datetime.now(UTC)

    active_goal = Goal(
        id=str(uuid.uuid4()),
        user_id=user.id,
        description="Lose 5 kg",
        created_at=now,
        completed_at=None,
    )
    completed_goal = Goal(
        id=str(uuid.uuid4()),
        user_id=user.id,
        description="Eat more vegetables",
        created_at=now,
        completed_at=now,
    )

    await adapter.save_goal(active_goal)
    await adapter.save_goal(completed_goal)

    active = await adapter.get_goals(user.id, active_only=True)
    assert len(active) == 1
    assert active[0].description == "Lose 5 kg"

    all_goals = await adapter.get_goals(user.id, active_only=False)
    assert len(all_goals) == 2


async def test_get_goals_scoped_to_user(adapter: PostgresAdapter):
    """Goals for one user must not leak to another (rule #6)."""
    user_a = await adapter.get_or_create_user(telegram_id=10)
    user_b = await adapter.get_or_create_user(telegram_id=20)

    goal = Goal(
        id=str(uuid.uuid4()),
        user_id=user_a.id,
        description="A's goal",
        created_at=datetime.now(UTC),
    )
    await adapter.save_goal(goal)

    assert len(await adapter.get_goals(user_a.id)) == 1
    assert len(await adapter.get_goals(user_b.id)) == 0


# ---------------------------------------------------------------------------
# Restrictions
# ---------------------------------------------------------------------------


async def test_save_and_get_restrictions(adapter: PostgresAdapter):
    user = await adapter.get_or_create_user(telegram_id=2)
    now = datetime.now(UTC)

    active = Restriction(
        id=str(uuid.uuid4()),
        user_id=user.id,
        type="allergy",
        description="Peanuts",
        created_at=now,
    )
    completed = Restriction(
        id=str(uuid.uuid4()),
        user_id=user.id,
        type="diet",
        description="Keto",
        created_at=now,
        completed_at=now,
    )

    await adapter.save_restriction(active)
    await adapter.save_restriction(completed)

    active_only = await adapter.get_restrictions(user.id, active_only=True)
    assert len(active_only) == 1
    assert active_only[0].description == "Peanuts"

    all_restrictions = await adapter.get_restrictions(user.id, active_only=False)
    assert len(all_restrictions) == 2


# ---------------------------------------------------------------------------
# Documents & Lab Markers
# ---------------------------------------------------------------------------


async def test_save_document(adapter: PostgresAdapter):
    user = await adapter.get_or_create_user(telegram_id=3)
    doc = Document(
        id=str(uuid.uuid4()),
        user_id=user.id,
        type="lab_report",
        filename="blood_test.pdf",
        summary="Routine blood work",
        ingested_at=datetime.now(UTC),
    )
    # Should not raise
    await adapter.save_document(doc)


async def test_save_and_get_lab_markers(adapter: PostgresAdapter):
    user = await adapter.get_or_create_user(telegram_id=4)
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        user_id=user.id,
        type="lab_report",
        filename="labs.pdf",
        summary="Lab results",
        ingested_at=datetime.now(UTC),
    )
    await adapter.save_document(doc)

    now = datetime.now(UTC)
    markers = [
        LabMarker(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            user_id=user.id,
            name="CHOLESTEROL",
            value=195.0,
            unit="mg/dL",
            reference_range="< 200 mg/dL",
            date=now,
        ),
        LabMarker(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            user_id=user.id,
            name="GLUCOSE",
            value=90.0,
            unit="mg/dL",
            reference_range="70-100 mg/dL",
            date=now,
        ),
    ]
    await adapter.save_lab_markers(markers)

    results = await adapter.get_lab_markers(user.id)
    assert len(results) == 2

    by_name = await adapter.get_lab_markers(user.id, name="CHOLESTEROL")
    assert len(by_name) == 1
    assert by_name[0].value == 195.0


async def test_get_lab_markers_since_filter(adapter: PostgresAdapter):
    user = await adapter.get_or_create_user(telegram_id=5)
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        user_id=user.id,
        type="lab_report",
        filename="labs.pdf",
        summary="Labs",
        ingested_at=datetime.now(UTC),
    )
    await adapter.save_document(doc)

    old_date = datetime(2024, 1, 1)
    recent_date = datetime(2025, 6, 1)

    old_marker = LabMarker(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        user_id=user.id,
        name="IRON",
        value=60.0,
        unit="mcg/dL",
        reference_range="60-170 mcg/dL",
        date=old_date,
    )
    recent_marker = LabMarker(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        user_id=user.id,
        name="IRON",
        value=80.0,
        unit="mcg/dL",
        reference_range="60-170 mcg/dL",
        date=recent_date,
    )
    await adapter.save_lab_markers([old_marker, recent_marker])

    since = datetime(2025, 1, 1)
    results = await adapter.get_lab_markers(user.id, since=since)
    assert len(results) == 1
    assert results[0].value == 80.0


async def test_get_lab_markers_scoped_to_user(adapter: PostgresAdapter):
    """Lab markers for one user must not leak to another (rule #6)."""
    user_a = await adapter.get_or_create_user(telegram_id=100)
    user_b = await adapter.get_or_create_user(telegram_id=200)

    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        user_id=user_a.id,
        type="lab_report",
        filename="a.pdf",
        summary="A's labs",
        ingested_at=datetime.now(UTC),
    )
    await adapter.save_document(doc)

    marker = LabMarker(
        id=str(uuid.uuid4()),
        document_id=doc_id,
        user_id=user_a.id,
        name="HDL",
        value=55.0,
        unit="mg/dL",
        reference_range="> 40 mg/dL",
        date=datetime.now(UTC),
    )
    await adapter.save_lab_markers([marker])

    assert len(await adapter.get_lab_markers(user_a.id)) == 1
    assert len(await adapter.get_lab_markers(user_b.id)) == 0

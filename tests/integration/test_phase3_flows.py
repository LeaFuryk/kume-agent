"""Integration tests verifying full Phase 3 flows with mocked external services."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kume.adapters.output.openai_vision import OpenAIVisionAdapter
from kume.adapters.output.postgres_db import PostgresMealRepository, PostgresUserRepository
from kume.adapters.output.postgres_models import Base
from kume.domain.context import ContextBuilder, ContextDataProvider
from kume.domain.conversation import ConversationEvent
from kume.domain.entities import Meal
from kume.infrastructure.image_store import ImageStore
from kume.infrastructure.session_store import SessionStore
from kume.ports.output.vision import VisionPort

# ---------------------------------------------------------------------------
# Test 1 — SessionStore preserves history within a session
# ---------------------------------------------------------------------------


async def test_session_store_preserves_history_within_session() -> None:
    store = SessionStore()
    user_id = "u-1"
    now = datetime.now(UTC)

    # First message
    store.add(
        user_id,
        ConversationEvent(id="e1", user_id=user_id, role="user", content="Hello", created_at=now),
    )
    store.add(
        user_id,
        ConversationEvent(id="e2", user_id=user_id, role="assistant", content="Hi!", created_at=now),
    )

    # Second message 10 minutes later
    later = now + timedelta(minutes=10)
    store.add(
        user_id,
        ConversationEvent(
            id="e3",
            user_id=user_id,
            role="user",
            content="What did I say?",
            created_at=later,
        ),
    )

    # Session should contain all 3 events
    session = store.get_session(user_id)
    assert len(session) == 3
    assert session[0].content == "Hello"
    assert session[2].content == "What did I say?"


# ---------------------------------------------------------------------------
# Test 2 — Session resets after one-hour gap
# ---------------------------------------------------------------------------


async def test_session_resets_after_one_hour_gap() -> None:
    store = SessionStore()
    user_id = "u-1"
    old_time = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)

    store.add(
        user_id,
        ConversationEvent(id="e1", user_id=user_id, role="user", content="Old message", created_at=old_time),
    )
    store.add(
        user_id,
        ConversationEvent(id="e2", user_id=user_id, role="assistant", content="Old reply", created_at=old_time),
    )

    # get_session uses datetime.now() which will be >1h after old_time
    session = store.get_session(user_id)
    assert len(session) == 0  # session expired


# ---------------------------------------------------------------------------
# Test 3 — ImageStore lifecycle (set, get, clear)
# ---------------------------------------------------------------------------


async def test_image_store_lifecycle() -> None:
    store = ImageStore()
    request_id = "req-1"
    img = b"fake-image-bytes"

    store.set_images(request_id, [img])
    assert store.get_image(request_id, 1) == img

    store.clear(request_id)
    assert store.get_image(request_id, 1) is None


# ---------------------------------------------------------------------------
# Test 4 — ContextBuilder includes meals
# ---------------------------------------------------------------------------


class _FakeContextDataProvider(ContextDataProvider):
    """Provider that returns a single meal and nothing else."""

    def __init__(self, meals: list[Any] | None = None) -> None:
        self._meals = meals or []

    async def get_goals(self, user_id: str) -> list[Any]:
        return []

    async def get_restrictions(self, user_id: str) -> list[Any]:
        return []

    async def get_lab_markers(self, user_id: str) -> list[Any]:
        return []

    async def search_documents(self, user_id: str, query: str) -> list[str]:
        return []

    async def get_recent_meals(self, user_id: str) -> list[Any]:
        return self._meals


async def test_context_builder_includes_meals() -> None:
    meal = Meal(
        id=str(uuid.uuid4()),
        user_id="u-1",
        description="Grilled salmon with rice",
        calories=520.0,
        protein_g=40.0,
        carbs_g=45.0,
        fat_g=18.0,
        fiber_g=3.0,
        sodium_mg=400.0,
        sugar_g=2.0,
        saturated_fat_g=3.5,
        cholesterol_mg=75.0,
        confidence=0.9,
        image_present=False,
        logged_at=datetime.now(UTC),
    )
    provider = _FakeContextDataProvider(meals=[meal])
    builder = ContextBuilder(provider=provider)

    context = await builder.build("u-1", "How am I doing today?")

    # Verify Recent Meals section is present
    assert "## Recent Meals" in context
    assert "Grilled salmon with rice" in context
    assert "520.0 kcal" in context
    assert "40.0g protein" in context
    assert "45.0g carbs" in context
    assert "18.0g fat" in context

    # Verify Current Question appears after meals (correct ordering)
    meals_pos = context.index("## Recent Meals")
    question_pos = context.index("## Current Question")
    assert meals_pos < question_pos


# ---------------------------------------------------------------------------
# Test 5 — MealRepository round-trip via SQLite
# ---------------------------------------------------------------------------


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


async def test_meal_repository_round_trip(session_factory) -> None:
    # Create a user first (FK constraint)
    user_repo = PostgresUserRepository(session_factory)
    user = await user_repo.get_or_create(telegram_id=42, name="Test User")

    meal_repo = PostgresMealRepository(session_factory)
    now = datetime.now(UTC)

    meal = Meal(
        id=str(uuid.uuid4()),
        user_id=user.id,
        description="Chicken Caesar salad",
        calories=380.0,
        protein_g=32.0,
        carbs_g=12.0,
        fat_g=22.0,
        fiber_g=4.0,
        sodium_mg=850.0,
        sugar_g=3.0,
        saturated_fat_g=6.0,
        cholesterol_mg=95.0,
        confidence=0.88,
        image_present=True,
        logged_at=now,
    )
    await meal_repo.save(meal)

    # Retrieve and verify all nutritional fields
    meals = await meal_repo.get_by_user(user.id)
    assert len(meals) == 1

    retrieved = meals[0]
    assert retrieved.description == "Chicken Caesar salad"
    assert retrieved.calories == 380.0
    assert retrieved.protein_g == 32.0
    assert retrieved.carbs_g == 12.0
    assert retrieved.fat_g == 22.0
    assert retrieved.fiber_g == 4.0
    assert retrieved.sodium_mg == 850.0
    assert retrieved.sugar_g == 3.0
    assert retrieved.saturated_fat_g == 6.0
    assert retrieved.cholesterol_mg == 95.0
    assert retrieved.confidence == 0.88
    assert retrieved.image_present is True


# ---------------------------------------------------------------------------
# Test 6 — VisionPort contract compliance
# ---------------------------------------------------------------------------


def test_vision_port_contract() -> None:
    adapter = OpenAIVisionAdapter(api_key="fake-key")
    assert isinstance(adapter, VisionPort)
    assert hasattr(adapter, "analyze_image")

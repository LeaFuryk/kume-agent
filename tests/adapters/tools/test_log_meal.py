from datetime import UTC, datetime

import pytest

from kume.adapters.tools.log_meal import LogMealTool
from kume.domain.entities import Meal
from kume.infrastructure.request_context import RequestContext, _current, set_context
from tests.adapters.tools.conftest import FakeMealRepository


class TestLogMealTool:
    def _make_tool(self, user_id: str = "u1") -> tuple[LogMealTool, FakeMealRepository]:
        repo = FakeMealRepository()
        tool = LogMealTool(meal_repo=repo)
        set_context(RequestContext(user_id=user_id, telegram_id=1, language="en"))
        return tool, repo

    def _meal_kwargs(self, **overrides) -> dict:
        defaults = {
            "description": "Grilled chicken with rice",
            "calories": 550.0,
            "protein_g": 40.0,
            "carbs_g": 60.0,
            "fat_g": 12.0,
        }
        defaults.update(overrides)
        return defaults

    @pytest.mark.asyncio
    async def test_saves_meal_with_correct_fields(self) -> None:
        tool, repo = self._make_tool(user_id="u42")
        await tool.ainvoke(
            self._meal_kwargs(
                fiber_g=5.0,
                sodium_mg=800.0,
                sugar_g=3.0,
                saturated_fat_g=2.0,
                cholesterol_mg=90.0,
                confidence=0.9,
                image_present=True,
            )
        )
        assert len(repo.saved_meals) == 1
        meal = repo.saved_meals[0]
        assert isinstance(meal, Meal)
        assert meal.user_id == "u42"
        assert meal.description == "Grilled chicken with rice"
        assert meal.calories == 550.0
        assert meal.protein_g == 40.0
        assert meal.carbs_g == 60.0
        assert meal.fat_g == 12.0
        assert meal.fiber_g == 5.0
        assert meal.sodium_mg == 800.0
        assert meal.sugar_g == 3.0
        assert meal.saturated_fat_g == 2.0
        assert meal.cholesterol_mg == 90.0
        assert meal.confidence == 0.9
        assert meal.image_present is True
        assert meal.id  # UUID generated

    @pytest.mark.asyncio
    async def test_returns_confirmation(self) -> None:
        tool, _repo = self._make_tool()
        result = await tool.ainvoke(self._meal_kwargs())
        assert "Meal logged" in result
        assert "Grilled chicken with rice" in result
        assert "550 kcal" in result
        assert "40g protein" in result
        assert "60g carbs" in result
        assert "12g fat" in result

    @pytest.mark.asyncio
    async def test_no_user_context_returns_error(self) -> None:
        repo = FakeMealRepository()
        tool = LogMealTool(meal_repo=repo)
        _current.set(None)
        result = await tool.ainvoke(self._meal_kwargs())
        assert "Error" in result
        assert len(repo.saved_meals) == 0

    def test_user_id_not_in_schema(self) -> None:
        tool, _repo = self._make_tool()
        schema_fields = tool.args_schema.model_fields
        assert "user_id" not in schema_fields

    @pytest.mark.asyncio
    async def test_default_logged_at_is_now(self) -> None:
        tool, repo = self._make_tool()
        before = datetime.now(tz=UTC)
        await tool.ainvoke(self._meal_kwargs())
        after = datetime.now(tz=UTC)
        meal = repo.saved_meals[0]
        assert before <= meal.logged_at <= after

    @pytest.mark.asyncio
    async def test_custom_logged_at(self) -> None:
        tool, repo = self._make_tool()
        await tool.ainvoke(self._meal_kwargs(logged_at="2025-06-15T12:30:00"))
        meal = repo.saved_meals[0]
        assert meal.logged_at == datetime(2025, 6, 15, 12, 30, tzinfo=UTC)

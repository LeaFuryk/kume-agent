"""Tests for RequestReportTool (daily nutrition summary)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kume.adapters.tools.daily_summary import RequestReportTool
from kume.domain.entities import Goal, Meal
from kume.infrastructure.request_context import RequestContext, _current, set_context
from tests.adapters.tools.conftest import FakeGoalRepository, FakeMealRepository


def _make_meal(user_id: str = "u1", calories: float = 500.0, protein_g: float = 30.0) -> Meal:
    return Meal(
        id="meal-1",
        user_id=user_id,
        description="Test meal",
        calories=calories,
        protein_g=protein_g,
        carbs_g=50.0,
        fat_g=15.0,
        fiber_g=5.0,
        sodium_mg=200.0,
        sugar_g=10.0,
        saturated_fat_g=3.0,
        cholesterol_mg=50.0,
        confidence=0.9,
        image_present=False,
        logged_at=datetime.now(UTC),
    )


def _make_goal(user_id: str = "u1") -> Goal:
    return Goal(
        id="goal-1",
        user_id=user_id,
        description="Eat 2000 kcal per day",
        created_at=datetime.now(UTC),
    )


class TestDailySummaryTool:
    def _make_tool(
        self, user_id: str = "u1", meals: list[Meal] | None = None, goals: list[Goal] | None = None
    ) -> RequestReportTool:
        meal_repo = FakeMealRepository()
        if meals:
            for m in meals:
                meal_repo.saved_meals.append(m)

        goal_repo = FakeGoalRepository()
        if goals:
            goal_repo.saved_goals = list(goals)
            _original_get_by_user = goal_repo.get_by_user

            async def _get_by_user_with_goals(uid: str, active_only: bool = True) -> list[Goal]:
                return [g for g in goal_repo.saved_goals if g.user_id == uid]

            goal_repo.get_by_user = _get_by_user_with_goals  # type: ignore[assignment]

        set_context(RequestContext(user_id=user_id, telegram_id=1, language="en"))
        return RequestReportTool(meal_repo=meal_repo, goal_repo=goal_repo)

    @pytest.mark.asyncio
    async def test_daily_summary_with_meals(self) -> None:
        meals = [
            _make_meal(user_id="u1", calories=500.0, protein_g=30.0),
            Meal(
                id="meal-2",
                user_id="u1",
                description="Second meal",
                calories=700.0,
                protein_g=45.0,
                carbs_g=80.0,
                fat_g=20.0,
                fiber_g=8.0,
                sodium_mg=300.0,
                sugar_g=15.0,
                saturated_fat_g=5.0,
                cholesterol_mg=70.0,
                confidence=0.85,
                image_present=False,
                logged_at=datetime.now(UTC),
            ),
        ]
        tool = self._make_tool(user_id="u1", meals=meals)
        result = await tool.ainvoke({"date": "today"})

        assert "Daily Summary" in result
        assert "Meals logged: 2" in result
        assert "1,200 kcal" in result  # 500 + 700
        assert "75g" in result  # 30 + 45 protein

    @pytest.mark.asyncio
    async def test_daily_summary_no_meals(self) -> None:
        tool = self._make_tool(user_id="u1", meals=[])
        result = await tool.ainvoke({"date": "today"})

        assert "No meals logged today" in result

    @pytest.mark.asyncio
    async def test_daily_summary_with_goals(self) -> None:
        meals = [_make_meal(user_id="u1", calories=800.0, protein_g=50.0)]
        goals = [_make_goal(user_id="u1")]
        tool = self._make_tool(user_id="u1", meals=meals, goals=goals)
        result = await tool.ainvoke({"date": "today"})

        assert "Daily Summary" in result
        assert "Eat 2000 kcal per day" in result
        assert "Active goals:" in result

    @pytest.mark.asyncio
    async def test_daily_summary_no_context(self) -> None:
        meal_repo = FakeMealRepository()
        goal_repo = FakeGoalRepository()
        tool = RequestReportTool(meal_repo=meal_repo, goal_repo=goal_repo)
        _current.set(None)
        result = await tool.ainvoke({"date": "today"})

        assert "no user context" in result.lower()

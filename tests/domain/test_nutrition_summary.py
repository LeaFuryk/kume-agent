from __future__ import annotations

from datetime import UTC, datetime

from kume.domain.entities import Goal, Meal
from kume.domain.nutrition_summary import NutritionTotals, aggregate_nutrition, compare_against_goals


def _make_meal(**overrides: object) -> Meal:
    defaults = dict(
        id="m1", user_id="u1", description="test meal",
        calories=500.0, protein_g=30.0, carbs_g=60.0, fat_g=20.0, fiber_g=5.0,
        sodium_mg=400.0, sugar_g=10.0, saturated_fat_g=5.0, cholesterol_mg=50.0,
        confidence=0.9, image_present=False, logged_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return Meal(**defaults)


def test_aggregate_nutrition_single_meal() -> None:
    meals = [_make_meal(calories=500, protein_g=30, carbs_g=60, fat_g=20, fiber_g=5)]
    totals = aggregate_nutrition(meals)
    assert totals.calories == 500.0
    assert totals.protein_g == 30.0
    assert totals.carbs_g == 60.0
    assert totals.fat_g == 20.0
    assert totals.fiber_g == 5.0
    assert totals.meal_count == 1


def test_aggregate_nutrition_multiple_meals() -> None:
    meals = [
        _make_meal(id="m1", calories=500, protein_g=30, carbs_g=60, fat_g=20, fiber_g=5),
        _make_meal(id="m2", calories=300, protein_g=20, carbs_g=40, fat_g=10, fiber_g=3),
    ]
    totals = aggregate_nutrition(meals)
    assert totals.calories == 800.0
    assert totals.protein_g == 50.0
    assert totals.meal_count == 2


def test_aggregate_nutrition_empty_list() -> None:
    totals = aggregate_nutrition([])
    assert totals.calories == 0.0
    assert totals.meal_count == 0


def test_compare_against_goals_on_track() -> None:
    totals = NutritionTotals(calories=1800, protein_g=100, carbs_g=220, fat_g=60, fiber_g=25, meal_count=3)
    goals = [Goal(id="g1", user_id="u1", description="Eat 2000 calories per day", created_at=datetime.now(UTC))]
    result = compare_against_goals(totals, goals)
    assert "1,800" in result or "1800" in result
    assert "3" in result


def test_compare_against_goals_no_goals() -> None:
    totals = NutritionTotals(calories=1800, protein_g=100, carbs_g=220, fat_g=60, fiber_g=25, meal_count=3)
    result = compare_against_goals(totals, [])
    assert "1,800" in result or "1800" in result
    assert "No nutrition goals" in result or "no goals" in result.lower()

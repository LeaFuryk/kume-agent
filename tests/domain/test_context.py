from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kume.domain.context import ContextBuilder, ContextDataProvider

USER_ID = "user-42"
QUERY = "What should I eat for breakfast?"


def _make_goal(description: str) -> SimpleNamespace:
    return SimpleNamespace(description=description)


def _make_restriction(type_: str, description: str) -> SimpleNamespace:
    return SimpleNamespace(type=type_, description=description)


def _make_marker(name: str, value: str, unit: str, reference_range: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, value=value, unit=unit, reference_range=reference_range)


def _make_meal(
    description: str,
    calories: int,
    protein_g: float,
    carbs_g: float,
    fat_g: float,
) -> SimpleNamespace:
    from datetime import datetime

    return SimpleNamespace(
        description=description,
        calories=calories,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        logged_at=datetime(2026, 4, 13, 12, 30, 0),
    )


def _make_provider(
    *,
    goals: list | None = None,
    restrictions: list | None = None,
    markers: list | None = None,
    docs: list[str] | None = None,
    meals: list | None = None,
) -> ContextDataProvider:
    provider = AsyncMock(spec=ContextDataProvider)
    provider.get_goals = AsyncMock(return_value=goals or [])
    provider.get_restrictions = AsyncMock(return_value=restrictions or [])
    provider.get_lab_markers = AsyncMock(return_value=markers or [])
    provider.search_documents = AsyncMock(return_value=docs or [])
    provider.get_recent_meals = AsyncMock(return_value=meals or [])
    return provider


@pytest.mark.asyncio
async def test_all_sections_populated_and_order() -> None:
    goals = [_make_goal("Lose 5kg"), _make_goal("Improve energy")]
    restrictions = [_make_restriction("allergy", "Peanuts")]
    docs = ["Doc snippet about fiber.", "Doc snippet about protein."]
    markers = [_make_marker("Vitamin D", "28", "ng/mL", "30-100")]
    meals = [_make_meal("Grilled salmon", 500, 40.0, 5.0, 25.0)]

    provider = _make_provider(goals=goals, restrictions=restrictions, docs=docs, markers=markers, meals=meals)
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    expected_headings = [
        "## User Goals",
        "## Dietary Restrictions",
        "## Relevant Health Documents",
        "## Recent Lab Results",
        "## Recent Meals",
        "## Current Question",
    ]
    positions = [result.index(h) for h in expected_headings]
    assert positions == sorted(positions), "Sections are not in the prescribed order"

    assert "- Lose 5kg" in result
    assert "- [allergy] Peanuts" in result
    assert "Doc snippet about fiber." in result
    assert "- Vitamin D: 28 ng/mL (ref: 30-100)" in result
    assert "- [2026-04-13 12:30] Grilled salmon (500 kcal, 40.0g protein, 5.0g carbs, 25.0g fat)" in result
    assert QUERY in result


@pytest.mark.asyncio
async def test_empty_goals_omitted() -> None:
    provider = _make_provider(restrictions=[_make_restriction("intolerance", "Lactose")])
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    assert "## User Goals" not in result
    assert "## Dietary Restrictions" in result


@pytest.mark.asyncio
async def test_empty_restrictions_omitted() -> None:
    provider = _make_provider(goals=[_make_goal("Build muscle")])
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    assert "## Dietary Restrictions" not in result
    assert "## User Goals" in result


@pytest.mark.asyncio
async def test_no_documents_omitted() -> None:
    provider = _make_provider(goals=[_make_goal("Sleep better")])
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    assert "## Relevant Health Documents" not in result


@pytest.mark.asyncio
async def test_no_lab_markers_omitted() -> None:
    provider = _make_provider(goals=[_make_goal("Reduce inflammation")])
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    assert "## Recent Lab Results" not in result


@pytest.mark.asyncio
async def test_everything_empty_only_question() -> None:
    provider = _make_provider()
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    assert result == f"## Current Question\n{QUERY}"


@pytest.mark.asyncio
async def test_provider_receives_correct_args() -> None:
    provider = _make_provider()
    builder = ContextBuilder(provider=provider)

    await builder.build(user_id=USER_ID, query=QUERY)

    provider.get_goals.assert_awaited_once_with(USER_ID)
    provider.get_restrictions.assert_awaited_once_with(USER_ID)
    provider.get_lab_markers.assert_awaited_once_with(USER_ID)
    provider.search_documents.assert_awaited_once_with(USER_ID, QUERY)
    provider.get_recent_meals.assert_awaited_once_with(USER_ID)


@pytest.mark.asyncio
async def test_meals_section_in_correct_position() -> None:
    markers = [_make_marker("Iron", "80", "ug/dL", "60-170")]
    meals = [_make_meal("Grilled chicken salad", 450, 35.0, 20.0, 18.0)]

    provider = _make_provider(markers=markers, meals=meals)
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    lab_pos = result.index("## Recent Lab Results")
    meals_pos = result.index("## Recent Meals")
    question_pos = result.index("## Current Question")
    assert lab_pos < meals_pos < question_pos, "Recent Meals must appear after Lab Results and before Current Question"


@pytest.mark.asyncio
async def test_empty_meals_omitted() -> None:
    provider = _make_provider(goals=[_make_goal("Eat more fiber")])
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    assert "## Recent Meals" not in result


@pytest.mark.asyncio
async def test_meal_formatting_includes_description_and_macros() -> None:
    meals = [
        _make_meal("Oatmeal with berries", 320, 12.0, 55.0, 6.0),
        _make_meal("Protein shake", 200, 30.0, 10.0, 5.0),
    ]
    provider = _make_provider(meals=meals)
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    assert "- [2026-04-13 12:30] Oatmeal with berries (320 kcal, 12.0g protein, 55.0g carbs, 6.0g fat)" in result
    assert "- [2026-04-13 12:30] Protein shake (200 kcal, 30.0g protein, 10.0g carbs, 5.0g fat)" in result


def test_context_data_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        ContextDataProvider()  # type: ignore[abstract]

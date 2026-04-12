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


def _make_provider(
    *,
    goals: list | None = None,
    restrictions: list | None = None,
    markers: list | None = None,
    docs: list[str] | None = None,
) -> ContextDataProvider:
    provider = AsyncMock(spec=ContextDataProvider)
    provider.get_goals = AsyncMock(return_value=goals or [])
    provider.get_restrictions = AsyncMock(return_value=restrictions or [])
    provider.get_lab_markers = AsyncMock(return_value=markers or [])
    provider.search_documents = AsyncMock(return_value=docs or [])
    return provider


@pytest.mark.asyncio
async def test_all_sections_populated_and_order() -> None:
    goals = [_make_goal("Lose 5kg"), _make_goal("Improve energy")]
    restrictions = [_make_restriction("allergy", "Peanuts")]
    docs = ["Doc snippet about fiber.", "Doc snippet about protein."]
    markers = [_make_marker("Vitamin D", "28", "ng/mL", "30-100")]

    provider = _make_provider(goals=goals, restrictions=restrictions, docs=docs, markers=markers)
    builder = ContextBuilder(provider=provider)

    result = await builder.build(user_id=USER_ID, query=QUERY)

    expected_headings = [
        "## User Goals",
        "## Dietary Restrictions",
        "## Relevant Health Documents",
        "## Recent Lab Results",
        "## Current Question",
    ]
    positions = [result.index(h) for h in expected_headings]
    assert positions == sorted(positions), "Sections are not in the prescribed order"

    assert "- Lose 5kg" in result
    assert "- [allergy] Peanuts" in result
    assert "Doc snippet about fiber." in result
    assert "- Vitamin D: 28 ng/mL (ref: 30-100)" in result
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


def test_context_data_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        ContextDataProvider()  # type: ignore[abstract]

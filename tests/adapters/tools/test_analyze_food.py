from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kume.adapters.tools.analyze_food import AnalyzeFoodTool
from kume.domain.context import ContextBuilder
from tests.adapters.tools.conftest import FakeChatModel


class TestAnalyzeFoodTool:
    def _make_tool(
        self,
        response_text: str = "This food is healthy",
        context_builder: ContextBuilder | None = None,
    ) -> AnalyzeFoodTool:
        llm = FakeChatModel(response_text=response_text)
        return AnalyzeFoodTool(llm=llm, context_builder=context_builder)

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "analyze_food"
        assert "food" in tool.description.lower()

    def test_delegates_to_domain_handler(self) -> None:
        tool = self._make_tool("High in fiber")
        result = tool.invoke({"description": "A bowl of oatmeal with berries"})
        assert result == "High in fiber"

    def test_passes_description_through_to_llm(self) -> None:
        tool = self._make_tool("Nutritious meal")
        result = tool.invoke({"description": "Grilled salmon with rice"})
        assert isinstance(result, str)
        assert result == "Nutritious meal"

    def test_returns_llm_response_content(self) -> None:
        tool = self._make_tool("Low calorie option")
        result = tool.invoke({"description": "Garden salad"})
        assert result == "Low calorie option"

    def test_works_without_context_builder(self) -> None:
        tool = self._make_tool("No context analysis")
        result = tool.invoke({"description": "A banana"})
        assert result == "No context analysis"

    def test_works_without_user_id_set(self) -> None:
        """Even with a context_builder, if user_id is not set, context should be empty."""
        mock_builder = AsyncMock(spec=ContextBuilder)
        tool = self._make_tool("Analysis without user", context_builder=mock_builder)
        result = tool.invoke({"description": "A banana"})
        assert result == "Analysis without user"
        mock_builder.build.assert_not_awaited()

    def test_set_user_id(self) -> None:
        tool = self._make_tool()
        tool.set_user_id("user-123")
        assert tool._current_user_id == "user-123"


class TestAnalyzeFoodToolAsync:
    @pytest.mark.asyncio
    async def test_arun_calls_context_builder(self) -> None:
        mock_builder = AsyncMock(spec=ContextBuilder)
        mock_builder.build.return_value = "## Dietary Restrictions\n- [allergy] Peanuts"

        llm = FakeChatModel(response_text="Personalized analysis")
        tool = AnalyzeFoodTool(llm=llm, context_builder=mock_builder)
        tool.set_user_id("user-42")

        result = await tool.ainvoke({"description": "pad thai with peanut sauce"})

        assert result == "Personalized analysis"
        mock_builder.build.assert_awaited_once_with("user-42", "pad thai with peanut sauce")

    @pytest.mark.asyncio
    async def test_arun_graceful_when_context_builder_fails(self) -> None:
        mock_builder = AsyncMock(spec=ContextBuilder)
        mock_builder.build.side_effect = RuntimeError("DB unavailable")

        llm = FakeChatModel(response_text="Fallback analysis")
        tool = AnalyzeFoodTool(llm=llm, context_builder=mock_builder)
        tool.set_user_id("user-42")

        result = await tool.ainvoke({"description": "A pizza"})

        assert result == "Fallback analysis"

    @pytest.mark.asyncio
    async def test_arun_without_context_builder(self) -> None:
        llm = FakeChatModel(response_text="Generic analysis")
        tool = AnalyzeFoodTool(llm=llm)

        result = await tool.ainvoke({"description": "Sushi roll"})

        assert result == "Generic analysis"

    @pytest.mark.asyncio
    async def test_arun_without_user_id(self) -> None:
        mock_builder = AsyncMock(spec=ContextBuilder)
        llm = FakeChatModel(response_text="No user analysis")
        tool = AnalyzeFoodTool(llm=llm, context_builder=mock_builder)

        result = await tool.ainvoke({"description": "Sushi roll"})

        assert result == "No user analysis"
        mock_builder.build.assert_not_awaited()

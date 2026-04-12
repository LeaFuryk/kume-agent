from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kume.adapters.tools.ask_recommendation import AskRecommendationTool
from kume.domain.context import ContextBuilder
from tests.adapters.tools.conftest import FakeChatModel


class TestAskRecommendationTool:
    def _make_tool(
        self,
        response_text: str = "Eat more vegetables",
        context_builder: ContextBuilder | None = None,
    ) -> AskRecommendationTool:
        llm = FakeChatModel(response_text=response_text)
        return AskRecommendationTool(llm=llm, context_builder=context_builder)

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "ask_recommendation"
        assert "nutrition" in tool.description.lower()

    def test_delegates_to_domain_handler(self) -> None:
        tool = self._make_tool("Eat more protein")
        result = tool.invoke({"query": "How much protein should I eat?"})
        assert result == "Eat more protein"

    def test_passes_query_through_to_llm(self) -> None:
        tool = self._make_tool("Great advice")
        result = tool.invoke({"query": "What should I eat?"})
        assert isinstance(result, str)
        assert result == "Great advice"

    def test_returns_llm_response_content(self) -> None:
        tool = self._make_tool("Balanced diet recommended")
        result = tool.invoke({"query": "Give me advice"})
        assert result == "Balanced diet recommended"

    def test_works_without_context_builder(self) -> None:
        tool = self._make_tool("No context advice")
        result = tool.invoke({"query": "What should I eat?"})
        assert result == "No context advice"

    def test_works_without_user_id_set(self) -> None:
        """Even with a context_builder, if user_id is not set, context should be empty."""
        mock_builder = AsyncMock(spec=ContextBuilder)
        tool = self._make_tool("Advice without user", context_builder=mock_builder)
        result = tool.invoke({"query": "What should I eat?"})
        assert result == "Advice without user"
        mock_builder.build.assert_not_awaited()

    def test_set_user_id(self) -> None:
        tool = self._make_tool()
        tool.set_user_id("user-123")
        assert tool._current_user_id == "user-123"


class TestAskRecommendationToolAsync:
    @pytest.mark.asyncio
    async def test_arun_calls_context_builder(self) -> None:
        mock_builder = AsyncMock(spec=ContextBuilder)
        mock_builder.build.return_value = "## User Goals\n- Lose weight"

        llm = FakeChatModel(response_text="Personalized recommendation")
        tool = AskRecommendationTool(llm=llm, context_builder=mock_builder)
        tool.set_user_id("user-42")

        result = await tool.ainvoke({"query": "What should I eat?"})

        assert result == "Personalized recommendation"
        mock_builder.build.assert_awaited_once_with("user-42", "What should I eat?")

    @pytest.mark.asyncio
    async def test_arun_graceful_when_context_builder_fails(self) -> None:
        mock_builder = AsyncMock(spec=ContextBuilder)
        mock_builder.build.side_effect = RuntimeError("DB unavailable")

        llm = FakeChatModel(response_text="Fallback advice")
        tool = AskRecommendationTool(llm=llm, context_builder=mock_builder)
        tool.set_user_id("user-42")

        result = await tool.ainvoke({"query": "What should I eat?"})

        assert result == "Fallback advice"

    @pytest.mark.asyncio
    async def test_arun_without_context_builder(self) -> None:
        llm = FakeChatModel(response_text="Generic advice")
        tool = AskRecommendationTool(llm=llm)

        result = await tool.ainvoke({"query": "Breakfast ideas?"})

        assert result == "Generic advice"

    @pytest.mark.asyncio
    async def test_arun_without_user_id(self) -> None:
        mock_builder = AsyncMock(spec=ContextBuilder)
        llm = FakeChatModel(response_text="No user advice")
        tool = AskRecommendationTool(llm=llm, context_builder=mock_builder)

        result = await tool.ainvoke({"query": "Breakfast ideas?"})

        assert result == "No user advice"
        mock_builder.build.assert_not_awaited()

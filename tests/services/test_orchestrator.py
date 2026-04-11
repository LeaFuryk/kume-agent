from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from kume.infrastructure.metrics import MetricsCallbackHandler, MetricsCollector
from kume.services.orchestrator import OrchestratorService


class FakeChatModel(BaseChatModel):
    """A minimal BaseChatModel implementation for testing."""

    response_text: str = "fake response"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response_text))])


class FakeTool(BaseTool):
    """A minimal tool for constructing the agent."""

    name: str = "fake_tool"
    description: str = "A fake tool for testing"

    def _run(self, query: str = "") -> str:
        return "fake tool result"


@pytest.fixture()
def fake_llm() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture()
def fake_tools() -> list[BaseTool]:
    return [FakeTool()]


@pytest.fixture()
def metrics_collector() -> MetricsCollector:
    return MetricsCollector()


@pytest.fixture()
def orchestrator(
    fake_llm: FakeChatModel, fake_tools: list[BaseTool], metrics_collector: MetricsCollector
) -> OrchestratorService:
    return OrchestratorService(
        llm=fake_llm,
        tools=fake_tools,
        metrics_collector=metrics_collector,
    )


async def test_process_returns_string_response(orchestrator: OrchestratorService) -> None:
    """process() returns a string when the agent runs successfully."""
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="Here is your nutrition advice.")]},
    ):
        result = await orchestrator.process(telegram_id=12345, text="What should I eat?")

    assert isinstance(result, str)
    assert result == "Here is your nutrition advice."


async def test_process_returns_fallback_on_exception(orchestrator: OrchestratorService) -> None:
    """process() returns the fallback error message when the agent raises an exception."""
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM connection failed"),
    ):
        result = await orchestrator.process(telegram_id=12345, text="Hello")

    assert result == "Sorry, something went wrong. Please try again."


async def test_process_collects_metrics(orchestrator: OrchestratorService, metrics_collector: MetricsCollector) -> None:
    """process() calls start_request and end_request on the metrics collector."""
    with (
        patch.object(metrics_collector, "start_request", wraps=metrics_collector.start_request) as mock_start,
        patch.object(metrics_collector, "end_request", wraps=metrics_collector.end_request) as mock_end,
        patch.object(
            orchestrator._agent,
            "ainvoke",
            new_callable=AsyncMock,
            return_value={"messages": [AIMessage(content="test")]},
        ),
    ):
        await orchestrator.process(telegram_id=99, text="hi")

    mock_start.assert_called_once_with(99)
    mock_end.assert_called_once()


async def test_process_passes_callback_handler(orchestrator: OrchestratorService) -> None:
    """process() passes a MetricsCallbackHandler to the agent's ainvoke call."""
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="ok")]},
    ) as mock_ainvoke:
        await orchestrator.process(telegram_id=1, text="test")

    mock_ainvoke.assert_called_once()
    call_kwargs = mock_ainvoke.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    callbacks = config["callbacks"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], MetricsCallbackHandler)


async def test_process_returns_default_when_messages_empty(orchestrator: OrchestratorService) -> None:
    """process() returns the fallback text when agent result has no messages."""
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": []},
    ):
        result = await orchestrator.process(telegram_id=1, text="test")

    assert result == "I wasn't able to process that request."


async def test_process_returns_default_when_messages_key_missing(orchestrator: OrchestratorService) -> None:
    """process() returns the fallback text when agent result has no 'messages' key."""
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={},
    ):
        result = await orchestrator.process(telegram_id=1, text="test")

    assert result == "I wasn't able to process that request."


async def test_metrics_collected_even_on_exception(
    orchestrator: OrchestratorService, metrics_collector: MetricsCollector
) -> None:
    """end_request is called even when the agent raises an exception (finally block)."""
    with (
        patch.object(metrics_collector, "end_request", wraps=metrics_collector.end_request) as mock_end,
        patch.object(
            orchestrator._agent,
            "ainvoke",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ),
    ):
        await orchestrator.process(telegram_id=42, text="fail")

    mock_end.assert_called_once()

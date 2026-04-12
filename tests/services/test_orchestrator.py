from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from kume.infrastructure.metrics import MetricsCallbackHandler
from kume.infrastructure.request_context import get_context
from kume.services.orchestrator import OrchestratorService, _extract_text_content
from tests.adapters.tools.conftest import FakeUserRepository


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
def orchestrator(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> OrchestratorService:
    return OrchestratorService(
        llm=fake_llm,
        tools=fake_tools,
    )


async def test_process_returns_string_response(orchestrator: OrchestratorService) -> None:
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
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM connection failed"),
    ):
        result = await orchestrator.process(telegram_id=12345, text="Hello")

    assert result == "Sorry, something went wrong. Please try again."


async def test_process_passes_callback_handler(orchestrator: OrchestratorService) -> None:
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
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": []},
    ):
        result = await orchestrator.process(telegram_id=1, text="test")

    assert result == "I wasn't able to process that request."


async def test_process_returns_default_when_messages_key_missing(orchestrator: OrchestratorService) -> None:
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={},
    ):
        result = await orchestrator.process(telegram_id=1, text="test")

    assert result == "I wasn't able to process that request."


async def test_process_handles_structured_content_blocks(orchestrator: OrchestratorService) -> None:
    """process() extracts text from structured content blocks instead of leaking repr."""
    structured_content = [{"type": "text", "text": "Hello from structured block"}]
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content=structured_content)]},
    ):
        result = await orchestrator.process(telegram_id=1, text="test")

    assert result == "Hello from structured block"
    assert "[{" not in result


async def test_process_sets_request_context_via_contextvar(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> None:
    """process() sets RequestContext contextvar when user_repo is provided."""
    user_repo = FakeUserRepository()
    orch = OrchestratorService(llm=fake_llm, tools=fake_tools, user_repo=user_repo)

    with patch.object(
        orch._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="ok")]},
    ):
        await orch.process(telegram_id=99, text="hi")

    ctx = get_context()
    assert ctx is not None
    assert ctx.user_id == "fake-user"
    assert ctx.telegram_id == 99
    assert ctx.language == "en"


# --- _extract_text_content tests ---


def test_extract_text_content_string() -> None:
    assert _extract_text_content("hello") == "hello"


def test_extract_text_content_structured_blocks() -> None:
    blocks = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
    assert _extract_text_content(blocks) == "part1part2"


def test_extract_text_content_none() -> None:
    assert _extract_text_content(None) == ""


def test_extract_text_content_mixed_list() -> None:
    blocks = ["plain", {"type": "text", "text": "structured"}]
    assert _extract_text_content(blocks) == "plainstructured"

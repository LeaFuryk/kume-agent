from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from kume.domain.conversation import ConversationEvent
from kume.infrastructure.image_store import ImageStore
from kume.infrastructure.metrics import MetricsCallbackHandler, ReasoningCallbackHandler
from kume.infrastructure.request_context import get_context
from kume.infrastructure.session_store import SessionStore
from kume.services.orchestrator import OrchestratorService, Resource, _extract_text_content
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
        result = await orchestrator.process(telegram_id=12345, user_message="What should I eat?")

    assert isinstance(result, str)
    assert result == "Here is your nutrition advice."


async def test_process_returns_fallback_on_exception(orchestrator: OrchestratorService) -> None:
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM connection failed"),
    ):
        result = await orchestrator.process(telegram_id=12345, user_message="Hello")

    assert result == "Sorry, something went wrong. Please try again."


async def test_process_passes_callback_handler(orchestrator: OrchestratorService) -> None:
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="ok")]},
    ) as mock_ainvoke:
        await orchestrator.process(telegram_id=1, user_message="test")

    mock_ainvoke.assert_called_once()
    call_kwargs = mock_ainvoke.call_args
    config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
    callbacks = config["callbacks"]
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], MetricsCallbackHandler)
    assert isinstance(callbacks[1], ReasoningCallbackHandler)


async def test_process_returns_default_when_messages_empty(orchestrator: OrchestratorService) -> None:
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": []},
    ):
        result = await orchestrator.process(telegram_id=1, user_message="test")

    assert result == "I wasn't able to process that request."


async def test_process_returns_default_when_messages_key_missing(orchestrator: OrchestratorService) -> None:
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={},
    ):
        result = await orchestrator.process(telegram_id=1, user_message="test")

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
        result = await orchestrator.process(telegram_id=1, user_message="test")

    assert result == "Hello from structured block"
    assert "[{" not in result


async def test_process_sets_request_context_via_contextvar(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> None:
    """process() sets RequestContext contextvar during agent invocation."""
    user_repo = FakeUserRepository()
    orch = OrchestratorService(llm=fake_llm, tools=fake_tools, user_repo=user_repo)

    captured_ctx = None

    async def capture_context(*args: Any, **kwargs: Any) -> dict:
        nonlocal captured_ctx
        captured_ctx = get_context()
        return {"messages": [AIMessage(content="ok")]}

    with patch.object(orch._agent, "ainvoke", new_callable=AsyncMock, side_effect=capture_context):
        await orch.process(telegram_id=99, user_message="hi")

    # Context is set during the agent call (cleared in finally)
    assert captured_ctx is not None
    assert captured_ctx.user_id == "fake-user"
    assert captured_ctx.telegram_id == 99
    assert captured_ctx.language == "en"
    # After process() returns, context should be cleared
    assert get_context() is None


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


# --- Session & Image store tests ---


async def test_session_history_loaded(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> None:
    """Session events are converted to HumanMessage/AIMessage and prepended to agent input."""
    user_repo = FakeUserRepository()
    session_store = SessionStore()
    now = datetime.now(UTC)

    # Pre-populate session with one exchange
    session_store.add(
        "fake-user",
        ConversationEvent(id="e1", user_id="fake-user", role="user", content="previous question", created_at=now),
    )
    session_store.add(
        "fake-user",
        ConversationEvent(id="e2", user_id="fake-user", role="assistant", content="previous answer", created_at=now),
    )

    orch = OrchestratorService(
        llm=fake_llm,
        tools=fake_tools,
        user_repo=user_repo,
        session_store=session_store,
    )

    with patch.object(
        orch._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="current response")]},
    ) as mock_ainvoke:
        await orch.process(telegram_id=99, user_message="current question")

    # Verify the messages passed to agent include history
    call_args = mock_ainvoke.call_args
    passed_messages = call_args[0][0]["messages"] if call_args[0] else call_args.kwargs["messages"]
    assert len(passed_messages) == 3
    assert isinstance(passed_messages[0], HumanMessage)
    assert passed_messages[0].content == "previous question"
    assert isinstance(passed_messages[1], AIMessage)
    assert passed_messages[1].content == "previous answer"
    assert isinstance(passed_messages[2], HumanMessage)


async def test_events_saved_after_response(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> None:
    """SessionStore.add is called with user + assistant events after a successful response."""
    user_repo = FakeUserRepository()
    session_store = SessionStore()

    orch = OrchestratorService(
        llm=fake_llm,
        tools=fake_tools,
        user_repo=user_repo,
        session_store=session_store,
    )

    with patch.object(
        orch._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="bot reply")]},
    ):
        await orch.process(telegram_id=99, user_message="hello")

    # Session should now contain 2 events (user + assistant)
    events = session_store.get_session("fake-user")
    assert len(events) == 2
    assert events[0].role == "user"
    assert "hello" in events[0].content
    assert events[1].role == "assistant"
    assert events[1].content == "bot reply"


async def test_images_set_and_cleared(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> None:
    """ImageStore.set_images is called with image bytes and clear is called after."""
    user_repo = FakeUserRepository()
    image_store = ImageStore()

    orch = OrchestratorService(
        llm=fake_llm,
        tools=fake_tools,
        user_repo=user_repo,
        image_store=image_store,
    )

    resources = [
        Resource(mime_type="image/jpeg", transcript="a photo of food", raw_bytes=b"jpeg-bytes"),
        Resource(mime_type="application/pdf", transcript="a pdf doc", raw_bytes=b"pdf-bytes"),
    ]

    with patch.object(
        orch._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="analyzed")]},
    ) as mock_ainvoke:
        # Intercept to check images are set during invocation
        async def check_images_set(*args: Any, **kwargs: Any) -> dict:
            # Images should be stored at this point (before clear)
            assert image_store._data  # at least one request_id has images
            return {"messages": [AIMessage(content="analyzed")]}

        mock_ainvoke.side_effect = check_images_set
        result = await orch.process(telegram_id=99, user_message="analyze", resources=resources)

    assert result == "analyzed"
    # After process() returns, images should be cleared
    assert len(image_store._data) == 0


async def test_images_cleared_on_exception(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> None:
    """ImageStore.clear is called even when the agent raises an exception."""
    image_store = ImageStore()

    orch = OrchestratorService(
        llm=fake_llm,
        tools=fake_tools,
        image_store=image_store,
    )

    resources = [
        Resource(mime_type="image/png", transcript="a photo", raw_bytes=b"png-bytes"),
    ]

    with patch.object(
        orch._agent,
        "ainvoke",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = await orch.process(telegram_id=1, user_message="test", resources=resources)

    assert result == "Sorry, something went wrong. Please try again."
    # Images should still be cleared in the finally block
    assert len(image_store._data) == 0


async def test_backward_compat_no_stores(orchestrator: OrchestratorService) -> None:
    """Existing behavior is unchanged when session_store and image_store are None."""
    with patch.object(
        orchestrator._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="works fine")]},
    ):
        result = await orchestrator.process(telegram_id=12345, user_message="hi there")

    assert result == "works fine"


# --- Language instruction tests ---


async def test_language_instruction_included_when_language_provided(
    fake_llm: FakeChatModel, fake_tools: list[BaseTool]
) -> None:
    """When language is provided, a '[Respond in: ...]' instruction appears in the message."""
    user_repo = FakeUserRepository()
    orch = OrchestratorService(llm=fake_llm, tools=fake_tools, user_repo=user_repo)

    with patch.object(
        orch._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="Hola")]},
    ) as mock_ainvoke:
        await orch.process(telegram_id=99, user_message="hi", language="es")

    call_args = mock_ainvoke.call_args
    passed_messages = call_args[0][0]["messages"] if call_args[0] else call_args.kwargs["messages"]
    human_msg = passed_messages[-1]
    assert isinstance(human_msg, HumanMessage)
    assert "[Respond in: Spanish]" in human_msg.content


async def test_no_language_instruction_when_language_is_none(
    fake_llm: FakeChatModel, fake_tools: list[BaseTool]
) -> None:
    """When language is None, no '[Respond in: ...]' instruction appears in the message."""
    user_repo = FakeUserRepository()
    orch = OrchestratorService(llm=fake_llm, tools=fake_tools, user_repo=user_repo)

    with patch.object(
        orch._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="Hello")]},
    ) as mock_ainvoke:
        await orch.process(telegram_id=99, user_message="hi", language=None)

    call_args = mock_ainvoke.call_args
    passed_messages = call_args[0][0]["messages"] if call_args[0] else call_args.kwargs["messages"]
    human_msg = passed_messages[-1]
    assert isinstance(human_msg, HumanMessage)
    assert "[Respond in:" not in human_msg.content


async def test_language_sets_request_context(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> None:
    """When language is provided, RequestContext.language reflects it."""
    user_repo = FakeUserRepository()
    orch = OrchestratorService(llm=fake_llm, tools=fake_tools, user_repo=user_repo)

    captured_ctx = None

    async def capture_context(*args: Any, **kwargs: Any) -> dict:
        nonlocal captured_ctx
        captured_ctx = get_context()
        return {"messages": [AIMessage(content="ok")]}

    with patch.object(orch._agent, "ainvoke", new_callable=AsyncMock, side_effect=capture_context):
        await orch.process(telegram_id=99, user_message="hi", language="pt")

    assert captured_ctx is not None
    assert captured_ctx.language == "pt"


async def test_language_defaults_to_en_when_none(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> None:
    """When language is None, RequestContext.language defaults to 'en'."""
    user_repo = FakeUserRepository()
    orch = OrchestratorService(llm=fake_llm, tools=fake_tools, user_repo=user_repo)

    captured_ctx = None

    async def capture_context(*args: Any, **kwargs: Any) -> dict:
        nonlocal captured_ctx
        captured_ctx = get_context()
        return {"messages": [AIMessage(content="ok")]}

    with patch.object(orch._agent, "ainvoke", new_callable=AsyncMock, side_effect=capture_context):
        await orch.process(telegram_id=99, user_message="hi", language=None)

    assert captured_ctx is not None
    assert captured_ctx.language == "en"


async def test_unknown_language_code_used_as_is(fake_llm: FakeChatModel, fake_tools: list[BaseTool]) -> None:
    """When language code is not in the mapping, the raw code is used."""
    user_repo = FakeUserRepository()
    orch = OrchestratorService(llm=fake_llm, tools=fake_tools, user_repo=user_repo)

    with patch.object(
        orch._agent,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={"messages": [AIMessage(content="ok")]},
    ) as mock_ainvoke:
        await orch.process(telegram_id=99, user_message="hi", language="ja")

    call_args = mock_ainvoke.call_args
    passed_messages = call_args[0][0]["messages"] if call_args[0] else call_args.kwargs["messages"]
    human_msg = passed_messages[-1]
    assert "[Respond in: ja]" in human_msg.content

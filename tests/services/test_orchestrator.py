from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from kume.domain.conversation import ConversationEvent
from kume.infrastructure.image_store import ImageStore
from kume.infrastructure.request_context import get_context
from kume.infrastructure.session_store import SessionStore
from kume.ports.output.messaging import MessagingPort
from kume.services.orchestrator import OrchestratorService, ProcessResult, Resource, _extract_text_content
from tests.adapters.tools.conftest import FakeUserRepository


def _make_mock_graph(formatted_response: str = "fake response") -> AsyncMock:
    """Create a mock graph whose ainvoke returns a dict with formatted_response."""
    mock = AsyncMock()
    mock.ainvoke.return_value = {"formatted_response": formatted_response}
    return mock


@pytest.fixture()
def mock_graph() -> AsyncMock:
    return _make_mock_graph()


@pytest.fixture()
def orchestrator(mock_graph: AsyncMock) -> OrchestratorService:
    return OrchestratorService(graph=mock_graph)


async def test_process_returns_process_result(orchestrator: OrchestratorService) -> None:
    result = await orchestrator.process(telegram_id=12345, user_message="What should I eat?")

    assert isinstance(result, ProcessResult)
    assert result.text == "fake response"
    assert result.streamed is False


async def test_process_returns_fallback_on_exception(orchestrator: OrchestratorService) -> None:
    orchestrator._graph.ainvoke.side_effect = RuntimeError("LLM connection failed")

    result = await orchestrator.process(telegram_id=12345, user_message="Hello")

    assert result.text == "Sorry, something went wrong. Please try again."
    assert result.streamed is False


async def test_process_returns_default_when_formatted_response_empty(mock_graph: AsyncMock) -> None:
    mock_graph.ainvoke.return_value = {"formatted_response": ""}
    orch = OrchestratorService(graph=mock_graph)

    result = await orch.process(telegram_id=1, user_message="test")

    assert result.text == "I wasn't able to process that request."
    assert result.streamed is False


async def test_process_returns_default_when_formatted_response_missing(mock_graph: AsyncMock) -> None:
    mock_graph.ainvoke.return_value = {}
    orch = OrchestratorService(graph=mock_graph)

    result = await orch.process(telegram_id=1, user_message="test")

    assert result.text == "I wasn't able to process that request."
    assert result.streamed is False


async def test_process_handles_whitespace_only_response(mock_graph: AsyncMock) -> None:
    """process() treats whitespace-only formatted_response as empty."""
    mock_graph.ainvoke.return_value = {"formatted_response": "   "}
    orch = OrchestratorService(graph=mock_graph)

    result = await orch.process(telegram_id=1, user_message="test")

    assert result.text == "I wasn't able to process that request."
    assert result.streamed is False


async def test_process_sets_request_context_via_contextvar() -> None:
    """process() sets RequestContext contextvar during graph invocation."""
    user_repo = FakeUserRepository()
    mock_graph = _make_mock_graph("ok")

    captured_ctx = None

    async def capture_context(state: dict[str, Any]) -> dict:
        nonlocal captured_ctx
        captured_ctx = get_context()
        return {"formatted_response": "ok"}

    mock_graph.ainvoke.side_effect = capture_context
    orch = OrchestratorService(graph=mock_graph, user_repo=user_repo)

    await orch.process(telegram_id=99, user_message="hi")

    # Context is set during the graph call (cleared in finally)
    assert captured_ctx is not None
    assert captured_ctx.user_id == "fake-user"
    assert captured_ctx.telegram_id == 99
    assert captured_ctx.language == "en"
    # After process() returns, context should be cleared
    assert get_context() is None


async def test_graph_receives_state_fields() -> None:
    """Verify the graph is invoked with all expected state fields."""
    mock_graph = _make_mock_graph("response")
    orch = OrchestratorService(graph=mock_graph)

    await orch.process(telegram_id=1, user_message="test", language="es")

    mock_graph.ainvoke.assert_called_once()
    state = mock_graph.ainvoke.call_args[0][0]
    assert state["user_language"] == "es"
    assert state["input_safe"] is True
    assert state["output_safe"] is True
    assert state["guardrail_violation"] is None
    assert state["formatted_response"] == ""
    assert state["memory_summarized"] is False
    assert state["tool_error_count"] == 0


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


async def test_session_history_loaded() -> None:
    """Session events are converted to HumanMessage/AIMessage and prepended to graph input."""
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

    mock_graph = _make_mock_graph("current response")
    orch = OrchestratorService(
        graph=mock_graph,
        user_repo=user_repo,
        session_store=session_store,
    )

    await orch.process(telegram_id=99, user_message="current question")

    # Verify the messages passed to graph include history
    call_args = mock_graph.ainvoke.call_args
    passed_messages = call_args[0][0]["messages"]
    assert len(passed_messages) == 3
    assert isinstance(passed_messages[0], HumanMessage)
    assert passed_messages[0].content == "previous question"
    assert isinstance(passed_messages[1], AIMessage)
    assert passed_messages[1].content == "previous answer"
    assert isinstance(passed_messages[2], HumanMessage)


async def test_events_saved_after_response() -> None:
    """SessionStore.add is called with user + assistant events after a successful response."""
    user_repo = FakeUserRepository()
    session_store = SessionStore()
    mock_graph = _make_mock_graph("bot reply")

    orch = OrchestratorService(
        graph=mock_graph,
        user_repo=user_repo,
        session_store=session_store,
    )

    await orch.process(telegram_id=99, user_message="hello")

    # Session should now contain 2 events (user + assistant)
    events = session_store.get_session("fake-user")
    assert len(events) == 2
    assert events[0].role == "user"
    assert "hello" in events[0].content
    assert events[1].role == "assistant"
    assert events[1].content == "bot reply"


async def test_images_set_and_cleared() -> None:
    """ImageStore.set_images is called with image bytes and clear is called after."""
    user_repo = FakeUserRepository()
    image_store = ImageStore()
    mock_graph = _make_mock_graph("analyzed")

    orch = OrchestratorService(
        graph=mock_graph,
        user_repo=user_repo,
        image_store=image_store,
    )

    resources = [
        Resource(mime_type="image/jpeg", transcript="a photo of food", raw_bytes=b"jpeg-bytes"),
        Resource(mime_type="application/pdf", transcript="a pdf doc", raw_bytes=b"pdf-bytes"),
    ]

    # Intercept to check images are set during invocation
    async def check_images_set(state: dict[str, Any]) -> dict:
        # Images should be stored at this point (before clear)
        assert image_store._data  # at least one request_id has images
        return {"formatted_response": "analyzed"}

    mock_graph.ainvoke.side_effect = check_images_set
    result = await orch.process(telegram_id=99, user_message="analyze", resources=resources)

    assert result.text == "analyzed"
    # After process() returns, images should be cleared
    assert len(image_store._data) == 0


async def test_images_cleared_on_exception() -> None:
    """ImageStore.clear is called even when the graph raises an exception."""
    image_store = ImageStore()
    mock_graph = _make_mock_graph()
    mock_graph.ainvoke.side_effect = RuntimeError("boom")

    orch = OrchestratorService(
        graph=mock_graph,
        image_store=image_store,
    )

    resources = [
        Resource(mime_type="image/png", transcript="a photo", raw_bytes=b"png-bytes"),
    ]

    result = await orch.process(telegram_id=1, user_message="test", resources=resources)

    assert result.text == "Sorry, something went wrong. Please try again."
    # Images should still be cleared in the finally block
    assert len(image_store._data) == 0


async def test_backward_compat_no_stores(orchestrator: OrchestratorService) -> None:
    """Existing behavior is unchanged when session_store and image_store are None."""
    result = await orchestrator.process(telegram_id=12345, user_message="hi there")

    assert result.text == "fake response"


# --- Language instruction tests ---


async def test_language_instruction_included_when_language_provided() -> None:
    """When language is provided, a '[Respond in: ...]' instruction appears in the message."""
    user_repo = FakeUserRepository()
    mock_graph = _make_mock_graph("Hola")
    orch = OrchestratorService(graph=mock_graph, user_repo=user_repo)

    await orch.process(telegram_id=99, user_message="hi", language="es")

    call_args = mock_graph.ainvoke.call_args
    passed_messages = call_args[0][0]["messages"]
    human_msg = passed_messages[-1]
    assert isinstance(human_msg, HumanMessage)
    assert "[Respond in: Spanish]" in human_msg.content


async def test_no_language_instruction_when_language_is_none() -> None:
    """When language is None, no '[Respond in: ...]' instruction appears in the message."""
    user_repo = FakeUserRepository()
    mock_graph = _make_mock_graph("Hello")
    orch = OrchestratorService(graph=mock_graph, user_repo=user_repo)

    await orch.process(telegram_id=99, user_message="hi", language=None)

    call_args = mock_graph.ainvoke.call_args
    passed_messages = call_args[0][0]["messages"]
    human_msg = passed_messages[-1]
    assert isinstance(human_msg, HumanMessage)
    assert "[Respond in:" not in human_msg.content


async def test_language_sets_request_context() -> None:
    """When language is provided, RequestContext.language reflects it."""
    user_repo = FakeUserRepository()
    mock_graph = _make_mock_graph("ok")

    captured_ctx = None

    async def capture_context(state: dict[str, Any]) -> dict:
        nonlocal captured_ctx
        captured_ctx = get_context()
        return {"formatted_response": "ok"}

    mock_graph.ainvoke.side_effect = capture_context
    orch = OrchestratorService(graph=mock_graph, user_repo=user_repo)

    await orch.process(telegram_id=99, user_message="hi", language="pt")

    assert captured_ctx is not None
    assert captured_ctx.language == "pt"


async def test_language_defaults_to_en_when_none() -> None:
    """When language is None, RequestContext.language defaults to 'en'."""
    user_repo = FakeUserRepository()
    mock_graph = _make_mock_graph("ok")

    captured_ctx = None

    async def capture_context(state: dict[str, Any]) -> dict:
        nonlocal captured_ctx
        captured_ctx = get_context()
        return {"formatted_response": "ok"}

    mock_graph.ainvoke.side_effect = capture_context
    orch = OrchestratorService(graph=mock_graph, user_repo=user_repo)

    await orch.process(telegram_id=99, user_message="hi", language=None)

    assert captured_ctx is not None
    assert captured_ctx.language == "en"


async def test_unknown_language_code_used_as_is() -> None:
    """When language code is not in the mapping, the raw code is used."""
    user_repo = FakeUserRepository()
    mock_graph = _make_mock_graph("ok")
    orch = OrchestratorService(graph=mock_graph, user_repo=user_repo)

    await orch.process(telegram_id=99, user_message="hi", language="ja")

    call_args = mock_graph.ainvoke.call_args
    passed_messages = call_args[0][0]["messages"]
    human_msg = passed_messages[-1]
    assert "[Respond in: ja]" in human_msg.content


# --- Streaming integration tests ---


@pytest.fixture()
def mock_messaging() -> AsyncMock:
    messaging = AsyncMock(spec=MessagingPort)
    messaging.send_and_get_id.return_value = 42
    return messaging


async def test_streaming_sends_placeholder(orchestrator: OrchestratorService, mock_messaging: AsyncMock) -> None:
    """Verify send_and_get_id is called when messaging+chat_id are provided."""
    result = await orchestrator.process(telegram_id=1, user_message="test", messaging=mock_messaging, chat_id=99)

    mock_messaging.send_and_get_id.assert_awaited_once_with(99, "...")
    assert result.text == "fake response"


async def test_streaming_result_streamed_true(orchestrator: OrchestratorService, mock_messaging: AsyncMock) -> None:
    """Verify result.streamed is True when streaming setup succeeds."""
    result = await orchestrator.process(telegram_id=1, user_message="test", messaging=mock_messaging, chat_id=99)

    assert result.streamed is True


async def test_placeholder_edited_with_response(orchestrator: OrchestratorService, mock_messaging: AsyncMock) -> None:
    """Verify placeholder message is edited with the final response."""
    result = await orchestrator.process(telegram_id=1, user_message="test", messaging=mock_messaging, chat_id=99)

    # Placeholder sent, then edited with final response
    mock_messaging.send_and_get_id.assert_awaited_once_with(99, "...")
    mock_messaging.edit_message.assert_awaited_once_with(99, 42, "fake response")
    assert result.streamed is True


async def test_streaming_fallback_on_setup_failure(
    orchestrator: OrchestratorService, mock_messaging: AsyncMock
) -> None:
    """When send_and_get_id raises, streaming is skipped and result.streamed is False."""
    mock_messaging.send_and_get_id.side_effect = RuntimeError("Telegram API down")

    result = await orchestrator.process(telegram_id=1, user_message="test", messaging=mock_messaging, chat_id=99)

    assert result.text == "fake response"
    assert result.streamed is False


async def test_no_streaming_when_no_messaging(orchestrator: OrchestratorService) -> None:
    """Verify no streaming when messaging is None."""
    result = await orchestrator.process(telegram_id=1, user_message="test")

    assert result.text == "fake response"
    assert result.streamed is False


async def test_no_streaming_when_no_chat_id(orchestrator: OrchestratorService, mock_messaging: AsyncMock) -> None:
    """Verify no streaming when chat_id is None even if messaging is provided."""
    result = await orchestrator.process(telegram_id=1, user_message="test", messaging=mock_messaging, chat_id=None)

    mock_messaging.send_and_get_id.assert_not_awaited()
    assert result.streamed is False

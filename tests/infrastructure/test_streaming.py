from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from kume.infrastructure.streaming import StreamingCallbackHandler
from kume.ports.output.messaging import MessagingPort


def _make_handler(messaging: AsyncMock | None = None) -> tuple[AsyncMock, StreamingCallbackHandler]:
    """Create a StreamingCallbackHandler with a mocked MessagingPort."""
    if messaging is None:
        messaging = AsyncMock(spec=MessagingPort)
    handler = StreamingCallbackHandler(messaging=messaging, chat_id=123, message_id=456)
    return messaging, handler


class TestTokenAccumulation:
    @pytest.mark.asyncio
    async def test_token_accumulation(self) -> None:
        """Send 5 tokens, verify full_text accumulates correctly."""
        messaging, handler = _make_handler()
        # Set last_flush_time far in the future to prevent time-based flushing
        handler._last_flush_time = time.monotonic() + 9999

        tokens = ["Hello", " ", "world", "!", " Test"]
        for token in tokens:
            await handler.on_llm_new_token(token)

        assert handler.full_text == "Hello world! Test"


class TestThrottling:
    @pytest.mark.asyncio
    async def test_throttling(self) -> None:
        """Send tokens rapidly, verify edit_message NOT called for every token."""
        messaging, handler = _make_handler()
        # Set flush time to now so we start with a fresh interval
        handler._last_flush_time = time.monotonic()

        # Send many small tokens rapidly (well under FLUSH_CHAR_THRESHOLD each)
        for _i in range(20):
            await handler.on_llm_new_token("a")

        # edit_message should NOT have been called 20 times — throttling should
        # collapse some updates. It may be called 0 times if threshold not hit.
        assert messaging.edit_message.call_count < 20


class TestFinalFlushNoCursor:
    @pytest.mark.asyncio
    async def test_final_flush_no_cursor(self) -> None:
        """Call on_llm_end, verify last edit_message text doesn't contain cursor."""
        messaging, handler = _make_handler()
        handler._buffer = "Final answer"
        handler._last_flush_text = ""

        await handler.on_llm_end(response=None)

        messaging.edit_message.assert_called_once()
        call_text = messaging.edit_message.call_args[0][2]
        assert "\u25cd" not in call_text
        assert call_text == "Final answer"


class TestToolStatusShown:
    @pytest.mark.asyncio
    async def test_tool_status_shown(self) -> None:
        """Call on_tool_start, verify edit includes tool name."""
        messaging, handler = _make_handler()
        handler._buffer = "Thinking"
        handler._last_flush_text = ""

        await handler.on_tool_start(serialized={"name": "search_food"}, input_str="apple")

        messaging.edit_message.assert_called_once()
        call_text = messaging.edit_message.call_args[0][2]
        assert "search_food" in call_text
        assert "\u23f3" in call_text


class TestToolStatusCleared:
    @pytest.mark.asyncio
    async def test_tool_status_cleared(self) -> None:
        """Call on_tool_end, verify tool status is gone from internal state."""
        _, handler = _make_handler()
        handler._tool_status = "\n\n\u23f3 Using search_food..."

        await handler.on_tool_end(output="result")

        assert handler._tool_status == ""


class TestFlushErrorCaught:
    @pytest.mark.asyncio
    async def test_flush_error_caught(self) -> None:
        """Mock edit_message to raise, verify no exception propagated."""
        messaging, handler = _make_handler()
        messaging.edit_message.side_effect = RuntimeError("Telegram API error")
        handler._buffer = "Some text"
        handler._last_flush_text = ""

        # Should not raise
        await handler.on_llm_end(response=None)


class TestFullTextProperty:
    @pytest.mark.asyncio
    async def test_full_text_property(self) -> None:
        """Verify full_text returns buffer without cursor or tool status."""
        _, handler = _make_handler()
        handler._buffer = "Hello world"
        handler._tool_status = "\n\n\u23f3 Using search..."

        assert handler.full_text == "Hello world"
        assert "\u23f3" not in handler.full_text
        assert "\u25cd" not in handler.full_text

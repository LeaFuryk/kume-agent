from __future__ import annotations

import asyncio
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


# ── Edge-case tests ──────────────────────────────────────────────────


class TestEmptyResponseNoFlush:
    @pytest.mark.asyncio
    async def test_empty_response_no_flush(self) -> None:
        """No tokens means no edit_message calls."""
        messaging, handler = _make_handler()
        await handler.on_llm_end(response=None)
        messaging.edit_message.assert_not_called()


class TestLongMessageOver4096Chars:
    @pytest.mark.asyncio
    async def test_long_message_over_4096_chars(self) -> None:
        """Messages over 4096 chars should still flush without error."""
        messaging, handler = _make_handler()
        long_text = "x" * 5000
        await handler.on_llm_new_token(long_text)
        await handler.on_llm_end(response=None)
        # Should have attempted to flush (adapter/Telegram handles truncation)
        assert messaging.edit_message.called
        # full_text should contain all content
        assert len(handler.full_text) == 5000


class TestRapidBurstThrottling:
    @pytest.mark.asyncio
    async def test_rapid_burst_throttling(self) -> None:
        """200 tokens in quick succession should result in far fewer edits."""
        messaging, handler = _make_handler()
        for _i in range(200):
            await handler.on_llm_new_token("a")
        await handler.on_llm_end(response=None)
        # Should be significantly fewer edits than 200
        # At minimum: some char-threshold flushes + final flush
        assert messaging.edit_message.call_count < 50
        assert handler.full_text == "a" * 200


class TestConcurrentFlushSafety:
    @pytest.mark.asyncio
    async def test_concurrent_flushes_dont_corrupt(self) -> None:
        """Multiple concurrent on_llm_new_token calls shouldn't corrupt the buffer."""
        messaging, handler = _make_handler()

        # Simulate concurrent token arrivals
        tokens = [f"token{i} " for i in range(20)]
        await asyncio.gather(*[handler.on_llm_new_token(t) for t in tokens])

        # All tokens should be in the buffer
        for t in tokens:
            assert t in handler.full_text


class TestToolInterleaving:
    @pytest.mark.asyncio
    async def test_tool_interleaving(self) -> None:
        """Tool start/end during streaming should show status then resume cleanly."""
        messaging, handler = _make_handler()

        await handler.on_llm_new_token("Starting analysis")
        await handler.on_tool_start({"name": "fetch_user_context"}, "query")
        # Should have flushed with tool status
        await handler.on_tool_end("context data")
        await handler.on_llm_new_token(". Here are the results.")
        await handler.on_llm_end(response=None)

        assert handler.full_text == "Starting analysis. Here are the results."
        # Tool status should NOT be in full_text
        assert "fetch_user_context" not in handler.full_text


class TestIdenticalContentSkipsEdit:
    @pytest.mark.asyncio
    async def test_identical_content_skips_edit(self) -> None:
        """If buffer hasn't changed since last flush, skip the edit."""
        messaging, handler = _make_handler()

        await handler.on_llm_new_token("hello")
        await handler._flush(cursor=False)
        call_count_after_first = messaging.edit_message.call_count

        # Flush again with same content
        await handler._flush(cursor=False)
        assert messaging.edit_message.call_count == call_count_after_first  # no new call


class TestErrorRecovery:
    @pytest.mark.asyncio
    async def test_continues_after_edit_failure(self) -> None:
        """If edit_message fails, subsequent tokens still accumulate."""
        messaging, handler = _make_handler()
        messaging.edit_message.side_effect = [Exception("network error"), None, None]

        await handler.on_llm_new_token("first ")
        await handler._flush(cursor=False)  # fails
        await handler.on_llm_new_token("second")
        await handler._flush(cursor=False)  # succeeds

        assert handler.full_text == "first second"

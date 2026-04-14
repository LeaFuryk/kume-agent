from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

from kume.ports.output.messaging import MessagingPort

logger = logging.getLogger("kume.streaming")

FLUSH_INTERVAL = 0.5  # seconds between edits
FLUSH_CHAR_THRESHOLD = 80  # chars since last flush triggers immediate edit


class StreamingCallbackHandler(AsyncCallbackHandler):
    """Streams LLM tokens to Telegram by editing a message in-place.

    Throttles edits to respect Telegram rate limits (~500ms between updates).
    Shows tool invocation status during agent execution.
    """

    def __init__(self, messaging: MessagingPort, chat_id: int, message_id: int) -> None:
        self._messaging = messaging
        self._chat_id = chat_id
        self._message_id = message_id
        self._buffer = ""
        self._last_flush_text = ""
        self._last_flush_time = 0.0
        self._lock = asyncio.Lock()
        self._tool_status = ""

    @property
    def full_text(self) -> str:
        """The complete accumulated text (without cursor or tool status)."""
        return self._buffer

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._buffer += token
        now = time.monotonic()
        chars_since_flush = len(self._buffer) - len(self._last_flush_text)
        time_since_flush = now - self._last_flush_time

        if chars_since_flush >= FLUSH_CHAR_THRESHOLD or time_since_flush >= FLUSH_INTERVAL:
            await self._flush(cursor=True)

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        # Final flush without cursor
        if self._buffer and self._buffer != self._last_flush_text:
            await self._flush(cursor=False)

    async def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "unknown")
        self._tool_status = f"\n\n\u23f3 Using {tool_name}..."
        await self._flush(cursor=False)

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._tool_status = ""

    async def _flush(self, *, cursor: bool = True) -> None:
        async with self._lock:
            display = self._buffer + self._tool_status
            if cursor and not self._tool_status:
                display += " \u25cd"
            if display == self._last_flush_text:
                return
            try:
                await self._messaging.edit_message(self._chat_id, self._message_id, display)
                self._last_flush_text = display
                self._last_flush_time = time.monotonic()
            except Exception:
                logger.debug("Failed to flush streaming update", exc_info=True)

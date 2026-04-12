from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("kume.batcher")

MAX_BATCH_BYTES = 50 * 1024 * 1024  # 50 MB total per user


@dataclass
class MediaItem:
    """A downloaded media file waiting to be processed in a batch."""

    raw_bytes: bytes
    mime_type: str
    caption: str


@dataclass
class BatchItem:
    """A single ordered item in a batch — either text or media."""

    type: str  # "text" or "media"
    text: str | None = None
    media: MediaItem | None = None


class PendingBatch:
    """Accumulates messages for one user within the debounce window."""

    def __init__(self) -> None:
        self.items: list[BatchItem] = []
        self.chat_id: int = 0
        self.language: str | None = None
        self.user_name: str | None = None
        self.total_bytes: int = 0


class MessageBatcher:
    """Per-user debounce queue.

    Collects text and media messages within a silence window, then fires
    the ``on_batch_ready`` callback with the accumulated batch.
    """

    def __init__(
        self,
        debounce_seconds: float,
        on_batch_ready: Callable[[int, PendingBatch], Coroutine[Any, Any, None]],
    ) -> None:
        self._debounce = debounce_seconds
        self._on_batch_ready = on_batch_ready
        self._batches: dict[int, PendingBatch] = {}
        self._timers: dict[int, asyncio.TimerHandle] = {}
        self._processing_locks: dict[int, asyncio.Lock] = {}

    async def add_text(
        self,
        telegram_id: int,
        chat_id: int,
        text: str,
        language: str | None = None,
        user_name: str | None = None,
    ) -> None:
        """Add a text message and reset the debounce timer."""
        batch = self._get_or_create_batch(telegram_id, chat_id, language)
        batch.items.append(BatchItem(type="text", text=text))
        if user_name:
            batch.user_name = user_name
        self._reset_timer(telegram_id)

    async def add_media(
        self,
        telegram_id: int,
        chat_id: int,
        item: MediaItem,
        language: str | None = None,
        user_name: str | None = None,
    ) -> None:
        """Add a downloaded media item and reset the debounce timer.

        Raises ValueError if adding this item would exceed the per-user
        memory cap (MAX_BATCH_BYTES).
        """
        batch = self._get_or_create_batch(telegram_id, chat_id, language)
        item_bytes = len(item.raw_bytes)
        if batch.total_bytes + item_bytes > MAX_BATCH_BYTES:
            raise ValueError(
                f"Batch memory cap exceeded: adding {item_bytes} bytes "
                f"would exceed {MAX_BATCH_BYTES} byte limit "
                f"(current: {batch.total_bytes} bytes)"
            )
        if user_name:
            batch.user_name = user_name
        batch.items.append(BatchItem(type="media", media=item))
        batch.total_bytes += item_bytes
        self._reset_timer(telegram_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_batch(
        self,
        telegram_id: int,
        chat_id: int,
        language: str | None,
    ) -> PendingBatch:
        if telegram_id not in self._batches:
            self._batches[telegram_id] = PendingBatch()
        batch = self._batches[telegram_id]
        batch.chat_id = chat_id
        if language is not None:
            batch.language = language
        return batch

    def _reset_timer(self, telegram_id: int) -> None:
        if telegram_id in self._timers:
            self._timers[telegram_id].cancel()
        loop = asyncio.get_running_loop()
        self._timers[telegram_id] = loop.call_later(
            self._debounce,
            lambda tid=telegram_id: asyncio.ensure_future(self._fire(tid)),
        )

    async def _fire(self, telegram_id: int) -> None:
        batch = self._batches.pop(telegram_id, None)
        self._timers.pop(telegram_id, None)
        if batch is None:
            return

        # Per-user lock: if a previous batch is still being processed,
        # this batch waits. Messages arriving during processing accumulate
        # in a new batch and fire after this one completes.
        if telegram_id not in self._processing_locks:
            self._processing_locks[telegram_id] = asyncio.Lock()

        async with self._processing_locks[telegram_id]:
            try:
                await self._on_batch_ready(telegram_id, batch)
            except Exception:
                logger.exception(
                    "Error in batch callback for telegram_id=%d",
                    telegram_id,
                )

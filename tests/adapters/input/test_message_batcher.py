import asyncio
from unittest.mock import AsyncMock

import pytest

from kume.adapters.input.message_batcher import MediaItem, MessageBatcher


@pytest.fixture
def on_batch_ready() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def batcher(on_batch_ready: AsyncMock) -> MessageBatcher:
    return MessageBatcher(debounce_seconds=0.1, on_batch_ready=on_batch_ready)


# ---- Timer behaviour ----


async def test_batch_fires_after_debounce_silence(
    batcher: MessageBatcher,
    on_batch_ready: AsyncMock,
) -> None:
    """A single text message fires the callback after the debounce period."""
    await batcher.add_text(111, 999, "hello", "en")
    # Should NOT have fired yet
    on_batch_ready.assert_not_awaited()

    await asyncio.sleep(0.2)

    on_batch_ready.assert_awaited_once()
    telegram_id, batch = on_batch_ready.call_args[0]
    assert telegram_id == 111
    assert len(batch.items) == 1
    assert batch.items[0].type == "text"
    assert batch.items[0].text == "hello"
    assert batch.chat_id == 999
    assert batch.language == "en"


async def test_timer_resets_on_new_message(
    batcher: MessageBatcher,
    on_batch_ready: AsyncMock,
) -> None:
    """Adding a second message before the debounce fires resets the timer."""
    await batcher.add_text(111, 999, "first")
    await asyncio.sleep(0.05)  # half the debounce
    await batcher.add_text(111, 999, "second")
    await asyncio.sleep(0.05)  # only 0.05s since last message — should not fire yet
    on_batch_ready.assert_not_awaited()

    await asyncio.sleep(0.1)  # now past 0.1s since last message
    on_batch_ready.assert_awaited_once()
    _, batch = on_batch_ready.call_args[0]
    assert [i.text for i in batch.items if i.type == "text"] == ["first", "second"]


# ---- Accumulation ----


async def test_multiple_texts_combine_in_one_batch(
    batcher: MessageBatcher,
    on_batch_ready: AsyncMock,
) -> None:
    """Multiple rapid texts all end up in the same batch."""
    await batcher.add_text(111, 999, "one")
    await batcher.add_text(111, 999, "two")
    await batcher.add_text(111, 999, "three")

    await asyncio.sleep(0.2)

    on_batch_ready.assert_awaited_once()
    _, batch = on_batch_ready.call_args[0]
    assert [i.text for i in batch.items if i.type == "text"] == ["one", "two", "three"]


async def test_media_items_accumulate_in_batch(
    batcher: MessageBatcher,
    on_batch_ready: AsyncMock,
) -> None:
    """Media items accumulate correctly in the batch."""
    item1 = MediaItem(raw_bytes=b"pdf1", mime_type="application/pdf", caption="Report 1")
    item2 = MediaItem(raw_bytes=b"pdf2", mime_type="application/pdf", caption="Report 2")

    await batcher.add_media(111, 999, item1)
    await batcher.add_media(111, 999, item2)

    await asyncio.sleep(0.2)

    on_batch_ready.assert_awaited_once()
    _, batch = on_batch_ready.call_args[0]
    media_items = [i.media for i in batch.items if i.type == "media"]
    assert len(media_items) == 2
    assert media_items[0].caption == "Report 1"
    assert media_items[1].caption == "Report 2"


async def test_text_and_media_combine_in_batch(
    batcher: MessageBatcher,
    on_batch_ready: AsyncMock,
) -> None:
    """Text and media sent together end up in the same batch."""
    await batcher.add_text(111, 999, "Here are my results")
    item = MediaItem(raw_bytes=b"pdf", mime_type="application/pdf", caption="")
    await batcher.add_media(111, 999, item)

    await asyncio.sleep(0.2)

    on_batch_ready.assert_awaited_once()
    _, batch = on_batch_ready.call_args[0]
    texts = [i.text for i in batch.items if i.type == "text"]
    media = [i.media for i in batch.items if i.type == "media"]
    assert texts == ["Here are my results"]
    assert len(media) == 1


# ---- User isolation ----


async def test_separate_users_get_separate_batches(
    batcher: MessageBatcher,
    on_batch_ready: AsyncMock,
) -> None:
    """Different users' messages are batched independently."""
    await batcher.add_text(111, 999, "user 1 msg")
    await batcher.add_text(222, 888, "user 2 msg")

    await asyncio.sleep(0.2)

    assert on_batch_ready.await_count == 2
    calls = on_batch_ready.call_args_list
    ids = {c[0][0] for c in calls}
    assert ids == {111, 222}

    for c in calls:
        tid, batch = c[0]
        if tid == 111:
            assert [i.text for i in batch.items if i.type == "text"] == ["user 1 msg"]
            assert batch.chat_id == 999
        else:
            assert [i.text for i in batch.items if i.type == "text"] == ["user 2 msg"]
            assert batch.chat_id == 888


# ---- Error handling ----


async def test_fire_handles_callback_error_gracefully(
    on_batch_ready: AsyncMock,
) -> None:
    """If the callback raises, the batcher logs but does not crash."""
    on_batch_ready.side_effect = RuntimeError("boom")
    batcher = MessageBatcher(debounce_seconds=0.05, on_batch_ready=on_batch_ready)

    await batcher.add_text(111, 999, "hello")
    await asyncio.sleep(0.15)

    # Callback was called (and failed), but batcher state is clean
    on_batch_ready.assert_awaited_once()
    assert 111 not in batcher._batches
    assert 111 not in batcher._timers

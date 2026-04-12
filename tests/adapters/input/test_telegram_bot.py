from unittest.mock import AsyncMock, MagicMock, call

import pytest
from telegram import Update
from telegram.ext import ContextTypes

from kume.adapters.input.message_batcher import BatchItem, MediaItem, MessageBatcher, PendingBatch
from kume.adapters.input.status_messages import get_status_message
from kume.adapters.input.telegram_bot import TelegramBotAdapter
from kume.ports.output.messaging import MessagingPort
from kume.services.ingestion import IngestionService, UnsupportedMediaType
from kume.services.orchestrator import OrchestratorService

# ---- Fixtures ----


@pytest.fixture
def orchestrator() -> AsyncMock:
    return AsyncMock(spec=OrchestratorService)


@pytest.fixture
def messaging() -> AsyncMock:
    return AsyncMock(spec=MessagingPort)


@pytest.fixture
def ingestion() -> AsyncMock:
    return AsyncMock(spec=IngestionService)


@pytest.fixture
def batcher() -> AsyncMock:
    return AsyncMock(spec=MessageBatcher)


@pytest.fixture
def adapter(
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    batcher: AsyncMock,
) -> TelegramBotAdapter:
    """Adapter with batcher (standard path)."""
    return TelegramBotAdapter(
        orchestrator=orchestrator,
        messaging=messaging,
        batcher=batcher,
    )


@pytest.fixture
def media_adapter(
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
    batcher: AsyncMock,
) -> TelegramBotAdapter:
    """Adapter with batcher + ingestion (media path)."""
    return TelegramBotAdapter(
        orchestrator=orchestrator,
        messaging=messaging,
        ingestion=ingestion,
        batcher=batcher,
    )


@pytest.fixture
def batch_adapter(
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> TelegramBotAdapter:
    """Adapter without batcher — for testing _process_batch directly."""
    return TelegramBotAdapter(
        orchestrator=orchestrator,
        messaging=messaging,
        ingestion=ingestion,
    )


# ---- Helpers ----


def _make_text_update(
    text: str,
    user_id: int = 12345,
    chat_id: int = 67890,
    language_code: str | None = "en",
) -> MagicMock:
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.text = text
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.language_code = language_code
    update.effective_user.first_name = "TestUser"
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    return update


def _make_media_update(
    *,
    chat_id: int = 67890,
    user_id: int = 12345,
    document: MagicMock | None = None,
    voice: MagicMock | None = None,
    audio: MagicMock | None = None,
    photo: list[MagicMock] | None = None,
    caption: str | None = None,
    language_code: str | None = "en",
) -> MagicMock:
    update = MagicMock(spec=Update)
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.language_code = language_code
    update.effective_user.first_name = "TestUser"
    update.message = MagicMock()
    update.message.document = document
    update.message.voice = voice
    update.message.audio = audio
    update.message.photo = photo or []
    update.message.caption = caption
    return update


def _make_context_with_file(
    file_bytes: bytes = b"raw-file-data",
    file_size: int | None = None,
) -> MagicMock:
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    tg_file = AsyncMock()
    tg_file.file_size = file_size if file_size is not None else len(file_bytes)
    tg_file.download_as_bytearray.return_value = bytearray(file_bytes)
    context.bot.get_file = AsyncMock(return_value=tg_file)
    return context


# ---- Text handler tests (with batcher) ----


async def test_handle_message_adds_text_to_batcher(
    adapter: TelegramBotAdapter,
    batcher: AsyncMock,
    orchestrator: AsyncMock,
) -> None:
    """Text messages are added to the batcher, not processed directly."""
    update = _make_text_update("What should I eat?")
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    batcher.add_text.assert_awaited_once()
    call_args = batcher.add_text.call_args
    assert call_args[0][:4] == (12345, 67890, "What should I eat?", "en")
    orchestrator.process.assert_not_awaited()


async def test_handle_message_non_text(
    adapter: TelegramBotAdapter,
    batcher: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.text = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    batcher.add_text.assert_not_awaited()
    messaging.send_message.assert_awaited_once_with(
        67890,
        "I can only handle text messages for now.",
    )


async def test_handle_message_no_message(
    adapter: TelegramBotAdapter,
    batcher: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    batcher.add_text.assert_not_awaited()
    messaging.send_message.assert_awaited_once_with(
        67890,
        "I can only handle text messages for now.",
    )


async def test_handle_message_no_message_no_chat(
    adapter: TelegramBotAdapter,
    batcher: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = None
    update.effective_chat = None
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    batcher.add_text.assert_not_awaited()
    messaging.send_message.assert_not_awaited()


async def test_handle_message_too_long(
    adapter: TelegramBotAdapter,
    batcher: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = _make_text_update("x" * 5000)
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    batcher.add_text.assert_not_awaited()
    messaging.send_message.assert_awaited_once()


# ---- Media handler tests (with batcher) ----


async def test_handle_media_downloads_then_adds_to_batcher(
    media_adapter: TelegramBotAdapter,
    batcher: AsyncMock,
    orchestrator: AsyncMock,
) -> None:
    """Media files are downloaded immediately, then added to the batcher."""
    doc = MagicMock()
    doc.file_id = "file_123"
    doc.mime_type = "application/pdf"

    update = _make_media_update(document=doc, caption="My lab results")
    context = _make_context_with_file(b"pdf-bytes")

    await media_adapter.handle_media(update, context)

    # File was downloaded
    context.bot.get_file.assert_awaited_once_with("file_123")

    # Added to batcher (not processed directly)
    batcher.add_media.assert_awaited_once()
    _, kwargs = batcher.add_media.call_args
    if not kwargs:
        args = batcher.add_media.call_args[0]
        assert args[0] == 12345  # telegram_id
        assert args[1] == 67890  # chat_id
        item = args[2]
        assert isinstance(item, MediaItem)
        assert item.raw_bytes == b"pdf-bytes"
        assert item.mime_type == "application/pdf"
        assert item.caption == "My lab results"

    orchestrator.process.assert_not_awaited()


async def test_handle_media_no_ingestion_service(
    adapter: TelegramBotAdapter,
    messaging: AsyncMock,
) -> None:
    """When ingestion is not configured, send a friendly error."""
    doc = MagicMock()
    doc.file_id = "file_123"
    doc.mime_type = "application/pdf"

    update = _make_media_update(document=doc)
    context = _make_context_with_file()

    await adapter.handle_media(update, context)

    messaging.send_message.assert_awaited_once_with(67890, "Media processing is not available.")


async def test_handle_media_rejects_oversized_file(
    media_adapter: TelegramBotAdapter,
    batcher: AsyncMock,
    messaging: AsyncMock,
) -> None:
    """Files exceeding MAX_FILE_SIZE (20 MB) are rejected."""
    doc = MagicMock()
    doc.file_id = "file_big"
    doc.mime_type = "application/pdf"

    update = _make_media_update(document=doc)
    context = _make_context_with_file(b"small", file_size=25 * 1024 * 1024)

    await media_adapter.handle_media(update, context)

    batcher.add_media.assert_not_awaited()
    assert any("too large" in str(c) for c in messaging.send_message.call_args_list)


# ---- _process_batch tests ----


async def test_process_batch_single_text(
    batch_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    """A batch with a single text message produces one orchestrator call."""
    batch = PendingBatch()
    batch.items = [BatchItem(type="text", text="What should I eat?")]
    batch.chat_id = 67890
    batch.language = "en"

    orchestrator.process.return_value = "Eat more vegetables!"

    await batch_adapter._process_batch(12345, batch)

    orchestrator.process.assert_awaited_once_with(12345, "[User message]\nWhat should I eat?", user_name=None)
    # Single text: no status message, just the response
    messaging.send_message.assert_awaited_once_with(67890, "Eat more vegetables!")


async def test_process_batch_single_pdf(
    batch_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> None:
    """A single PDF batch sends the reading_analysis status, then the response."""
    batch = PendingBatch()
    batch.items = [
        BatchItem(
            type="media",
            media=MediaItem(raw_bytes=b"pdf-bytes", mime_type="application/pdf", caption="My labs"),
        )
    ]
    batch.chat_id = 67890
    batch.language = "en"

    ingestion.process.return_value = "Extracted: cholesterol 200mg/dL"
    orchestrator.process.return_value = "Your cholesterol is within range."

    await batch_adapter._process_batch(12345, batch)

    ingestion.process.assert_awaited_once_with(b"pdf-bytes", "application/pdf")
    orchestrator.process.assert_awaited_once_with(
        12345, "[Document] (caption: My labs)\nExtracted: cholesterol 200mg/dL", user_name=None
    )
    messaging.send_message.assert_has_awaits(
        [
            call(67890, get_status_message("reading_analysis", "en")),
            call(67890, "Your cholesterol is within range."),
        ]
    )


async def test_process_batch_multiple_pdfs_parallel(
    batch_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> None:
    """Multiple PDFs are extracted in order and combined."""
    batch = PendingBatch()
    batch.items = [
        BatchItem(
            type="media",
            media=MediaItem(raw_bytes=b"pdf1", mime_type="application/pdf", caption="Report 1"),
        ),
        BatchItem(
            type="media",
            media=MediaItem(raw_bytes=b"pdf2", mime_type="application/pdf", caption="Report 2"),
        ),
        BatchItem(
            type="media",
            media=MediaItem(raw_bytes=b"pdf3", mime_type="application/pdf", caption=""),
        ),
    ]
    batch.chat_id = 67890
    batch.language = "es"

    ingestion.process.side_effect = ["Extracted 1", "Extracted 2", "Extracted 3"]
    orchestrator.process.return_value = "Comparative analysis..."

    await batch_adapter._process_batch(12345, batch)

    # All 3 PDFs processed
    assert ingestion.process.await_count == 3

    # Orchestrator gets combined content
    combined = orchestrator.process.call_args[0][1]
    assert "Report 1" in combined
    assert "Extracted 1" in combined
    assert "Report 2" in combined
    assert "Extracted 2" in combined
    assert "Extracted 3" in combined
    # Empty caption (Report 3) should not add extra text
    assert combined.count("Report") == 2  # only 2 captions

    # Batch status message for multiple items
    messaging.send_message.assert_has_awaits(
        [
            call(67890, get_status_message("processing_batch", "es")),
            call(67890, "Comparative analysis..."),
        ]
    )


async def test_process_batch_mixed_text_and_pdf(
    batch_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> None:
    """A batch with text + PDF produces one combined orchestrator call."""
    batch = PendingBatch()
    batch.items = [
        BatchItem(type="text", text="Estos son mis análisis"),
        BatchItem(
            type="media",
            media=MediaItem(raw_bytes=b"pdf-bytes", mime_type="application/pdf", caption="Lab results"),
        ),
    ]
    batch.chat_id = 67890
    batch.language = "es"

    ingestion.process.return_value = "cholesterol: 200"
    orchestrator.process.return_value = "Tu colesterol está bien."

    await batch_adapter._process_batch(12345, batch)

    combined = orchestrator.process.call_args[0][1]
    assert "Estos son mis análisis" in combined
    assert "Lab results" in combined
    assert "cholesterol: 200" in combined

    # Multiple items -> batch status
    messaging.send_message.assert_has_awaits(
        [
            call(67890, get_status_message("processing_batch", "es")),
            call(67890, "Tu colesterol está bien."),
        ]
    )


async def test_process_batch_single_audio(
    batch_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> None:
    """A single audio batch sends the transcribing status."""
    batch = PendingBatch()
    batch.items = [
        BatchItem(
            type="media",
            media=MediaItem(raw_bytes=b"ogg-bytes", mime_type="audio/ogg", caption=""),
        )
    ]
    batch.chat_id = 67890
    batch.language = "en"

    ingestion.process.return_value = "I ate a salad for lunch"
    orchestrator.process.return_value = "Great choice!"

    await batch_adapter._process_batch(12345, batch)

    messaging.send_message.assert_has_awaits(
        [
            call(67890, get_status_message("transcribing_audio", "en")),
            call(67890, "Great choice!"),
        ]
    )


async def test_process_batch_unsupported_media(
    batch_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> None:
    """Unsupported media type in batch is skipped; if all items are unsupported, error message is sent."""
    batch = PendingBatch()
    batch.items = [
        BatchItem(
            type="media",
            media=MediaItem(raw_bytes=b"video", mime_type="video/mp4", caption=""),
        )
    ]
    batch.chat_id = 67890
    batch.language = "en"

    ingestion.process.side_effect = UnsupportedMediaType("video/mp4")

    await batch_adapter._process_batch(12345, batch)

    orchestrator.process.assert_not_awaited()
    messaging.send_message.assert_has_awaits(
        [
            call(67890, get_status_message("processing_media", "en")),
            call(67890, get_status_message("unsupported_media", "en")),
        ]
    )


async def test_process_batch_error_handling(
    batch_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    """Orchestrator errors are caught and a friendly message is sent."""
    batch = PendingBatch()
    batch.items = [BatchItem(type="text", text="hello")]
    batch.chat_id = 67890
    batch.language = "en"

    orchestrator.process.side_effect = RuntimeError("LLM failed")

    await batch_adapter._process_batch(12345, batch)

    messaging.send_message.assert_awaited_once_with(67890, "Sorry, something went wrong. Please try again.")

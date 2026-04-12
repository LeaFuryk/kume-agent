from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Update
from telegram.ext import ContextTypes

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
def adapter(orchestrator: AsyncMock, messaging: AsyncMock) -> TelegramBotAdapter:
    return TelegramBotAdapter(orchestrator=orchestrator, messaging=messaging)


@pytest.fixture
def media_adapter(
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> TelegramBotAdapter:
    return TelegramBotAdapter(orchestrator=orchestrator, messaging=messaging, ingestion=ingestion)


# ---- Existing text handler tests ----


async def test_handle_message_valid_text(
    adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.text = "What should I eat?"
    update.effective_user = MagicMock()
    update.effective_user.id = 12345
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    orchestrator.process.return_value = "Eat more vegetables!"

    await adapter.handle_message(update, context)

    orchestrator.process.assert_awaited_once_with(12345, "What should I eat?")
    messaging.send_message.assert_awaited_once_with(67890, "Eat more vegetables!")


async def test_handle_message_non_text(
    adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.text = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    orchestrator.process.assert_not_awaited()
    messaging.send_message.assert_awaited_once_with(
        67890,
        "I can only handle text messages for now.",
    )


async def test_handle_message_no_message(
    adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    orchestrator.process.assert_not_awaited()
    messaging.send_message.assert_awaited_once_with(
        67890,
        "I can only handle text messages for now.",
    )


async def test_handle_message_no_message_no_chat(
    adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = None
    update.effective_chat = None
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    orchestrator.process.assert_not_awaited()
    messaging.send_message.assert_not_awaited()


# ---- Media handler tests ----


def _make_media_update(
    *,
    chat_id: int = 67890,
    user_id: int = 12345,
    document: MagicMock | None = None,
    voice: MagicMock | None = None,
    audio: MagicMock | None = None,
    photo: list[MagicMock] | None = None,
    caption: str | None = None,
) -> MagicMock:
    update = MagicMock(spec=Update)
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.message = MagicMock()
    update.message.document = document
    update.message.voice = voice
    update.message.audio = audio
    update.message.photo = photo or []
    update.message.caption = caption
    return update


def _make_context_with_file(file_bytes: bytes = b"raw-file-data") -> MagicMock:
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    tg_file = AsyncMock()
    tg_file.download_as_bytearray.return_value = bytearray(file_bytes)
    context.bot.get_file = AsyncMock(return_value=tg_file)
    return context


async def test_handle_media_pdf_document(
    media_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> None:
    doc = MagicMock()
    doc.file_id = "file_123"
    doc.mime_type = "application/pdf"

    update = _make_media_update(document=doc, caption="My lab results")
    context = _make_context_with_file(b"pdf-bytes")

    ingestion.process.return_value = "Extracted: cholesterol 200mg/dL"
    orchestrator.process.return_value = "Your cholesterol is within range."

    await media_adapter.handle_media(update, context)

    context.bot.get_file.assert_awaited_once_with("file_123")
    ingestion.process.assert_awaited_once_with(b"pdf-bytes", "application/pdf")
    orchestrator.process.assert_awaited_once_with(
        12345,
        "My lab results\n\nExtracted: cholesterol 200mg/dL",
    )
    messaging.send_message.assert_awaited_once_with(67890, "Your cholesterol is within range.")


async def test_handle_media_voice_message(
    media_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> None:
    voice = MagicMock()
    voice.file_id = "voice_456"
    voice.mime_type = "audio/ogg"

    update = _make_media_update(voice=voice)
    context = _make_context_with_file(b"ogg-bytes")

    ingestion.process.return_value = "I ate a salad for lunch"
    orchestrator.process.return_value = "Great choice!"

    await media_adapter.handle_media(update, context)

    context.bot.get_file.assert_awaited_once_with("voice_456")
    ingestion.process.assert_awaited_once_with(b"ogg-bytes", "audio/ogg")
    # No caption, so only extracted text is sent
    orchestrator.process.assert_awaited_once_with(12345, "I ate a salad for lunch")
    messaging.send_message.assert_awaited_once_with(67890, "Great choice!")


async def test_handle_media_unsupported_type_sends_error(
    media_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> None:
    doc = MagicMock()
    doc.file_id = "file_789"
    doc.mime_type = "video/mp4"

    update = _make_media_update(document=doc)
    context = _make_context_with_file(b"video-bytes")

    ingestion.process.side_effect = UnsupportedMediaType("video/mp4")

    await media_adapter.handle_media(update, context)

    orchestrator.process.assert_not_awaited()
    messaging.send_message.assert_awaited_once_with(
        67890,
        "Sorry, I don't support video/mp4 files yet.",
    )


async def test_handle_media_photo(
    media_adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
    ingestion: AsyncMock,
) -> None:
    photo_small = MagicMock()
    photo_small.file_id = "photo_small"
    photo_large = MagicMock()
    photo_large.file_id = "photo_large"

    update = _make_media_update(photo=[photo_small, photo_large], caption="What is this?")
    context = _make_context_with_file(b"jpeg-bytes")

    ingestion.process.return_value = "A plate of grilled chicken with rice"
    orchestrator.process.return_value = "That looks like a balanced meal."

    await media_adapter.handle_media(update, context)

    # Should use the largest photo (last in list)
    context.bot.get_file.assert_awaited_once_with("photo_large")
    ingestion.process.assert_awaited_once_with(b"jpeg-bytes", "image/jpeg")
    orchestrator.process.assert_awaited_once_with(
        12345,
        "What is this?\n\nA plate of grilled chicken with rice",
    )


async def test_handle_media_no_ingestion_service(
    adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    """When ingestion is not configured, send a friendly error."""
    doc = MagicMock()
    doc.file_id = "file_123"
    doc.mime_type = "application/pdf"

    update = _make_media_update(document=doc)
    context = _make_context_with_file()

    await adapter.handle_media(update, context)

    orchestrator.process.assert_not_awaited()
    messaging.send_message.assert_awaited_once_with(67890, "Media processing is not available.")

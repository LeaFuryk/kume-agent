import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from kume.adapters.input.message_batcher import MediaItem, MessageBatcher, PendingBatch
from kume.adapters.input.status_messages import get_status_message
from kume.ports.output.messaging import MessagingPort
from kume.services.ingestion import IngestionService, UnsupportedMediaType
from kume.services.orchestrator import OrchestratorService

logger = logging.getLogger("kume.telegram")

MAX_MESSAGE_LENGTH = 4096
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB — Telegram's own download limit
MAX_EXTRACTED_TEXT = 8000


class TelegramBotAdapter:
    def __init__(
        self,
        orchestrator: OrchestratorService,
        messaging: MessagingPort,
        ingestion: IngestionService | None = None,
        batcher: MessageBatcher | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._messaging = messaging
        self._ingestion = ingestion
        self._batcher = batcher

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.message.text:
            if update.effective_chat:
                await self._messaging.send_message(
                    update.effective_chat.id,
                    "I can only handle text messages for now.",
                )
            return

        telegram_id = update.effective_user.id if update.effective_user else 0
        chat_id = update.effective_chat.id if update.effective_chat else 0
        lang = update.effective_user.language_code if update.effective_user else None
        text = update.message.text

        if len(text) > MAX_MESSAGE_LENGTH:
            await self._messaging.send_message(
                chat_id, "Your message is too long. Please keep it under 4096 characters."
            )
            return

        if self._batcher:
            logger.info("Queuing text from telegram_id=%d", telegram_id)
            await self._batcher.add_text(telegram_id, chat_id, text, lang)
        else:
            # Fallback: process immediately (no batcher configured)
            logger.info("Received message from telegram_id=%d", telegram_id)
            response = await self._orchestrator.process(telegram_id, text)
            await self._messaging.send_message(chat_id, response)

    async def handle_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        telegram_id = update.effective_user.id if update.effective_user else 0
        lang = update.effective_user.language_code if update.effective_user else None

        if not update.message:
            return

        file_id: str | None = None
        mime_type: str | None = None

        if update.message.document:
            file_id = update.message.document.file_id
            mime_type = update.message.document.mime_type
        elif update.message.voice:
            file_id = update.message.voice.file_id
            mime_type = update.message.voice.mime_type
        elif update.message.audio:
            file_id = update.message.audio.file_id
            mime_type = update.message.audio.mime_type
        elif update.message.photo:
            photo = update.message.photo[-1]
            file_id = photo.file_id
            mime_type = "image/jpeg"

        if not file_id or not mime_type:
            await self._messaging.send_message(chat_id, "Could not determine the file type.")
            return

        if not self._ingestion:
            await self._messaging.send_message(chat_id, "Media processing is not available.")
            return

        # Download file bytes IMMEDIATELY — before adding to batcher.
        # We don't want the debounce timer to expire while waiting for a large download.
        logger.info("Downloading media (mime=%s) from telegram_id=%d", mime_type, telegram_id)
        try:
            tg_file = await context.bot.get_file(file_id)
            if tg_file.file_size and tg_file.file_size > MAX_FILE_SIZE:
                await self._messaging.send_message(chat_id, "The file is too large. Please send files under 20 MB.")
                return
            raw_bytes = bytes(await tg_file.download_as_bytearray())
        except Exception:
            logger.exception("Error downloading media for telegram_id=%d", telegram_id)
            await self._messaging.send_message(
                chat_id, "Sorry, something went wrong while downloading your file. Please try again."
            )
            return

        caption = update.message.caption or ""
        item = MediaItem(raw_bytes=raw_bytes, mime_type=mime_type, caption=caption)

        if self._batcher:
            logger.info("Queuing media (mime=%s) from telegram_id=%d", mime_type, telegram_id)
            await self._batcher.add_media(telegram_id, chat_id, item, lang)
        else:
            # Fallback: process immediately (no batcher configured)
            await self._process_single_media(telegram_id, chat_id, lang, item)

    async def _process_batch(self, telegram_id: int, batch: PendingBatch) -> None:
        """Process a debounced batch: extract content, combine, and call orchestrator once."""
        chat_id = batch.chat_id
        lang = batch.language

        total_items = len(batch.texts) + len(batch.media)
        is_single_item = total_items == 1

        try:
            # Send appropriate status message
            if is_single_item and len(batch.media) == 1:
                media = batch.media[0]
                if media.mime_type == "application/pdf":
                    await self._messaging.send_message(chat_id, get_status_message("reading_analysis", lang))
                elif media.mime_type.startswith("audio/"):
                    await self._messaging.send_message(chat_id, get_status_message("transcribing_audio", lang))
                else:
                    await self._messaging.send_message(chat_id, get_status_message("processing_media", lang))
            elif not is_single_item:
                await self._messaging.send_message(chat_id, get_status_message("processing_batch", lang))
            # Single text message: no status needed

            # Extract media content
            parts: list[str] = []

            # Add text messages in order
            for text in batch.texts:
                parts.append(text)

            # Separate PDFs (parallel) from audio (sequential) and other media
            pdf_items: list[MediaItem] = []
            audio_items: list[MediaItem] = []
            other_items: list[MediaItem] = []

            for media in batch.media:
                if media.mime_type == "application/pdf":
                    pdf_items.append(media)
                elif media.mime_type.startswith("audio/"):
                    audio_items.append(media)
                else:
                    other_items.append(media)

            assert self._ingestion is not None or not batch.media, "Ingestion required for media"

            # Extract PDFs in parallel
            pdf_results: list[str] = []
            if pdf_items and self._ingestion:
                pdf_tasks = [self._ingestion.process(m.raw_bytes, m.mime_type) for m in pdf_items]
                pdf_results = list(await asyncio.gather(*pdf_tasks))

            for item, extracted in zip(pdf_items, pdf_results, strict=True):
                if item.caption:
                    parts.append(item.caption)
                parts.append(extracted)

            # Process audio sequentially
            for item in audio_items:
                if self._ingestion:
                    extracted = await self._ingestion.process(item.raw_bytes, item.mime_type)
                    if item.caption:
                        parts.append(item.caption)
                    parts.append(extracted)

            # Process other media (images, etc.)
            for item in other_items:
                if self._ingestion:
                    extracted = await self._ingestion.process(item.raw_bytes, item.mime_type)
                    if item.caption:
                        parts.append(item.caption)
                    parts.append(extracted)

            combined = "\n\n".join(parts)

            # Truncate if needed
            if len(combined) > MAX_EXTRACTED_TEXT:
                combined = combined[:MAX_EXTRACTED_TEXT] + "\n\n[Text truncated — original was longer]"

            # ONE orchestrator call -> ONE response
            response = await self._orchestrator.process(telegram_id, combined)
            await self._messaging.send_message(chat_id, response)

        except UnsupportedMediaType:
            await self._messaging.send_message(chat_id, get_status_message("unsupported_media", lang))
        except Exception:
            logger.exception("Error processing batch for telegram_id=%d", telegram_id)
            await self._messaging.send_message(chat_id, "Sorry, something went wrong. Please try again.")

    async def _process_single_media(
        self,
        telegram_id: int,
        chat_id: int,
        lang: str | None,
        item: MediaItem,
    ) -> None:
        """Fallback for processing a single media item without the batcher."""
        if item.mime_type == "application/pdf":
            await self._messaging.send_message(chat_id, get_status_message("reading_analysis", lang))
        elif item.mime_type.startswith("audio/"):
            await self._messaging.send_message(chat_id, get_status_message("transcribing_audio", lang))
        else:
            await self._messaging.send_message(chat_id, get_status_message("processing_media", lang))

        try:
            assert self._ingestion is not None
            extracted_text = await self._ingestion.process(item.raw_bytes, item.mime_type)
            if len(extracted_text) > MAX_EXTRACTED_TEXT:
                extracted_text = extracted_text[:MAX_EXTRACTED_TEXT] + "\n\n[Text truncated — original was longer]"
        except UnsupportedMediaType:
            await self._messaging.send_message(chat_id, get_status_message("unsupported_media", lang))
            return
        except Exception:
            logger.exception("Error processing media for telegram_id=%d", telegram_id)
            await self._messaging.send_message(
                chat_id, "Sorry, something went wrong while processing your file. Please try again."
            )
            return

        combined = f"{item.caption}\n\n{extracted_text}".strip() if item.caption else extracted_text

        try:
            response = await self._orchestrator.process(telegram_id, combined)
            await self._messaging.send_message(chat_id, response)
        except Exception:
            logger.exception("Error in orchestrator for media message, telegram_id=%d", telegram_id)
            await self._messaging.send_message(chat_id, "Sorry, something went wrong. Please try again.")

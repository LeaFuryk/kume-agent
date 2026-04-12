from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import ContextTypes

from kume.adapters.input.message_batcher import MediaItem, MessageBatcher, PendingBatch
from kume.adapters.input.status_messages import get_status_message
from kume.ports.output.messaging import MessagingPort
from kume.services.ingestion import IngestionService, UnsupportedMediaType
from kume.services.orchestrator import OrchestratorService, Resource

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
        user_name = update.effective_user.first_name if update.effective_user else None
        text = update.message.text

        if len(text) > MAX_MESSAGE_LENGTH:
            await self._messaging.send_message(
                chat_id, "Your message is too long. Please keep it under 4096 characters."
            )
            return

        if self._batcher:
            logger.info("Queuing text from telegram_id=%d", telegram_id)
            await self._batcher.add_text(telegram_id, chat_id, text, lang, user_name=user_name)
        else:
            logger.info("Received message from telegram_id=%d", telegram_id)
            response = await self._orchestrator.process(telegram_id, user_message=text, user_name=user_name)
            await self._messaging.send_message(chat_id, response)

    async def handle_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        telegram_id = update.effective_user.id if update.effective_user else 0
        lang = update.effective_user.language_code if update.effective_user else None
        user_name = update.effective_user.first_name if update.effective_user else None

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

        # Notify batcher that a download is starting — prevents premature timer firing
        if self._batcher:
            self._batcher.notify_download_started(telegram_id)

        logger.info("Downloading media (mime=%s) from telegram_id=%d", mime_type, telegram_id)
        try:
            tg_file = await context.bot.get_file(file_id)
            if tg_file.file_size and tg_file.file_size > MAX_FILE_SIZE:
                if self._batcher:
                    self._batcher.notify_download_finished(telegram_id)
                await self._messaging.send_message(chat_id, "The file is too large. Please send files under 20 MB.")
                return
            raw_bytes = bytes(await tg_file.download_as_bytearray())
        except Exception:
            if self._batcher:
                self._batcher.notify_download_finished(telegram_id)
            logger.exception("Error downloading media for telegram_id=%d", telegram_id)
            await self._messaging.send_message(
                chat_id, "Sorry, something went wrong while downloading your file. Please try again."
            )
            return

        caption = update.message.caption or ""
        item = MediaItem(raw_bytes=raw_bytes, mime_type=mime_type, caption=caption)

        if self._batcher:
            self._batcher.notify_download_finished(telegram_id)
            logger.info("Queuing media (mime=%s) from telegram_id=%d", mime_type, telegram_id)
            try:
                await self._batcher.add_media(telegram_id, chat_id, item, lang, user_name=user_name)
            except ValueError:
                await self._messaging.send_message(chat_id, "Too many files at once. Please send fewer files.")
        else:
            # Fallback: process immediately (no batcher configured)
            await self._process_single_media(telegram_id, chat_id, lang, item, user_name=user_name)

    async def _process_batch(self, telegram_id: int, batch: PendingBatch) -> None:
        """Process a debounced batch: extract content, build resources, and call orchestrator once."""
        chat_id = batch.chat_id
        lang = batch.language

        total_items = len(batch.items)
        is_single_item = total_items == 1

        try:
            # Send appropriate status message
            if is_single_item and batch.items[0].type == "media":
                media = batch.items[0].media
                assert media is not None
                if media.mime_type == "application/pdf":
                    await self._messaging.send_message(chat_id, get_status_message("reading_analysis", lang))
                elif media.mime_type.startswith("audio/"):
                    await self._messaging.send_message(chat_id, get_status_message("transcribing_audio", lang))
                else:
                    await self._messaging.send_message(chat_id, get_status_message("processing_media", lang))
            elif not is_single_item:
                await self._messaging.send_message(chat_id, get_status_message("processing_batch", lang))
            # Single text message: no status needed

            user_texts: list[str] = []
            resources: list[Resource] = []
            skipped_items = 0

            for batch_item in batch.items:
                if batch_item.type == "text" and batch_item.text:
                    user_texts.append(batch_item.text)
                elif batch_item.type == "media" and batch_item.media:
                    media = batch_item.media
                    assert self._ingestion is not None, "Ingestion required for media"
                    try:
                        transcript = await self._ingestion.process(media.raw_bytes, media.mime_type)
                    except UnsupportedMediaType:
                        logger.warning(
                            "Skipping unsupported media type %s for telegram_id=%d",
                            media.mime_type,
                            telegram_id,
                        )
                        skipped_items += 1
                        continue
                    resources.append(
                        Resource(
                            mime_type=media.mime_type,
                            transcript=transcript,
                            raw_bytes=media.raw_bytes if media.mime_type.startswith("image/") else None,
                        )
                    )

            if not user_texts and not resources:
                # All items were skipped (unsupported)
                await self._messaging.send_message(chat_id, get_status_message("unsupported_media", lang))
                return

            user_message = "\n".join(user_texts)

            # Truncate transcripts if needed
            for idx, r in enumerate(resources):
                if len(r.transcript) > MAX_EXTRACTED_TEXT:
                    resources[idx] = Resource(
                        mime_type=r.mime_type,
                        transcript=r.transcript[:MAX_EXTRACTED_TEXT] + "\n[truncated]",
                        raw_bytes=r.raw_bytes,
                    )

            # ONE orchestrator call -> ONE response
            response = await self._orchestrator.process(
                telegram_id=telegram_id,
                user_message=user_message,
                user_name=batch.user_name,
                resources=resources if resources else None,
            )
            await self._messaging.send_message(chat_id, response)

        except Exception:
            logger.exception("Error processing batch for telegram_id=%d", telegram_id)
            await self._messaging.send_message(chat_id, "Sorry, something went wrong. Please try again.")

    async def _process_single_media(
        self,
        telegram_id: int,
        chat_id: int,
        lang: str | None,
        item: MediaItem,
        user_name: str | None = None,
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
            transcript = await self._ingestion.process(item.raw_bytes, item.mime_type)
            if len(transcript) > MAX_EXTRACTED_TEXT:
                transcript = transcript[:MAX_EXTRACTED_TEXT] + "\n[truncated]"
        except UnsupportedMediaType:
            await self._messaging.send_message(chat_id, get_status_message("unsupported_media", lang))
            return
        except Exception:
            logger.exception("Error processing media for telegram_id=%d", telegram_id)
            await self._messaging.send_message(
                chat_id, "Sorry, something went wrong while processing your file. Please try again."
            )
            return

        resource = Resource(
            mime_type=item.mime_type,
            transcript=transcript,
            raw_bytes=item.raw_bytes if item.mime_type.startswith("image/") else None,
        )
        user_message = item.caption or ""

        try:
            response = await self._orchestrator.process(
                telegram_id=telegram_id,
                user_message=user_message,
                user_name=user_name,
                resources=[resource],
            )
            await self._messaging.send_message(chat_id, response)
        except Exception:
            logger.exception("Error in orchestrator for media message, telegram_id=%d", telegram_id)
            await self._messaging.send_message(chat_id, "Sorry, something went wrong. Please try again.")

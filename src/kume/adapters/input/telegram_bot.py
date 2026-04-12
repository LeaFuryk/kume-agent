import logging

from telegram import Update
from telegram.ext import ContextTypes

from kume.ports.output.messaging import MessagingPort
from kume.services.ingestion import IngestionService, UnsupportedMediaType
from kume.services.orchestrator import OrchestratorService

logger = logging.getLogger("kume.telegram")

MAX_MESSAGE_LENGTH = 4096


class TelegramBotAdapter:
    def __init__(
        self,
        orchestrator: OrchestratorService,
        messaging: MessagingPort,
        ingestion: IngestionService | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._messaging = messaging
        self._ingestion = ingestion

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
        text = update.message.text

        if len(text) > MAX_MESSAGE_LENGTH:
            await self._messaging.send_message(
                chat_id, "Your message is too long. Please keep it under 4096 characters."
            )
            return

        logger.info("Received message from telegram_id=%d", telegram_id)

        response = await self._orchestrator.process(telegram_id, text)
        await self._messaging.send_message(chat_id, response)

    async def handle_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id if update.effective_chat else 0
        telegram_id = update.effective_user.id if update.effective_user else 0

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
            # photo is a list of PhotoSize; take the largest (last)
            photo = update.message.photo[-1]
            file_id = photo.file_id
            mime_type = "image/jpeg"

        if not file_id or not mime_type:
            await self._messaging.send_message(chat_id, "Could not determine the file type.")
            return

        if not self._ingestion:
            await self._messaging.send_message(chat_id, "Media processing is not available.")
            return

        logger.info("Received media (mime=%s) from telegram_id=%d", mime_type, telegram_id)

        try:
            tg_file = await context.bot.get_file(file_id)
            raw_bytes = bytes(await tg_file.download_as_bytearray())
            extracted_text = await self._ingestion.process(raw_bytes, mime_type)
        except UnsupportedMediaType as exc:
            await self._messaging.send_message(
                chat_id,
                f"Sorry, I don't support {exc.mime_type} files yet.",
            )
            return

        caption = update.message.caption or ""
        combined = f"{caption}\n\n{extracted_text}".strip() if caption else extracted_text

        response = await self._orchestrator.process(telegram_id, combined)
        await self._messaging.send_message(chat_id, response)

import logging

from telegram import Update
from telegram.ext import ContextTypes

from kume.ports.output.messaging import MessagingPort
from kume.services.orchestrator import OrchestratorService

logger = logging.getLogger("kume.telegram")


class TelegramBotAdapter:
    def __init__(
        self,
        orchestrator: OrchestratorService,
        messaging: MessagingPort,
    ) -> None:
        self._orchestrator = orchestrator
        self._messaging = messaging

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

        logger.info("Received message from telegram_id=%d", telegram_id)

        response = await self._orchestrator.process(telegram_id, text)
        await self._messaging.send_message(chat_id, response)

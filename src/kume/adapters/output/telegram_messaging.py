from telegram import Bot
from telegram.constants import ParseMode

from kume.adapters.output.telegram_formatting import markdown_to_telegram_html
from kume.ports.output.messaging import MessagingPort

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


class TelegramMessagingAdapter(MessagingPort):
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_message(self, chat_id: int, text: str) -> None:
        # Split raw text into safe chunks, format each independently,
        # then re-split if formatting expanded beyond the limit
        for chunk in _split_message(text):
            formatted = markdown_to_telegram_html(chunk)
            if len(formatted) <= TELEGRAM_MAX_MESSAGE_LENGTH:
                await self._bot.send_message(chat_id=chat_id, text=formatted, parse_mode=ParseMode.HTML)
            else:
                # Formatting expanded beyond limit — send as plain text chunks
                for plain_chunk in _split_message(formatted):
                    await self._bot.send_message(chat_id=chat_id, text=plain_chunk)


def _split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a message into chunks that fit within Telegram's message length limit."""
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks

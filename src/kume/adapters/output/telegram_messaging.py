from telegram import Bot
from telegram.constants import ParseMode

from kume.adapters.output.telegram_formatting import markdown_to_telegram_html
from kume.ports.output.messaging import MessagingPort

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


class TelegramMessagingAdapter(MessagingPort):
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_message(self, chat_id: int, text: str) -> None:
        for chunk in _split_message(text):
            formatted = markdown_to_telegram_html(chunk)
            # Keep re-splitting until every piece fits
            sub_chunks = _split_message(formatted)
            for sc in sub_chunks:
                await self._bot.send_message(chat_id=chat_id, text=sc, parse_mode=ParseMode.HTML)


def _split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a message into chunks that fit within Telegram's message length limit.

    Prefers splitting at newline boundaries. Preserves newlines in output.
    """
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        # Look for a newline to split at (skip index 0 to avoid empty chunks)
        split_at = text.rfind("\n", 1, max_length)
        if split_at <= 0:
            split_at = max_length
        chunk = text[:split_at]
        if chunk:
            chunks.append(chunk)
        # Keep the newline with the next chunk to preserve formatting
        text = text[split_at:]
    return chunks

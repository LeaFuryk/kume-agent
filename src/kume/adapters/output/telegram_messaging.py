from telegram import Bot
from telegram.constants import ParseMode

from kume.adapters.output.telegram_formatting import markdown_to_telegram_html
from kume.ports.output.messaging import MessagingPort

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


class TelegramMessagingAdapter(MessagingPort):
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_message(self, chat_id: int, text: str) -> None:
        # Split raw text into safe chunks, format each independently
        for chunk in _split_message(text):
            formatted = markdown_to_telegram_html(chunk)
            if len(formatted) <= TELEGRAM_MAX_MESSAGE_LENGTH:
                await self._bot.send_message(chat_id=chat_id, text=formatted, parse_mode=ParseMode.HTML)
            else:
                # Formatting expanded beyond limit — re-split the raw chunk
                # and format each sub-chunk individually
                for sub_chunk in _split_message(chunk, max_length=max(1, TELEGRAM_MAX_MESSAGE_LENGTH // 2)):
                    sub_formatted = markdown_to_telegram_html(sub_chunk)
                    await self._bot.send_message(chat_id=chat_id, text=sub_formatted, parse_mode=ParseMode.HTML)


def _split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a message into chunks that fit within Telegram's message length limit."""
    if len(text) <= max_length:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 1, max_length)
        if split_at <= 0:
            split_at = max_length
        chunk = text[:split_at]
        if chunk:
            chunks.append(chunk)
        text = text[split_at:].lstrip("\n")
    return chunks

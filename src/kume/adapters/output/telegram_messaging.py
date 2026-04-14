import html as html_lib
import re

from telegram import Bot
from telegram.constants import ParseMode

from kume.adapters.output.telegram_formatting import markdown_to_telegram_html
from kume.ports.output.messaging import MessagingPort

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


class TelegramMessagingAdapter(MessagingPort):
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_message(self, chat_id: int, text: str) -> None:
        formatted = markdown_to_telegram_html(text)
        for chunk in _split_message(formatted):
            if _is_valid_html(chunk):
                await self._bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML)
            else:
                # Chunk has broken HTML tags — strip tags and send as plain text
                plain = _strip_html(chunk)
                await self._bot.send_message(chat_id=chat_id, text=plain)

    async def send_and_get_id(self, chat_id: int, text: str) -> int:
        formatted = markdown_to_telegram_html(text)
        result = await self._bot.send_message(chat_id=chat_id, text=formatted, parse_mode=ParseMode.HTML)
        return result.message_id

    async def edit_message(self, chat_id: int, message_id: int, text: str) -> None:
        formatted = markdown_to_telegram_html(text)
        try:
            if _is_valid_html(formatted):
                await self._bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=formatted,
                    parse_mode=ParseMode.HTML,
                )
            else:
                plain = _strip_html(formatted)
                await self._bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=plain)
        except Exception as e:
            # Telegram raises BadRequest for "message is not modified" — swallow it
            if "message is not modified" in str(e).lower():
                return
            raise


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
        text = text[split_at:]
    return chunks


def _is_valid_html(text: str) -> bool:
    """Check if HTML tags are properly balanced (basic check for Telegram compatibility)."""
    tags = re.findall(r"<(/?)(\w+)[^>]*>", text)
    stack: list[str] = []
    for is_closing, tag_name in tags:
        if is_closing:
            if not stack or stack[-1] != tag_name:
                return False
            stack.pop()
        else:
            stack.append(tag_name)
    return len(stack) == 0


def _strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    clean = re.sub(r"<[^>]+>", "", text)
    return html_lib.unescape(clean)

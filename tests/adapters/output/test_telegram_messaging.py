from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Bot
from telegram.constants import ParseMode

from kume.adapters.output.telegram_messaging import TelegramMessagingAdapter, _split_message
from kume.ports.output.messaging import MessagingPort


@pytest.fixture
def bot() -> MagicMock:
    mock_bot = MagicMock(spec=Bot)
    mock_bot.send_message = AsyncMock()
    mock_bot.edit_message_text = AsyncMock()
    return mock_bot


@pytest.fixture
def adapter(bot: MagicMock) -> TelegramMessagingAdapter:
    return TelegramMessagingAdapter(bot=bot)


async def test_send_message_calls_bot_with_html_parse_mode(adapter: TelegramMessagingAdapter, bot: MagicMock) -> None:
    await adapter.send_message(chat_id=12345, text="Hello!")

    bot.send_message.assert_awaited_once_with(chat_id=12345, text="Hello!", parse_mode=ParseMode.HTML)


def test_implements_messaging_port(adapter: TelegramMessagingAdapter) -> None:
    assert isinstance(adapter, MessagingPort)


async def test_long_message_is_chunked(adapter: TelegramMessagingAdapter, bot: MagicMock) -> None:
    """Messages longer than 4096 chars after formatting are split into multiple send_message calls."""
    long_text = "a" * 5000
    await adapter.send_message(chat_id=1, text=long_text)
    assert bot.send_message.await_count >= 2


# --- send_and_get_id tests ---


async def test_send_and_get_id_returns_message_id(adapter: TelegramMessagingAdapter, bot: MagicMock) -> None:
    sent_message = MagicMock()
    sent_message.message_id = 42
    bot.send_message.return_value = sent_message

    result = await adapter.send_and_get_id(chat_id=12345, text="Hello!")

    assert result == 42
    bot.send_message.assert_awaited_once_with(chat_id=12345, text="Hello!", parse_mode=ParseMode.HTML)


# --- edit_message tests ---


async def test_edit_message_calls_edit_message_text(adapter: TelegramMessagingAdapter, bot: MagicMock) -> None:
    await adapter.edit_message(chat_id=12345, message_id=42, text="Updated!")

    bot.edit_message_text.assert_awaited_once_with(
        chat_id=12345,
        message_id=42,
        text="Updated!",
        parse_mode=ParseMode.HTML,
    )


async def test_edit_message_swallows_not_modified(adapter: TelegramMessagingAdapter, bot: MagicMock) -> None:
    bot.edit_message_text.side_effect = Exception("Bad Request: message is not modified")

    # Should not raise
    await adapter.edit_message(chat_id=12345, message_id=42, text="Same text")


async def test_edit_message_propagates_other_errors(adapter: TelegramMessagingAdapter, bot: MagicMock) -> None:
    bot.edit_message_text.side_effect = Exception("Bad Request: chat not found")

    with pytest.raises(Exception, match="chat not found"):
        await adapter.edit_message(chat_id=12345, message_id=42, text="Updated!")


# --- _split_message tests ---


def test_split_message_short() -> None:
    assert _split_message("short") == ["short"]


def test_split_message_exact_limit() -> None:
    text = "a" * 4096
    assert _split_message(text) == [text]


def test_split_message_over_limit() -> None:
    text = "a" * 5000
    chunks = _split_message(text)
    assert len(chunks) == 2
    assert "".join(chunks) == text


def test_split_message_prefers_newline_boundary() -> None:
    text = "line1\n" + "a" * 4090 + "\nline3"
    chunks = _split_message(text)
    assert len(chunks) >= 2
    assert chunks[0].endswith("line1") or "\n" not in chunks[0][-1:]

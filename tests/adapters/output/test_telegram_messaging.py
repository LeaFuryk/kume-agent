from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Bot

from kume.adapters.output.telegram_messaging import TelegramMessagingAdapter
from kume.ports.output.messaging import MessagingPort


@pytest.fixture
def bot() -> MagicMock:
    mock_bot = MagicMock(spec=Bot)
    mock_bot.send_message = AsyncMock()
    return mock_bot


@pytest.fixture
def adapter(bot: MagicMock) -> TelegramMessagingAdapter:
    return TelegramMessagingAdapter(bot=bot)


async def test_send_message_calls_bot(adapter: TelegramMessagingAdapter, bot: MagicMock) -> None:
    await adapter.send_message(chat_id=12345, text="Hello!")

    bot.send_message.assert_awaited_once_with(chat_id=12345, text="Hello!")


def test_implements_messaging_port(adapter: TelegramMessagingAdapter) -> None:
    assert isinstance(adapter, MessagingPort)

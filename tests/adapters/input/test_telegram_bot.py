from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Update
from telegram.ext import ContextTypes

from kume.adapters.input.telegram_bot import TelegramBotAdapter
from kume.ports.output.messaging import MessagingPort
from kume.services.orchestrator import OrchestratorService


@pytest.fixture
def orchestrator() -> AsyncMock:
    return AsyncMock(spec=OrchestratorService)


@pytest.fixture
def messaging() -> AsyncMock:
    return AsyncMock(spec=MessagingPort)


@pytest.fixture
def adapter(orchestrator: AsyncMock, messaging: AsyncMock) -> TelegramBotAdapter:
    return TelegramBotAdapter(orchestrator=orchestrator, messaging=messaging)


async def test_handle_message_valid_text(
    adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.text = "What should I eat?"
    update.effective_user = MagicMock()
    update.effective_user.id = 12345
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    orchestrator.process.return_value = "Eat more vegetables!"

    await adapter.handle_message(update, context)

    orchestrator.process.assert_awaited_once_with(12345, "What should I eat?")
    messaging.send_message.assert_awaited_once_with(67890, "Eat more vegetables!")


async def test_handle_message_non_text(
    adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = MagicMock()
    update.message.text = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    orchestrator.process.assert_not_awaited()
    messaging.send_message.assert_awaited_once_with(
        67890,
        "I can only handle text messages for now.",
    )


async def test_handle_message_no_message(
    adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = 67890
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    orchestrator.process.assert_not_awaited()
    messaging.send_message.assert_awaited_once_with(
        67890,
        "I can only handle text messages for now.",
    )


async def test_handle_message_no_message_no_chat(
    adapter: TelegramBotAdapter,
    orchestrator: AsyncMock,
    messaging: AsyncMock,
) -> None:
    update = MagicMock(spec=Update)
    update.message = None
    update.effective_chat = None
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    await adapter.handle_message(update, context)

    orchestrator.process.assert_not_awaited()
    messaging.send_message.assert_not_awaited()

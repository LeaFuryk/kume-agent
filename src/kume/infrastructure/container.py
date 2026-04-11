from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from telegram.ext import Application, MessageHandler, filters

from kume.adapters.input.telegram_bot import TelegramBotAdapter
from kume.adapters.output.telegram_messaging import TelegramMessagingAdapter
from kume.adapters.tools import (
    AnalyzeFoodTool,
    AskRecommendationTool,
    IngestContextTool,
    LogMealTool,
    RequestReportTool,
)
from kume.infrastructure.config import Settings
from kume.infrastructure.metrics import MetricsCollector
from kume.services.orchestrator import OrchestratorService


class Container:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def orchestrator_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self._settings.orchestrator_model,
            api_key=SecretStr(self._settings.openai_api_key),
        )

    def tool_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self._settings.tool_model,
            api_key=SecretStr(self._settings.openai_api_key),
        )

    def tools(self) -> list:
        tool_llm = self.tool_llm()
        return [
            AskRecommendationTool(llm=tool_llm),
            AnalyzeFoodTool(llm=tool_llm),
            IngestContextTool(),
            LogMealTool(),
            RequestReportTool(),
        ]

    def metrics_collector(self) -> MetricsCollector:
        return MetricsCollector()

    def orchestrator_service(self) -> OrchestratorService:
        return OrchestratorService(
            llm=self.orchestrator_llm(),
            tools=self.tools(),
            metrics_collector=self.metrics_collector(),
        )

    def telegram_application(self) -> Application:
        app = Application.builder().token(self._settings.telegram_token).build()
        orchestrator = self.orchestrator_service()
        messaging = TelegramMessagingAdapter(bot=app.bot)
        bot_adapter = TelegramBotAdapter(orchestrator=orchestrator, messaging=messaging)
        app.add_handler(MessageHandler(filters.ALL, bot_adapter.handle_message))
        return app

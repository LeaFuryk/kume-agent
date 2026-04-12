from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from telegram.ext import Application, MessageHandler, filters

from kume.adapters.input.telegram_bot import TelegramBotAdapter
from kume.adapters.output.langchain_llm import LangChainLLMAdapter
from kume.adapters.output.postgres_db import (
    PostgresDocumentRepository,
    PostgresGoalRepository,
    PostgresLabMarkerRepository,
    PostgresRestrictionRepository,
    PostgresUserRepository,
    create_session_factory,
)
from kume.adapters.output.telegram_messaging import TelegramMessagingAdapter
from kume.adapters.tools import (
    AnalyzeFoodTool,
    AskRecommendationTool,
    LogMealTool,
    RequestReportTool,
    SaveGoalTool,
    SaveHealthContextTool,
    SaveLabReportTool,
    SaveRestrictionTool,
)
from kume.infrastructure.config import Settings
from kume.ports.output.llm import LLMPort
from kume.ports.output.repositories import (
    DocumentRepository,
    EmbeddingRepository,
    GoalRepository,
    LabMarkerRepository,
    RestrictionRepository,
    UserRepository,
)
from kume.services.orchestrator import OrchestratorService


class _StubEmbeddingRepository(EmbeddingRepository):
    """No-op placeholder until a real embedding adapter is wired up."""

    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None:
        pass

    async def search(self, user_id: str, query: str, k: int = 5) -> list[str]:
        return []


class Container:
    """Dependency injection container that builds the full application graph from Settings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session_factory = create_session_factory(settings.database_url)

    # --- Repositories ---

    def user_repo(self) -> UserRepository:
        return PostgresUserRepository(self._session_factory)

    def goal_repo(self) -> GoalRepository:
        return PostgresGoalRepository(self._session_factory)

    def restriction_repo(self) -> RestrictionRepository:
        return PostgresRestrictionRepository(self._session_factory)

    def doc_repo(self) -> DocumentRepository:
        return PostgresDocumentRepository(self._session_factory)

    def marker_repo(self) -> LabMarkerRepository:
        return PostgresLabMarkerRepository(self._session_factory)

    def embedding_repo(self) -> EmbeddingRepository:
        return _StubEmbeddingRepository()

    # --- LLM ---

    def orchestrator_llm(self) -> ChatOpenAI:
        """Raw LangChain model for the orchestrator agent (needs BaseChatModel for create_agent)."""
        return ChatOpenAI(
            model=self._settings.orchestrator_model,
            api_key=SecretStr(self._settings.openai_api_key),
        )

    def tool_llm(self) -> LLMPort:
        """LLM port for tools — abstracts the provider."""
        model = ChatOpenAI(
            model=self._settings.tool_model,
            api_key=SecretStr(self._settings.openai_api_key),
        )
        return LangChainLLMAdapter(model)

    # --- Tools ---

    def tools(self) -> list[BaseTool]:
        tool_llm = self.tool_llm()

        return [
            AskRecommendationTool(llm=tool_llm),
            AnalyzeFoodTool(llm=tool_llm),
            LogMealTool(),
            RequestReportTool(),
            SaveGoalTool(goal_repo=self.goal_repo()),
            SaveRestrictionTool(restriction_repo=self.restriction_repo()),
            SaveHealthContextTool(doc_repo=self.doc_repo(), embedding_repo=self.embedding_repo()),
            SaveLabReportTool(
                llm=tool_llm,
                doc_repo=self.doc_repo(),
                marker_repo=self.marker_repo(),
                embedding_repo=self.embedding_repo(),
            ),
        ]

    def orchestrator_service(self) -> OrchestratorService:
        return OrchestratorService(
            llm=self.orchestrator_llm(),
            tools=self.tools(),
            max_iterations=self._settings.max_agent_iterations,
        )

    def telegram_application(self) -> Application:
        app = Application.builder().token(self._settings.telegram_token).build()
        orchestrator = self.orchestrator_service()
        messaging = TelegramMessagingAdapter(bot=app.bot)
        bot_adapter = TelegramBotAdapter(orchestrator=orchestrator, messaging=messaging)
        app.add_handler(MessageHandler(filters.ALL, bot_adapter.handle_message))
        return app

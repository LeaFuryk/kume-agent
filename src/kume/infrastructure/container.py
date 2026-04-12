from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from telegram.ext import Application, MessageHandler, filters

from kume.adapters.input.message_batcher import MessageBatcher
from kume.adapters.input.telegram_bot import TelegramBotAdapter
from kume.adapters.output.audio_processor import AudioProcessor
from kume.adapters.output.image_processor import ImageProcessor
from kume.adapters.output.langchain_llm import LangChainLLMAdapter
from kume.adapters.output.pdf_processor import PDFProcessor
from kume.adapters.output.pgvector_embedding import PGVectorEmbeddingRepository
from kume.adapters.output.postgres_db import (
    PostgresDocumentRepository,
    PostgresGoalRepository,
    PostgresLabMarkerRepository,
    PostgresRestrictionRepository,
    PostgresUserRepository,
    create_session_factory,
)
from kume.adapters.output.telegram_messaging import TelegramMessagingAdapter
from kume.adapters.output.whisper_stt import WhisperAdapter
from kume.adapters.tools import (
    AnalyzeFoodTool,
    AskRecommendationTool,
    FetchContextTool,
    LogMealTool,
    ProcessLabReportTool,
    RequestReportTool,
    SaveGoalTool,
    SaveHealthContextTool,
    SaveRestrictionTool,
    SaveUserNameTool,
)
from kume.domain.context import ContextBuilder, ContextDataProvider
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
from kume.services.ingestion import IngestionService
from kume.services.orchestrator import OrchestratorService


class _RepositoryContextDataProvider(ContextDataProvider):
    """ContextDataProvider backed by repository adapters."""

    def __init__(
        self,
        goal_repo: GoalRepository,
        restriction_repo: RestrictionRepository,
        marker_repo: LabMarkerRepository,
        embedding_repo: EmbeddingRepository,
    ) -> None:
        self._goal_repo = goal_repo
        self._restriction_repo = restriction_repo
        self._marker_repo = marker_repo
        self._embedding_repo = embedding_repo

    async def get_goals(self, user_id: str) -> list[Any]:
        return await self._goal_repo.get_by_user(user_id)

    async def get_restrictions(self, user_id: str) -> list[Any]:
        return await self._restriction_repo.get_by_user(user_id)

    async def get_lab_markers(self, user_id: str) -> list[Any]:
        return await self._marker_repo.get_by_user(user_id)

    async def search_documents(self, user_id: str, query: str) -> list[str]:
        return await self._embedding_repo.search(user_id, query)


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
        return PGVectorEmbeddingRepository(
            database_url=self._settings.database_url,
            openai_api_key=self._settings.openai_api_key,
            embedding_model=self._settings.openai_embedding_model,
        )

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

    # --- Resource processing ---

    def whisper_adapter(self) -> WhisperAdapter:
        return WhisperAdapter(api_key=self._settings.openai_api_key)

    def ingestion_service(self) -> IngestionService:
        whisper = self.whisper_adapter()
        processors = {
            "application/pdf": PDFProcessor(),
            "audio/ogg": AudioProcessor(stt=whisper),
            "audio/mpeg": AudioProcessor(stt=whisper),
            "audio/mp4": AudioProcessor(stt=whisper),
            "image/jpeg": ImageProcessor(),
            "image/png": ImageProcessor(),
        }
        return IngestionService(processors=processors)

    # --- Context ---

    def context_builder(self) -> ContextBuilder:
        provider = _RepositoryContextDataProvider(
            goal_repo=self.goal_repo(),
            restriction_repo=self.restriction_repo(),
            marker_repo=self.marker_repo(),
            embedding_repo=self.embedding_repo(),
        )
        return ContextBuilder(provider=provider)

    # --- Tools ---

    def tools(self) -> list[BaseTool]:
        tool_llm = self.tool_llm()
        cb = self.context_builder()

        return [
            AskRecommendationTool(llm=tool_llm, context_builder=cb),
            AnalyzeFoodTool(llm=tool_llm, context_builder=cb),
            LogMealTool(),
            RequestReportTool(),
            SaveGoalTool(goal_repo=self.goal_repo()),
            SaveRestrictionTool(restriction_repo=self.restriction_repo()),
            SaveHealthContextTool(doc_repo=self.doc_repo(), embedding_repo=self.embedding_repo()),
            ProcessLabReportTool(
                llm=tool_llm,
                doc_repo=self.doc_repo(),
                marker_repo=self.marker_repo(),
                embedding_repo=self.embedding_repo(),
            ),
            SaveUserNameTool(user_repo=self.user_repo()),
            FetchContextTool(context_builder=cb),
        ]

    def orchestrator_service(self) -> OrchestratorService:
        return OrchestratorService(
            llm=self.orchestrator_llm(),
            tools=self.tools(),
            max_iterations=self._settings.max_agent_iterations,
            user_repo=self.user_repo(),
        )

    def telegram_application(self) -> Application:
        app = Application.builder().token(self._settings.telegram_token).build()
        orchestrator = self.orchestrator_service()
        messaging = TelegramMessagingAdapter(bot=app.bot)
        ingestion = self.ingestion_service()

        # Create the bot adapter first (without batcher), then wire the batcher
        # pointing back to the adapter's _process_batch method.
        bot_adapter = TelegramBotAdapter(
            orchestrator=orchestrator,
            messaging=messaging,
            ingestion=ingestion,
        )
        batcher = MessageBatcher(
            debounce_seconds=2.0,
            on_batch_ready=bot_adapter._process_batch,
        )
        bot_adapter._batcher = batcher

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_adapter.handle_message))
        app.add_handler(
            MessageHandler(
                filters.Document.ALL | filters.VOICE | filters.AUDIO | filters.PHOTO,
                bot_adapter.handle_media,
            )
        )
        return app

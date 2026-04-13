from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI
from telegram.ext import MessageHandler

from kume.adapters.output.langchain_llm import LangChainLLMAdapter
from kume.adapters.output.openai_vision import OpenAIVisionAdapter
from kume.adapters.output.whisper_stt import WhisperAdapter
from kume.adapters.tools.analyze_food import AnalyzeFoodTool
from kume.adapters.tools.ask_recommendation import AskRecommendationTool
from kume.domain.context import ContextBuilder
from kume.infrastructure.config import Settings
from kume.infrastructure.container import Container
from kume.ports.output.llm import LLMPort
from kume.ports.output.repositories import EmbeddingRepository
from kume.services.ingestion import IngestionService


@pytest.fixture
def settings() -> Settings:
    return Settings(
        telegram_token="fake-token",
        openai_api_key="fake-key",
        orchestrator_model="gpt-4o",
        tool_model="gpt-4o-mini",
        vision_model="gpt-4o",
        max_agent_iterations=5,
        log_level="INFO",
        database_url="postgresql+asyncpg://kume:kume@localhost:5432/kume",
        openai_embedding_model="text-embedding-3-small",
        log_format="pretty",
    )


class _FakeEmbeddingRepository(EmbeddingRepository):
    """Test-only stub so container tests don't need a real PGVector / Postgres."""

    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None:
        pass

    async def search(self, user_id: str, query: str, k: int = 5) -> list[str]:
        return []


@pytest.fixture
def container(settings: Settings) -> Container:
    """Container with PGVector mocked out so tests don't need a real database."""
    with patch(
        "kume.infrastructure.container.PGVectorEmbeddingRepository",
        return_value=_FakeEmbeddingRepository(),
    ):
        yield Container(settings)


def test_container_instantiation(settings: Settings) -> None:
    container = Container(settings)
    assert container._settings is settings


def test_orchestrator_llm_returns_chat_openai(container: Container) -> None:
    llm = container.orchestrator_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o"


def test_tool_llm_returns_llm_port(container: Container) -> None:
    llm = container.tool_llm()
    assert isinstance(llm, LLMPort)
    assert isinstance(llm, LangChainLLMAdapter)


def test_tools_returns_twelve_tools(container: Container) -> None:
    tools = container.tools()
    assert isinstance(tools, list)
    assert len(tools) == 12
    names = {t.name for t in tools}
    assert names == {
        "ask_recommendation",
        "analyze_food",
        "analyze_food_image",
        "fetch_user_context",
        "fetch_lab_results",
        "log_meal",
        "request_report",
        "save_goal",
        "save_restriction",
        "save_health_context",
        "process_lab_report",
        "save_user_name",
    }


@patch("kume.services.orchestrator.create_agent")
def test_orchestrator_service_passes_max_iterations(mock_create_agent, container: Container) -> None:
    from kume.services.orchestrator import OrchestratorService

    service = container.orchestrator_service()
    assert isinstance(service, OrchestratorService)
    assert service._max_iterations == 5
    mock_create_agent.assert_called_once()


def test_telegram_application_method_exists(container: Container) -> None:
    assert hasattr(container, "telegram_application")
    assert callable(container.telegram_application)


def test_repository_methods_exist(container: Container) -> None:
    assert callable(container.user_repo)
    assert callable(container.goal_repo)
    assert callable(container.restriction_repo)
    assert callable(container.doc_repo)
    assert callable(container.marker_repo)


# --- New tests for P2.13 ---


def test_ingestion_service_returns_ingestion_service(container: Container) -> None:
    service = container.ingestion_service()
    assert isinstance(service, IngestionService)
    # Verify all expected mime types are registered
    expected_mimes = {
        "application/pdf",
        "audio/ogg",
        "audio/mpeg",
        "audio/mp4",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
    assert set(service._processors.keys()) == expected_mimes


def test_context_builder_returns_context_builder(container: Container) -> None:
    cb = container.context_builder()
    assert isinstance(cb, ContextBuilder)
    assert cb._provider is not None


@patch("kume.adapters.output.whisper_stt.AsyncOpenAI")
def test_whisper_adapter_returns_whisper_adapter(mock_openai, container: Container) -> None:
    adapter = container.whisper_adapter()
    assert isinstance(adapter, WhisperAdapter)
    mock_openai.assert_called_once_with(api_key="fake-key", max_retries=3)


def test_tools_include_context_builder_in_ask_recommendation_and_analyze_food(container: Container) -> None:
    tools = container.tools()
    ask_rec = next(t for t in tools if t.name == "ask_recommendation")
    analyze = next(t for t in tools if t.name == "analyze_food")

    assert isinstance(ask_rec, AskRecommendationTool)
    assert isinstance(analyze, AnalyzeFoodTool)
    assert ask_rec.context_builder is not None
    assert isinstance(ask_rec.context_builder, ContextBuilder)
    assert analyze.context_builder is not None
    assert isinstance(analyze.context_builder, ContextBuilder)


@patch("kume.services.orchestrator.create_agent")
def test_telegram_application_registers_media_handler(mock_create_agent, container: Container) -> None:
    app = container.telegram_application()
    handlers = app.handlers[0]  # default group 0
    assert len(handlers) == 2

    # First handler: text messages (excluding commands)
    text_handler = handlers[0]
    assert isinstance(text_handler, MessageHandler)

    # Second handler: media (documents, voice, audio, photo)
    media_handler = handlers[1]
    assert isinstance(media_handler, MessageHandler)


# --- New tests for P3.13 ---


def test_session_store_is_singleton(container: Container) -> None:
    assert container.session_store() is container.session_store()


def test_image_store_is_singleton(container: Container) -> None:
    assert container.image_store() is container.image_store()


@patch("kume.adapters.output.openai_vision.AsyncOpenAI")
def test_vision_port_created(mock_openai, container: Container) -> None:
    vision = container.vision_port()
    assert isinstance(vision, OpenAIVisionAdapter)


def test_analyze_food_image_tool_in_tools(container: Container) -> None:
    tools = container.tools()
    names = {t.name for t in tools}
    assert "analyze_food_image" in names


# --- Repo singleton tests ---


def test_user_repo_is_singleton(container: Container) -> None:
    assert container.user_repo() is container.user_repo()


def test_goal_repo_is_singleton(container: Container) -> None:
    assert container.goal_repo() is container.goal_repo()


def test_restriction_repo_is_singleton(container: Container) -> None:
    assert container.restriction_repo() is container.restriction_repo()


def test_doc_repo_is_singleton(container: Container) -> None:
    assert container.doc_repo() is container.doc_repo()


def test_marker_repo_is_singleton(container: Container) -> None:
    assert container.marker_repo() is container.marker_repo()


def test_meal_repo_is_singleton(container: Container) -> None:
    assert container.meal_repo() is container.meal_repo()


def test_embedding_repo_is_singleton(container: Container) -> None:
    assert container.embedding_repo() is container.embedding_repo()

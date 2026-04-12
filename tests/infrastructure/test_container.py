from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI

from kume.adapters.output.langchain_llm import LangChainLLMAdapter
from kume.infrastructure.config import Settings
from kume.infrastructure.container import Container
from kume.ports.output.llm import LLMPort


@pytest.fixture
def settings() -> Settings:
    return Settings(
        telegram_token="fake-token",
        openai_api_key="fake-key",
        orchestrator_model="gpt-4o",
        tool_model="gpt-4o-mini",
        max_agent_iterations=5,
        log_level="INFO",
        database_url="postgresql+asyncpg://kume:kume@localhost:5432/kume",
        openai_embedding_model="text-embedding-3-small",
        log_format="pretty",
    )


@pytest.fixture
def container(settings: Settings) -> Container:
    return Container(settings)


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


def test_tools_returns_eight_tools(container: Container) -> None:
    tools = container.tools()
    assert isinstance(tools, list)
    assert len(tools) == 8
    names = {t.name for t in tools}
    assert names == {
        "ask_recommendation",
        "analyze_food",
        "log_meal",
        "request_report",
        "save_goal",
        "save_restriction",
        "save_health_context",
        "save_lab_report",
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

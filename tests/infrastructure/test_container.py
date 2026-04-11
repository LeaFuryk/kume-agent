from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI

from kume.infrastructure.config import Settings
from kume.infrastructure.container import Container
from kume.infrastructure.metrics import MetricsCollector


@pytest.fixture
def settings() -> Settings:
    return Settings(
        telegram_token="fake-token",
        openai_api_key="fake-key",
        orchestrator_model="gpt-4o",
        tool_model="gpt-4o-mini",
        max_agent_iterations=5,
        log_level="INFO",
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


def test_tool_llm_returns_chat_openai(container: Container) -> None:
    llm = container.tool_llm()
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"


def test_tools_returns_five_tools(container: Container) -> None:
    tools = container.tools()
    assert isinstance(tools, list)
    assert len(tools) == 5
    names = {t.name for t in tools}
    assert names == {
        "ask_recommendation",
        "analyze_food",
        "ingest_context",
        "log_meal",
        "request_report",
    }


def test_metrics_collector_returns_instance(container: Container) -> None:
    collector = container.metrics_collector()
    assert isinstance(collector, MetricsCollector)


@patch("kume.services.orchestrator.create_agent")
def test_orchestrator_service_returns_instance(mock_create_agent, container: Container) -> None:
    from kume.services.orchestrator import OrchestratorService

    service = container.orchestrator_service()
    assert isinstance(service, OrchestratorService)
    mock_create_agent.assert_called_once()


def test_telegram_application_method_exists(container: Container) -> None:
    assert hasattr(container, "telegram_application")
    assert callable(container.telegram_application)

from __future__ import annotations

import os

import pytest


# Auto-skip eval tests when no API key is available
def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    if not os.environ.get("OPENAI_API_KEY"):
        skip_marker = pytest.mark.skip(reason="OPENAI_API_KEY not set")
        for item in items:
            if "eval" in item.keywords:
                item.add_marker(skip_marker)


@pytest.fixture
def eval_orchestrator():
    """Build a real orchestrator with ChatOpenAI + stub repos for evals.

    Only usable when OPENAI_API_KEY is set (eval-marked tests auto-skip otherwise).
    """
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    from kume.adapters.tools import (
        AnalyzeFoodTool,
        AskRecommendationTool,
        FetchContextTool,
        LogMealTool,
        RequestReportTool,
        SaveGoalTool,
        SaveHealthContextTool,
        SaveRestrictionTool,
        SaveUserNameTool,
    )
    from kume.adapters.tools.fetch_lab_results import FetchLabResultsTool
    from kume.domain.context import ContextBuilder
    from kume.services.orchestrator import OrchestratorService

    # Import fakes from the existing test conftest
    from tests.adapters.tools.conftest import (
        FakeDocumentRepository,
        FakeEmbeddingRepository,
        FakeGoalRepository,
        FakeLabMarkerRepository,
        FakeLLMPort,
        FakeMealRepository,
        FakeRestrictionRepository,
        FakeUserRepository,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=SecretStr(api_key), max_retries=3)
    tool_llm = FakeLLMPort("stub response")

    # Stub context builder (no real DB)
    from unittest.mock import AsyncMock

    from kume.domain.context import ContextDataProvider

    provider = AsyncMock(spec=ContextDataProvider)
    provider.get_goals.return_value = []
    provider.get_restrictions.return_value = []
    provider.get_lab_markers.return_value = []
    provider.search_documents.return_value = []
    provider.get_recent_meals.return_value = []
    cb = ContextBuilder(provider=provider)

    tools = [
        AskRecommendationTool(llm=tool_llm, context_builder=cb),
        AnalyzeFoodTool(llm=tool_llm, context_builder=cb),
        LogMealTool(meal_repo=FakeMealRepository()),
        RequestReportTool(),
        SaveGoalTool(goal_repo=FakeGoalRepository()),
        SaveRestrictionTool(restriction_repo=FakeRestrictionRepository()),
        SaveHealthContextTool(doc_repo=FakeDocumentRepository(), embedding_repo=FakeEmbeddingRepository()),
        SaveUserNameTool(user_repo=FakeUserRepository()),
        FetchContextTool(context_builder=cb),
        FetchLabResultsTool(marker_repo=FakeLabMarkerRepository()),
    ]

    return OrchestratorService(
        llm=llm,
        tools=tools,
        max_iterations=5,
        user_repo=FakeUserRepository(),
    )


@pytest.fixture
def eval_llm():
    """Provide a raw ChatOpenAI for LLM-as-judge calls."""
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    api_key = os.environ.get("OPENAI_API_KEY", "")
    return ChatOpenAI(model="gpt-4o-mini", api_key=SecretStr(api_key), max_retries=3)

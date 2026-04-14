from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# Env-configurable model for A/B testing: EVAL_MODEL=gpt-5-mini uv run pytest tests/evals/ -m eval -v
EVAL_MODEL = os.environ.get("EVAL_MODEL", "gpt-4o-mini")


# Auto-skip eval tests when no API key is available
def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    if not os.environ.get("OPENAI_API_KEY"):
        skip_marker = pytest.mark.skip(reason="OPENAI_API_KEY not set")
        for item in items:
            if "eval" in item.keywords:
                item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Cost tracker — collects costs across all eval tests, prints summary at end
# ---------------------------------------------------------------------------


class _CostTracker:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []
        self.model: str = ""

    def record(self, test_name: str, cost_usd: float, input_tokens: int, output_tokens: int, model: str) -> None:
        if model:
            self.model = model
        self.entries.append(
            {
                "test": test_name,
                "cost": cost_usd,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    @property
    def total_cost(self) -> float:
        return sum(float(e["cost"]) for e in self.entries)

    @property
    def total_input_tokens(self) -> int:
        return sum(int(e["input_tokens"]) for e in self.entries)

    @property
    def total_output_tokens(self) -> int:
        return sum(int(e["output_tokens"]) for e in self.entries)


_tracker = _CostTracker()


@pytest.fixture
def cost_tracker() -> _CostTracker:
    return _tracker


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # type: ignore[no-untyped-def]
    """Print eval cost summary at the end of the test run."""
    if not _tracker.entries:
        return

    terminalreporter.section("Eval Cost Summary")
    terminalreporter.write_line(f"Model: {_tracker.model or EVAL_MODEL}")
    terminalreporter.write_line(f"Total tests: {len(_tracker.entries)}")
    terminalreporter.write_line(f"Total input tokens: {_tracker.total_input_tokens:,}")
    terminalreporter.write_line(f"Total output tokens: {_tracker.total_output_tokens:,}")
    terminalreporter.write_line(f"Total cost: ${_tracker.total_cost:.4f}")
    terminalreporter.write_line("")

    # Per-test breakdown
    terminalreporter.write_line(f"{'Test':<60} {'Tokens':>12} {'Cost':>10}")
    terminalreporter.write_line("-" * 84)
    for entry in _tracker.entries:
        name = str(entry["test"])
        if len(name) > 58:
            name = name[:55] + "..."
        tokens = f"{entry['input_tokens']}+{entry['output_tokens']}"
        terminalreporter.write_line(f"{name:<60} {tokens:>12} ${float(entry['cost']):.4f}")
    terminalreporter.write_line("-" * 84)
    terminalreporter.write_line(
        f"{'TOTAL':<60} {_tracker.total_input_tokens}+{_tracker.total_output_tokens:>5} ${_tracker.total_cost:.4f}"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_orchestrator():
    """Build a real orchestrator with ChatOpenAI + stub repos for evals.

    Model is configurable via EVAL_MODEL env var (default: gpt-4o-mini).
    """
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    from kume.adapters.tools import (
        AnalyzeFoodImageTool,
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
    from kume.infrastructure.image_store import ImageStore
    from kume.services.orchestrator import OrchestratorService
    from tests.adapters.tools.conftest import (
        FakeDocumentRepository,
        FakeEmbeddingRepository,
        FakeGoalRepository,
        FakeLabMarkerRepository,
        FakeLLMPort,
        FakeMealRepository,
        FakeRestrictionRepository,
        FakeUserRepository,
        FakeVisionPort,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    llm = ChatOpenAI(model=EVAL_MODEL, api_key=SecretStr(api_key), max_retries=3)
    tool_llm = FakeLLMPort("stub response")

    from unittest.mock import AsyncMock

    from kume.domain.context import ContextDataProvider

    provider = AsyncMock(spec=ContextDataProvider)
    provider.get_goals.return_value = []
    provider.get_restrictions.return_value = []
    provider.get_lab_markers.return_value = []
    provider.search_documents.return_value = []
    provider.get_recent_meals.return_value = []
    cb = ContextBuilder(provider=provider)

    image_store = ImageStore()

    tools = [
        AskRecommendationTool(llm=tool_llm, context_builder=cb),
        AnalyzeFoodTool(llm=tool_llm, context_builder=cb),
        AnalyzeFoodImageTool(vision=FakeVisionPort(), context_builder=cb, image_store=image_store),
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
        image_store=image_store,
    )


@pytest.fixture
def eval_llm():
    """Provide a raw ChatOpenAI for LLM-as-judge calls."""
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    api_key = os.environ.get("OPENAI_API_KEY", "")
    return ChatOpenAI(model=EVAL_MODEL, api_key=SecretStr(api_key), max_retries=3)

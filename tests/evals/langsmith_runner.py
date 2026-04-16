"""Run Kume evals via LangSmith — results visible in the dashboard with pass/fail.

Usage:
    # Upload datasets + run evals (first time)
    uv run python tests/evals/langsmith_runner.py

    # Run with a different model
    EVAL_MODEL=gpt-5-mini uv run python tests/evals/langsmith_runner.py

    # Only upload datasets (no eval run)
    uv run python tests/evals/langsmith_runner.py --upload-only

Results: https://smith.langchain.com → project "kume"
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.evals.helpers import load_cases, load_quality_cases  # noqa: E402

EVAL_MODEL = os.environ.get("EVAL_MODEL", "gpt-4o-mini")
CASES_DIR = Path(__file__).parent / "cases"


def _get_client() -> Client:
    return Client()


# ---------------------------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------------------------


def upload_datasets(client: Client) -> None:
    """Create or update LangSmith datasets from YAML eval cases."""

    # Tool selection dataset
    _upload_tool_selection(client)
    _upload_intent_classification(client)
    _upload_response_quality(client)


def _upload_tool_selection(client: Client) -> None:
    dataset_name = "kume-tool-selection"
    cases = load_cases(CASES_DIR / "tool_selection.yaml")

    # Delete existing to re-upload fresh
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        client.delete_dataset(dataset_id=existing.id)
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Kume tool selection evals — verifies LLM picks the correct tool",
    )
    for case in cases:
        client.create_example(
            dataset_id=dataset.id,
            inputs={"user_message": case.input, "user_prefix": case.user_prefix},
            outputs={
                "expected_tools": case.expected_tools,
                "forbidden_tools": case.forbidden_tools,
            },
            metadata={"case_id": case.id, "description": case.description},
        )
    print(f"  Uploaded {len(cases)} cases to '{dataset_name}'")


def _upload_intent_classification(client: Client) -> None:
    dataset_name = "kume-intent-classification"
    cases = load_cases(CASES_DIR / "intent_classification.yaml")

    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        client.delete_dataset(dataset_id=existing.id)
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Kume intent classification evals — log vs analyze decisions",
    )
    for case in cases:
        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "user_message": case.input,
                "user_prefix": case.user_prefix,
                "has_image": case.has_image,
            },
            outputs={
                "expected_tools": case.expected_tools,
                "forbidden_tools": case.forbidden_tools,
            },
            metadata={"case_id": case.id, "description": case.description},
        )
    print(f"  Uploaded {len(cases)} cases to '{dataset_name}'")


def _upload_response_quality(client: Client) -> None:
    dataset_name = "kume-response-quality"
    cases = load_quality_cases(CASES_DIR / "response_quality.yaml")

    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        client.delete_dataset(dataset_id=existing.id)
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Kume response quality evals — LLM-as-judge scoring",
    )
    for case in cases:
        client.create_example(
            dataset_id=dataset.id,
            inputs={"user_message": case.input, "user_prefix": case.user_prefix},
            outputs={
                "criteria": case.criteria,
                "expected_language": case.expected_language,
            },
            metadata={"case_id": case.id, "description": case.description},
        )
    print(f"  Uploaded {len(cases)} cases to '{dataset_name}'")


# ---------------------------------------------------------------------------
# Build orchestrator (same as conftest but standalone)
# ---------------------------------------------------------------------------


def _build_orchestrator():  # type: ignore[no-untyped-def]
    from unittest.mock import AsyncMock

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
    from kume.domain.context import ContextBuilder, ContextDataProvider
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


EVAL_TIMEOUT = 30  # seconds per case — prevents stuck agent loops


def _run_with_timeout(coro):  # type: ignore[no-untyped-def]
    """Run an async coroutine with a timeout. Returns the result or raises TimeoutError."""

    async def _wrapped():  # type: ignore[no-untyped-def]
        return await asyncio.wait_for(coro, timeout=EVAL_TIMEOUT)

    return asyncio.run(_wrapped())


# ---------------------------------------------------------------------------
# Run evals via langsmith.evaluate()
# ---------------------------------------------------------------------------


def run_tool_selection_eval(client: Client) -> None:
    """Run tool selection eval — results appear in LangSmith dashboard."""
    from langsmith import evaluate

    from tests.evals.helpers import run_eval

    orchestrator = _build_orchestrator()

    def target(inputs: dict) -> dict:
        try:
            result = _run_with_timeout(
                run_eval(
                    orchestrator,
                    user_message=inputs["user_message"],
                    user_prefix=inputs.get("user_prefix", ""),
                )
            )
            return {
                "tool_calls": result.tool_calls,
                "response": result.response_text,
                "cost_usd": result.cost_usd,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        except TimeoutError:
            return {"tool_calls": [], "response": "[TIMEOUT]", "cost_usd": 0, "input_tokens": 0, "output_tokens": 0}

    def tool_selection_correct(run, example) -> dict:
        actual = set(run.outputs.get("tool_calls", []))
        expected = set(example.outputs.get("expected_tools", []))
        forbidden = set(example.outputs.get("forbidden_tools", []))

        # At least one expected tool called (or none expected and none called)
        if expected:
            has_expected = bool(actual & expected)
        else:
            has_expected = True  # no tool expected

        # No forbidden tools called
        has_forbidden = bool(actual & forbidden)

        return {"key": "tool_correct", "score": 1.0 if (has_expected and not has_forbidden) else 0.0}

    print(f"\nRunning tool selection eval with {EVAL_MODEL}...")
    evaluate(
        target,
        data="kume-tool-selection",
        evaluators=[tool_selection_correct],
        experiment_prefix=f"tool-selection-{EVAL_MODEL}",
        client=client,
        max_concurrency=2,
    )


def run_intent_classification_eval(client: Client) -> None:
    """Run intent classification eval."""
    from langsmith import evaluate

    from kume.services.orchestrator import Resource
    from tests.evals.helpers import run_eval

    orchestrator = _build_orchestrator()

    def target(inputs: dict) -> dict:
        resources = None
        if inputs.get("has_image"):
            resources = [
                Resource(
                    mime_type="image/jpeg",
                    transcript="[Image attached for analysis]",
                    raw_bytes=b"fake-image",
                )
            ]
        try:
            result = _run_with_timeout(
                run_eval(
                    orchestrator,
                    user_message=inputs["user_message"],
                    user_prefix=inputs.get("user_prefix", ""),
                    resources=resources,
                )
            )
            return {
                "tool_calls": result.tool_calls,
                "response": result.response_text,
                "cost_usd": result.cost_usd,
            }
        except TimeoutError:
            return {"tool_calls": [], "response": "[TIMEOUT]", "cost_usd": 0}

    def intent_correct(run, example) -> dict:
        actual = set(run.outputs.get("tool_calls", []))
        expected = set(example.outputs.get("expected_tools", []))
        forbidden = set(example.outputs.get("forbidden_tools", []))

        # At least one expected tool called (same logic as pytest evals)
        has_expected = bool(actual & expected) if expected else True
        has_forbidden = bool(actual & forbidden)

        return {"key": "intent_correct", "score": 1.0 if (has_expected and not has_forbidden) else 0.0}

    print(f"\nRunning intent classification eval with {EVAL_MODEL}...")
    evaluate(
        target,
        data="kume-intent-classification",
        evaluators=[intent_correct],
        experiment_prefix=f"intent-{EVAL_MODEL}",
        client=client,
        max_concurrency=2,
    )


def run_response_quality_eval(client: Client) -> None:
    """Run response quality eval with LLM-as-judge."""
    from langsmith import evaluate

    from tests.evals.helpers import judge_response, run_eval

    orchestrator = _build_orchestrator()

    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    judge_llm = ChatOpenAI(
        model=EVAL_MODEL,
        api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "")),
        max_retries=3,
    )

    def target(inputs: dict) -> dict:
        try:
            result = _run_with_timeout(
                run_eval(
                    orchestrator,
                    user_message=inputs["user_message"],
                    user_prefix=inputs.get("user_prefix", ""),
                )
            )
            return {
                "response": result.response_text,
                "cost_usd": result.cost_usd,
            }
        except TimeoutError:
            return {"response": "[TIMEOUT]", "cost_usd": 0}

    def quality_score(run, example) -> list[dict]:
        criteria = example.outputs.get("criteria", [])
        expected_lang = example.outputs.get("expected_language", "en")
        response = run.outputs.get("response", "")
        user_msg = example.inputs.get("user_message", "")

        scores = asyncio.run(
            judge_response(
                llm=judge_llm,
                user_message=user_msg,
                response_text=response,
                criteria=criteria,
                expected_language=expected_lang,
            )
        )

        return [{"key": f"quality_{k}", "score": v / 5.0} for k, v in scores.items()]

    print(f"\nRunning response quality eval with {EVAL_MODEL}...")
    evaluate(
        target,
        data="kume-response-quality",
        evaluators=[quality_score],
        experiment_prefix=f"quality-{EVAL_MODEL}",
        client=client,
        max_concurrency=2,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    client = _get_client()

    upload_only = "--upload-only" in sys.argv

    print("Uploading eval datasets to LangSmith...")
    upload_datasets(client)
    print("Done.\n")

    if upload_only:
        print("Datasets uploaded. Skipping eval runs (--upload-only).")
        return

    print(f"Running evals with model: {EVAL_MODEL}")
    print("=" * 60)

    run_tool_selection_eval(client)
    run_intent_classification_eval(client)
    run_response_quality_eval(client)

    print("\n" + "=" * 60)
    print("All evals complete. View results at: https://smith.langchain.com")


if __name__ == "__main__":
    main()

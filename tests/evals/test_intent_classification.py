from pathlib import Path

import pytest

from kume.services.orchestrator import Resource
from tests.evals.helpers import EvalResult, load_cases, run_eval

CASES = load_cases(Path(__file__).parent / "cases" / "intent_classification.yaml")


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case_well_formed(case):
    """Verify cases are well-formed (no API needed)."""
    assert case.id and case.input
    assert isinstance(case.expected_tools, list)
    assert isinstance(case.forbidden_tools, list)
    overlap = set(case.expected_tools) & set(case.forbidden_tools)
    assert not overlap, f"Case {case.id}: overlap {overlap}"
    assert len(case.expected_tools) > 0, f"Case {case.id}: must have expected tools"


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_intent_classification_llm(case, eval_orchestrator, cost_tracker):
    """Run each case through the real LLM and verify intent decisions."""
    resources = None
    if case.has_image:
        resources = [
            Resource(
                mime_type="image/jpeg",
                transcript="[Image attached for analysis]",
                raw_bytes=b"fake-image",
            )
        ]

    result: EvalResult = await run_eval(
        eval_orchestrator,
        user_message=case.input,
        user_prefix=case.user_prefix,
        resources=resources,
    )

    cost_tracker.record(
        f"intent/{case.id}",
        result.cost_usd,
        result.input_tokens,
        result.output_tokens,
        result.model,
    )

    for tool in case.expected_tools:
        assert tool in result.tool_calls, f"[{case.id}] Expected tool '{tool}' not called. Actual: {result.tool_calls}"

    for tool in case.forbidden_tools:
        assert tool not in result.tool_calls, (
            f"[{case.id}] Forbidden tool '{tool}' was called. Actual: {result.tool_calls}"
        )

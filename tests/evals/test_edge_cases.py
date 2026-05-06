from pathlib import Path

import pytest

from tests.evals.helpers import EvalResult, load_cases, run_eval

CASES = load_cases(Path(__file__).parent / "cases" / "edge_cases.yaml")


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_edge_cases_well_formed(case):
    """Verify edge cases are well-formed (no API needed)."""
    assert case.id and case.input
    assert isinstance(case.expected_tools, list)
    assert isinstance(case.forbidden_tools, list)


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_edge_case_llm(case, eval_orchestrator, cost_tracker):
    """Run edge cases through the real LLM."""
    result: EvalResult = await run_eval(
        eval_orchestrator,
        user_message=case.input,
        user_prefix=case.user_prefix,
    )

    cost_tracker.record(
        f"edge/{case.id}",
        result.cost_usd,
        result.input_tokens,
        result.output_tokens,
        result.model,
    )

    if case.expected_tools:
        called = set(result.tool_calls)
        expected = set(case.expected_tools)
        assert called & expected, (
            f"[{case.id}] None of the expected tools {case.expected_tools} were called. Actual: {result.tool_calls}"
        )

    for tool in case.forbidden_tools:
        assert tool not in result.tool_calls, (
            f"[{case.id}] Forbidden tool '{tool}' was called. Actual: {result.tool_calls}"
        )

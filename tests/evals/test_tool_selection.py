from pathlib import Path

import pytest

from tests.evals.helpers import EvalResult, load_cases, run_eval

CASES = load_cases(Path(__file__).parent / "cases" / "tool_selection.yaml")


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_tool_selection_cases_well_formed(case):
    """Verify all tool selection cases are well-formed (no API needed)."""
    assert case.id, "Case must have an id"
    assert case.input, "Case must have input"
    assert isinstance(case.expected_tools, list)
    assert isinstance(case.forbidden_tools, list)
    overlap = set(case.expected_tools) & set(case.forbidden_tools)
    assert not overlap, f"Tools in both expected and forbidden: {overlap}"


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_tool_selection_llm(case, eval_orchestrator, cost_tracker):
    """Run each case through the real LLM and verify tool selection."""
    result: EvalResult = await run_eval(
        eval_orchestrator,
        user_message=case.input,
        user_prefix=case.user_prefix,
    )

    cost_tracker.record(
        f"tool_selection/{case.id}",
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

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
async def test_tool_selection_llm(case, eval_orchestrator):
    """Run each case through the real LLM and verify tool selection.

    Pass criteria: expected tools are called, forbidden tools are NOT called.
    Target: >= 80% pass rate (LLMs are non-deterministic).
    """
    result: EvalResult = await run_eval(
        eval_orchestrator,
        user_message=case.input,
        user_prefix=case.user_prefix,
    )

    # Check expected tools were called
    for tool in case.expected_tools:
        assert tool in result.tool_calls, f"[{case.id}] Expected tool '{tool}' not called. Actual: {result.tool_calls}"

    # Check forbidden tools were NOT called
    for tool in case.forbidden_tools:
        assert tool not in result.tool_calls, (
            f"[{case.id}] Forbidden tool '{tool}' was called. Actual: {result.tool_calls}"
        )

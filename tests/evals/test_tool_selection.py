import pytest
from pathlib import Path
from tests.evals.helpers import load_cases

CASES = load_cases(Path(__file__).parent / "cases" / "tool_selection.yaml")


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_tool_selection_cases_loaded(case):
    """Verify all tool selection cases are well-formed."""
    assert case.id, "Case must have an id"
    assert case.input, "Case must have input"
    assert isinstance(case.expected_tools, list)
    assert isinstance(case.forbidden_tools, list)
    # Verify expected and forbidden don't overlap
    overlap = set(case.expected_tools) & set(case.forbidden_tools)
    assert not overlap, f"Tools in both expected and forbidden: {overlap}"

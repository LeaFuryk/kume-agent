from pathlib import Path

import pytest

from tests.evals.helpers import load_cases

CASES = load_cases(Path(__file__).parent / "cases" / "intent_classification.yaml")


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case_loads(case):
    """Verify cases load correctly and are well-formed."""
    assert case.id, "Case must have a non-empty id"
    assert isinstance(case.input, str) and case.input, "Case must have non-empty input"
    assert isinstance(case.expected_tools, list), "expected_tools must be a list"
    assert isinstance(case.forbidden_tools, list), "forbidden_tools must be a list"


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_no_overlap_expected_forbidden(case):
    """Forbidden tools must never overlap with expected tools."""
    overlap = set(case.expected_tools) & set(case.forbidden_tools)
    assert not overlap, f"Case {case.id}: tools appear in both expected and forbidden: {overlap}"


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_has_expected_tools(case):
    """Every case must declare at least one expected tool."""
    assert len(case.expected_tools) > 0, f"Case {case.id}: must have at least one expected tool"


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_image_cases_use_image_tool(case):
    """Cases with has_image=True should expect an image-aware tool."""
    if case.has_image:
        image_tools = [t for t in case.expected_tools if "image" in t]
        assert image_tools, f"Case {case.id}: has_image=True but no image tool in expected_tools"

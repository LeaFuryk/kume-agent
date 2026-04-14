from pathlib import Path

import pytest

from tests.evals.helpers import load_quality_cases

CASES = load_quality_cases(Path(__file__).parent / "cases" / "response_quality.yaml")


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_quality_cases_loaded(case):
    """Verify quality cases load correctly and have required fields."""
    assert case.id
    assert case.input
    assert len(case.criteria) > 0, "Quality cases must have at least one criterion"
    assert case.expected_language

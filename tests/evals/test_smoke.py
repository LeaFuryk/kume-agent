from pathlib import Path

import pytest

from tests.evals.helpers import load_cases

CASES = load_cases(Path(__file__).parent / "cases" / "smoke.yaml")


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_smoke_case_loads(case):
    """Verify the YAML loader works — just check cases are loaded correctly."""
    assert case.id
    assert isinstance(case.input, str)
    assert isinstance(case.expected_tools, list)

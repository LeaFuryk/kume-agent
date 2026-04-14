from pathlib import Path

import pytest

from tests.evals.helpers import judge_response, load_quality_cases, run_eval

CASES = load_quality_cases(Path(__file__).parent / "cases" / "response_quality.yaml")


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_quality_cases_well_formed(case):
    """Verify quality cases are well-formed (no API needed)."""
    assert case.id
    assert case.input
    assert len(case.criteria) > 0, "Quality cases must have at least one criterion"
    assert case.expected_language


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_response_quality_llm(case, eval_orchestrator, eval_llm):
    """Run each case through the real LLM, then score with LLM-as-judge.

    Each criterion must score >= 3 out of 5.
    Cost: ~$0.01 per case (orchestrator call + judge call).
    """
    result = await run_eval(
        eval_orchestrator,
        user_message=case.input,
        user_prefix=case.user_prefix,
    )

    scores = await judge_response(
        llm=eval_llm,
        user_message=case.input,
        response_text=result.response_text,
        criteria=case.criteria,
        expected_language=case.expected_language,
    )

    # Report scores for debugging
    score_report = " | ".join(f"{k}: {v}/5" for k, v in scores.items())
    print(f"\n  [{case.id}] Scores: {score_report}")

    for criterion in case.criteria:
        score = scores.get(criterion, 0)
        assert score >= 3, (
            f"[{case.id}] Criterion '{criterion}' scored {score}/5 (minimum 3). Response: {result.response_text[:200]}"
        )

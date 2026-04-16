from pathlib import Path

import pytest

from tests.evals.helpers import judge_response, load_quality_cases, run_eval

CASES = load_quality_cases(Path(__file__).parent / "cases" / "language_handling.yaml")


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_language_cases_well_formed(case):
    """Verify language cases are well-formed (no API needed)."""
    assert case.id and case.input
    assert case.expected_language
    assert len(case.criteria) > 0


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
async def test_language_handling_llm(case, eval_orchestrator, eval_llm, cost_tracker):
    """Run language cases through the real LLM and score with judge."""
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

    cost_tracker.record(
        f"language/{case.id}",
        result.cost_usd,
        result.input_tokens,
        result.output_tokens,
        result.model,
    )

    score_report = " | ".join(f"{k}: {v}/5" for k, v in scores.items())
    print(f"\n  [{case.id}] Scores: {score_report}")

    for criterion in case.criteria:
        score = scores.get(criterion, 0)
        assert score >= 3, (
            f"[{case.id}] Criterion '{criterion}' scored {score}/5 (minimum 3). Response: {result.response_text[:200]}"
        )

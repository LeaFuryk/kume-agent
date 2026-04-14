lfuryk@192 kume-agent % uv run pytest tests/evals/ -m eval -v
=========================================== test session starts ===========================================
platform darwin -- Python 3.12.12, pytest-9.0.3, pluggy-1.6.0 -- /Users/lfuryk/Documents/kume-agent/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/lfuryk/Documents/kume-agent
configfile: pyproject.toml
plugins: cov-7.1.0, asyncio-1.3.0, langsmith-0.7.30, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 64 items / 31 deselected / 33 selected

tests/evals/test_intent_classification.py::test_intent_classification_llm[analyze_only_healthy] PASSED [  3%]
tests/evals/test_intent_classification.py::test_intent_classification_llm[analyze_only_whats_in] PASSED [  6%]
tests/evals/test_intent_classification.py::test_intent_classification_llm[analyze_only_can_i_eat] PASSED [  9%]
tests/evals/test_intent_classification.py::test_intent_classification_llm[analyze_only_calories] PASSED [ 12%]
tests/evals/test_intent_classification.py::test_intent_classification_llm[analyze_only_text] FAILED [ 15%]
tests/evals/test_intent_classification.py::test_intent_classification_llm[log_just_ate] PASSED      [ 18%]
tests/evals/test_intent_classification.py::test_intent_classification_llm[log_dinner] PASSED        [ 21%]
tests/evals/test_intent_classification.py::test_intent_classification_llm[log_text_meal] FAILED     [ 24%]
tests/evals/test_intent_classification.py::test_intent_classification_llm[log_text_eggs] PASSED     [ 27%]
tests/evals/test_intent_classification.py::test_intent_classification_llm[log_text_with_time] FAILED [ 30%]
tests/evals/test_response_quality.py::test_response_quality_llm[spanish_response] PASSED            [ 33%]
tests/evals/test_response_quality.py::test_response_quality_llm[returning_user_no_intro] PASSED     [ 36%]
tests/evals/test_response_quality.py::test_response_quality_llm[first_time_user_intro] PASSED       [ 39%]
tests/evals/test_response_quality.py::test_response_quality_llm[concise_nutrition_answer] PASSED    [ 42%]
tests/evals/test_response_quality.py::test_response_quality_llm[goal_saving_encouragement] PASSED   [ 45%]
tests/evals/test_response_quality.py::test_response_quality_llm[closure_followup] PASSED            [ 48%]
tests/evals/test_smoke.py::test_smoke_case_loads[greeting_no_tool] PASSED                           [ 51%]
tests/evals/test_smoke.py::test_smoke_case_loads[nutrition_question_uses_tool] PASSED               [ 54%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[save_goal_explicit] PASSED              [ 57%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[save_goal_vague] PASSED                 [ 60%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[save_restriction_allergy] PASSED        [ 63%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[save_restriction_diet] PASSED           [ 66%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[save_health_context] PASSED             [ 69%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[fetch_lab_specific] PASSED              [ 72%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[fetch_context_broad] PASSED             [ 75%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[ask_recommendation] PASSED              [ 78%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[analyze_food_text] PASSED               [ 81%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[log_meal_text] PASSED                                     [ 84%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[greeting_no_tool] PASSED                                  [ 87%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[thanks_no_tool] PASSED                                    [ 90%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[off_topic_no_tool] PASSED                                 [ 93%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[multi_goal_restriction] PASSED                            [ 96%]
tests/evals/test_tool_selection.py::test_tool_selection_llm[nutrition_advice_specific] PASSED                         [100%]

========================================================= FAILURES ==========================================================
_____________________________________ test_intent_classification_llm[analyze_only_text] _____________________________________

case = EvalCase(id='analyze_only_text', description='Text-only food question', input='Is sushi good for lowering cholesterol?', expected_tools=['analyze_food'], forbidden_tools=['log_meal'], has_image=False, user_prefix='')
eval_orchestrator = <kume.services.orchestrator.OrchestratorService object at 0x10c572f90>
cost_tracker = <tests.evals.conftest._CostTracker object at 0x10a3afd10>

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
>           assert tool in result.tool_calls, f"[{case.id}] Expected tool '{tool}' not called. Actual: {result.tool_calls}"
E           AssertionError: [analyze_only_text] Expected tool 'analyze_food' not called. Actual: ['fetch_user_context']
E           assert 'analyze_food' in ['fetch_user_context']
E            +  where ['fetch_user_context'] = EvalResult(tool_calls=['fetch_user_context'], response_text='Sushi can be a great option for a healthy diet! 🍣 It often includes fish, which can help lower cholesterol levels. \n\nTo give you the best advice, could you share any specific health goals or dietary restrictions you have?', input_tokens=2826, output_tokens=64, cost_usd=0.0004623, model='gpt-4o-mini-2024-07-18').tool_calls

tests/evals/test_intent_classification.py:52: AssertionError
_______________________________________ test_intent_classification_llm[log_text_meal] _______________________________________

case = EvalCase(id='log_text_meal', description='User describes meal without image', input='I had a grilled chicken salad for lunch', expected_tools=['log_meal'], forbidden_tools=['analyze_food_image'], has_image=False, user_prefix='')
eval_orchestrator = <kume.services.orchestrator.OrchestratorService object at 0x10c645a30>
cost_tracker = <tests.evals.conftest._CostTracker object at 0x10a3afd10>

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
>           assert tool in result.tool_calls, f"[{case.id}] Expected tool '{tool}' not called. Actual: {result.tool_calls}"
E           AssertionError: [log_text_meal] Expected tool 'log_meal' not called. Actual: ['analyze_food_image']
E           assert 'log_meal' in ['analyze_food_image']
E            +  where ['analyze_food_image'] = EvalResult(tool_calls=['analyze_food_image'], response_text='It seems I need a photo of your grilled chicken salad to analyze it! 📸 \n\nCould you share an image, or would you like me to help with a general nutritional breakdown? 😊', input_tokens=2824, output_tokens=63, cost_usd=0.00046139999999999994, model='gpt-4o-mini-2024-07-18').tool_calls

tests/evals/test_intent_classification.py:52: AssertionError
____________________________________ test_intent_classification_llm[log_text_with_time] _____________________________________

case = EvalCase(id='log_text_with_time', description='Meal with time reference', input='I had pizza at noon', expected_tools=['log_meal'], forbidden_tools=[], has_image=False, user_prefix='')
eval_orchestrator = <kume.services.orchestrator.OrchestratorService object at 0x10c689160>
cost_tracker = <tests.evals.conftest._CostTracker object at 0x10a3afd10>

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
>           assert tool in result.tool_calls, f"[{case.id}] Expected tool '{tool}' not called. Actual: {result.tool_calls}"
E           AssertionError: [log_text_with_time] Expected tool 'log_meal' not called. Actual: ['analyze_food_image']
E           assert 'log_meal' in ['analyze_food_image']
E            +  where ['analyze_food_image'] = EvalResult(tool_calls=['analyze_food_image'], response_text='It looks like I need a photo of the pizza to analyze its nutritional content. 📸 If you want, just upload the image, and I can take a look!', input_tokens=2818, output_tokens=59, cost_usd=0.00045809999999999997, model='gpt-4o-mini-2024-07-18').tool_calls

tests/evals/test_intent_classification.py:52: AssertionError
===================================================== Eval Cost Summary =====================================================
Model: gpt-4o-mini-2024-07-18
Total tests: 31
Total input tokens: 82,119
Total output tokens: 1,971
Total cost: $0.0135

Test                                                               Tokens       Cost
------------------------------------------------------------------------------------
intent/analyze_only_healthy                                       2851+44 $0.0005
intent/analyze_only_whats_in                                      2852+50 $0.0005
intent/analyze_only_can_i_eat                                     2863+58 $0.0005
intent/analyze_only_calories                                      2857+49 $0.0005
intent/analyze_only_text                                          2826+64 $0.0005
intent/log_just_ate                                               2859+57 $0.0005
intent/log_dinner                                                 2849+50 $0.0005
intent/log_text_meal                                              2824+63 $0.0005
intent/log_text_eggs                                             4332+135 $0.0007
intent/log_text_with_time                                         2818+59 $0.0005
quality/spanish_response                                         2823+100 $0.0005
quality/returning_user_no_intro                                   2817+91 $0.0005
quality/first_time_user_intro                                     1384+34 $0.0002
quality/concise_nutrition_answer                                 2825+155 $0.0005
quality/goal_saving_encouragement                                 2820+59 $0.0005
quality/closure_followup                                          1384+23 $0.0002
tool_selection/save_goal_explicit                                 2810+51 $0.0005
tool_selection/save_goal_vague                                    2816+40 $0.0004
tool_selection/save_restriction_allergy                           2814+55 $0.0005
tool_selection/save_restriction_diet                              2816+47 $0.0005
tool_selection/save_health_context                                2871+89 $0.0005
tool_selection/fetch_lab_specific                                 2816+48 $0.0005
tool_selection/fetch_context_broad                                2839+57 $0.0005
tool_selection/ask_recommendation                                 2839+64 $0.0005
tool_selection/analyze_food_text                                  2812+53 $0.0005
tool_selection/log_meal_text                                      2863+83 $0.0005
tool_selection/greeting_no_tool                                   1384+31 $0.0002
tool_selection/thanks_no_tool                                     1387+26 $0.0002
tool_selection/off_topic_no_tool                                  1388+28 $0.0002
tool_selection/multi_goal_restriction                             2866+83 $0.0005
tool_selection/nutrition_advice_specific                         2814+125 $0.0005
------------------------------------------------------------------------------------
TOTAL                                                        82119+ 1971 $0.0135
================================================== short test summary info ==================================================
FAILED tests/evals/test_intent_classification.py::test_intent_classification_llm[analyze_only_text] - AssertionError: [analyze_only_text] Expected tool 'analyze_food' not called. Actual: ['fetch_user_context']
FAILED tests/evals/test_intent_classification.py::test_intent_classification_llm[log_text_meal] - AssertionError: [log_text_meal] Expected tool 'log_meal' not called. Actual: ['analyze_food_image']
FAILED tests/evals/test_intent_classification.py::test_intent_classification_llm[log_text_with_time] - AssertionError: [log_text_with_time] Expected tool 'log_meal' not called. Actual: ['analyze_food_image']
================================== 3 failed, 30 passed, 31 deselected in 106.17s (0:01:46) ==================================
lfuryk@192 kume-agent %
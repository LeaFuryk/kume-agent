from kume.domain.tools.ask_recommendation import ask_recommendation


def test_ask_recommendation_calls_llm_with_prompt_containing_query() -> None:
    captured_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Eat more vegetables."

    query = "What should I eat for breakfast?"
    result = ask_recommendation(query, fake_llm)

    assert len(captured_prompts) == 1
    assert query in captured_prompts[0]
    assert result == "Eat more vegetables."


def test_ask_recommendation_returns_llm_response() -> None:
    expected = "Try oatmeal with berries for a balanced breakfast."

    result = ask_recommendation("breakfast ideas", lambda _: expected)

    assert result == expected


def test_ask_recommendation_includes_context_in_prompt() -> None:
    captured_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Personalized advice."

    context = "## User Goals\n- Lose 5kg"
    query = "What should I eat?"
    result = ask_recommendation(query, fake_llm, context=context)

    assert len(captured_prompts) == 1
    assert context in captured_prompts[0]
    assert query in captured_prompts[0]
    assert result == "Personalized advice."


def test_ask_recommendation_works_without_context() -> None:
    captured_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Generic advice."

    query = "What should I eat?"
    result = ask_recommendation(query, fake_llm)

    assert len(captured_prompts) == 1
    assert query in captured_prompts[0]
    assert "nutrition expert" in captured_prompts[0].lower()
    assert result == "Generic advice."


def test_ask_recommendation_empty_context_is_graceful() -> None:
    captured_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Advice."

    query = "Breakfast ideas?"
    result = ask_recommendation(query, fake_llm, context="")

    assert len(captured_prompts) == 1
    assert query in captured_prompts[0]
    assert result == "Advice."

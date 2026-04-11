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

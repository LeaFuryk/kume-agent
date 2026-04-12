from kume.domain.tools.analyze_food import analyze_food


def test_analyze_food_calls_llm_with_prompt_containing_description() -> None:
    captured_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "This food is high in protein."

    description = "grilled chicken breast with steamed broccoli"
    result = analyze_food(description, fake_llm)

    assert len(captured_prompts) == 1
    assert description in captured_prompts[0]
    assert result == "This food is high in protein."


def test_analyze_food_returns_llm_response() -> None:
    expected = "High in healthy fats and fiber."

    result = analyze_food("avocado toast", lambda _: expected)

    assert result == expected


def test_analyze_food_includes_context_in_prompt() -> None:
    captured_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Personalized analysis."

    context = "## Dietary Restrictions\n- [allergy] Peanuts"
    description = "pad thai with peanut sauce"
    result = analyze_food(description, fake_llm, context=context)

    assert len(captured_prompts) == 1
    assert context in captured_prompts[0]
    assert description in captured_prompts[0]
    assert result == "Personalized analysis."


def test_analyze_food_works_without_context() -> None:
    captured_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Generic analysis."

    description = "Caesar salad"
    result = analyze_food(description, fake_llm)

    assert len(captured_prompts) == 1
    assert description in captured_prompts[0]
    assert "nutrition expert" in captured_prompts[0].lower()
    assert result == "Generic analysis."


def test_analyze_food_empty_context_is_graceful() -> None:
    captured_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Analysis."

    description = "A banana"
    result = analyze_food(description, fake_llm, context="")

    assert len(captured_prompts) == 1
    assert description in captured_prompts[0]
    assert result == "Analysis."

from kume.services.prompts import AGENT_SYSTEM_PROMPT, FORMATTER_PROMPT, SYSTEM_PROMPT


def test_agent_prompt_has_tool_rules() -> None:
    assert "ALWAYS use tools" in AGENT_SYSTEM_PROMPT
    assert "Log vs Analyze" in AGENT_SYSTEM_PROMPT


def test_agent_prompt_has_behavioral_rules() -> None:
    assert "Anticipatory Messages" in AGENT_SYSTEM_PROMPT
    assert "First Interaction" in AGENT_SYSTEM_PROMPT


def test_agent_prompt_no_formatting_instructions() -> None:
    agent_lower = AGENT_SYSTEM_PROMPT.lower()
    assert "emoji" not in agent_lower
    assert "bullet lists" not in agent_lower
    assert "{language}" not in AGENT_SYSTEM_PROMPT
    assert "{user_name}" not in AGENT_SYSTEM_PROMPT
    assert "3-5 short lines" not in agent_lower
    assert "aligned numbers" not in agent_lower


def test_formatter_prompt_has_formatting_rules() -> None:
    assert "emoji" in FORMATTER_PROMPT.lower()
    assert "{language}" in FORMATTER_PROMPT
    assert "{user_name}" in FORMATTER_PROMPT


def test_formatter_prompt_no_tool_rules() -> None:
    assert "fetch_user_context" not in FORMATTER_PROMPT
    assert "analyze_food_image" not in FORMATTER_PROMPT


def test_system_prompt_alias_backward_compat() -> None:
    """SYSTEM_PROMPT alias must exist and equal AGENT_SYSTEM_PROMPT for backward compat."""
    assert SYSTEM_PROMPT is AGENT_SYSTEM_PROMPT

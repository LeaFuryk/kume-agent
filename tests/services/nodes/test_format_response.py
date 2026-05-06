"""Tests for the format_response graph node."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from kume.services.nodes.format_response import format_response


class TestFormatResponse:
    """format_response node reformats agent output for Telegram UX.

    After the graph reorder (Fix 1), format_response extracts the raw
    agent response from the last AIMessage in the conversation, then
    sets both ``raw_agent_response`` and ``formatted_response``.
    """

    def test_returns_formatted_text(self) -> None:
        """The node returns formatted text from the LLM."""
        state = {
            "messages": [
                HumanMessage(content="What should I eat?"),
                AIMessage(content="You should eat more vegetables and lean proteins."),
            ],
            "user_name": "Ana",
            "user_language": "en",
        }

        with patch(
            "kume.services.nodes.format_response._call_formatter_llm",
            return_value="Hey Ana! Here's what I'd suggest:\n- More veggies\n- Lean proteins",
        ):
            result = format_response(state)

        assert result["formatted_response"] == "Hey Ana! Here's what I'd suggest:\n- More veggies\n- Lean proteins"
        assert result["raw_agent_response"] == "You should eat more vegetables and lean proteins."

    def test_passes_user_context_to_llm(self) -> None:
        """The node passes user_name and language to the formatter LLM."""
        state = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content="Eat balanced meals."),
            ],
            "user_name": "Carlos",
            "user_language": "es",
        }

        with patch(
            "kume.services.nodes.format_response._call_formatter_llm",
            return_value="Hola Carlos!",
        ) as mock_llm:
            format_response(state)

        mock_llm.assert_called_once_with("Eat balanced meals.", "Carlos", "es")

    def test_empty_messages_returns_fallback(self) -> None:
        """When there are no AIMessages, return a fallback without calling LLM."""
        state = {
            "messages": [HumanMessage(content="Hello")],
            "user_name": "Ana",
            "user_language": "en",
        }

        with patch(
            "kume.services.nodes.format_response._call_formatter_llm",
        ) as mock_llm:
            result = format_response(state)

        mock_llm.assert_not_called()
        assert result["formatted_response"] == "How can I help with your nutrition goals today?"

    def test_exception_returns_raw_as_fallback(self) -> None:
        """When the LLM raises an exception, return the raw response."""
        state = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content="Eat more fiber."),
            ],
            "user_name": None,
            "user_language": "en",
        }

        with patch(
            "kume.services.nodes.format_response._call_formatter_llm",
            side_effect=Exception("LLM connection failed"),
        ):
            result = format_response(state)

        assert result["formatted_response"] == "Eat more fiber."
        assert result["raw_agent_response"] == "Eat more fiber."

    def test_extracts_from_structured_content(self) -> None:
        """The node uses _extract_text_content for structured AIMessage content."""
        state = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content=[{"type": "text", "text": "Structured response."}]),
            ],
            "user_name": None,
            "user_language": "en",
        }

        with patch(
            "kume.services.nodes.format_response._call_formatter_llm",
            return_value="Formatted structured response.",
        ):
            result = format_response(state)

        assert result["raw_agent_response"] == "Structured response."
        assert result["formatted_response"] == "Formatted structured response."

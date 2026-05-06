"""Tests for the output_guardrail graph node."""

from __future__ import annotations

import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from kume.services.nodes.output_guardrail import output_guardrail


class TestOutputGuardrail:
    """output_guardrail node checks agent responses for safety issues.

    After the graph reorder (Fix 1), the output guardrail validates
    ``formatted_response`` (the final user-facing text) rather than the
    raw AIMessage directly.
    """

    def test_safe_response_passes(self) -> None:
        """A safe formatted response is marked output_safe."""
        safe_response = json.dumps({"safe": True, "category": None, "reason": "Normal nutrition advice"})
        state = {
            "formatted_response": "I recommend a balanced diet with proteins and vegetables.",
            "raw_agent_response": "I recommend a balanced diet with proteins and vegetables.",
            "messages": [
                HumanMessage(content="What should I eat?"),
                AIMessage(content="I recommend a balanced diet with proteins and vegetables."),
            ],
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value=safe_response,
        ):
            result = output_guardrail(state)

        assert result["output_safe"] is True
        assert result["guardrail_violation"] is None

    def test_dangerous_medical_advice_blocked(self) -> None:
        """Dangerous medical advice in formatted output is flagged."""
        unsafe_response = json.dumps(
            {
                "safe": False,
                "category": "dangerous_medical_advice",
                "reason": "Prescribes specific medication dosages",
            }
        )
        state = {
            "formatted_response": "Take 80mg of atorvastatin daily without consulting your doctor.",
            "messages": [
                HumanMessage(content="How to lower cholesterol?"),
                AIMessage(content="Take 80mg of atorvastatin daily without consulting your doctor."),
            ],
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value=unsafe_response,
        ):
            result = output_guardrail(state)

        assert result["output_safe"] is False
        assert result["guardrail_violation"] == "dangerous_medical_advice"

    def test_eating_disorder_trigger_blocked(self) -> None:
        """Content that could trigger eating disorders is flagged."""
        unsafe_response = json.dumps(
            {
                "safe": False,
                "category": "eating_disorder_trigger",
                "reason": "Promotes extreme caloric restriction",
            }
        )
        state = {
            "formatted_response": "Eat only 200 calories a day for rapid weight loss.",
            "messages": [
                HumanMessage(content="How can I lose weight fast?"),
                AIMessage(content="Eat only 200 calories a day for rapid weight loss."),
            ],
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value=unsafe_response,
        ):
            result = output_guardrail(state)

        assert result["output_safe"] is False
        assert result["guardrail_violation"] == "eating_disorder_trigger"

    def test_prefers_formatted_response(self) -> None:
        """The node validates formatted_response (not the raw AIMessage)."""
        safe_response = json.dumps({"safe": True, "category": None, "reason": "OK"})
        state = {
            "formatted_response": "Formatted: Second answer",
            "raw_agent_response": "Second answer",
            "messages": [
                HumanMessage(content="First question"),
                AIMessage(content="First answer"),
                HumanMessage(content="Second question"),
                AIMessage(content="Second answer"),
            ],
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value=safe_response,
        ) as mock_llm:
            output_guardrail(state)

        # Should check the formatted_response, not the raw AIMessage
        mock_llm.assert_called_once_with("Formatted: Second answer")

    def test_falls_back_to_ai_message_when_no_formatted_response(self) -> None:
        """When formatted_response is empty, falls back to the last AIMessage."""
        safe_response = json.dumps({"safe": True, "category": None, "reason": "OK"})
        state = {
            "formatted_response": "",
            "raw_agent_response": "",
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content="Raw answer"),
            ],
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value=safe_response,
        ) as mock_llm:
            output_guardrail(state)

        mock_llm.assert_called_once_with("Raw answer")

    def test_malformed_json_fails_closed(self) -> None:
        """When the LLM returns unparseable JSON, fail closed (block as precaution)."""
        state = {
            "formatted_response": "Take vitamin D.",
            "messages": [
                HumanMessage(content="What vitamins?"),
                AIMessage(content="Take vitamin D."),
            ],
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value="not json {{{",
        ):
            result = output_guardrail(state)

        assert result["output_safe"] is False
        assert result["guardrail_violation"] == "guardrail_error"

    def test_non_object_json_fails_closed(self) -> None:
        """Valid JSON that isn't an object (e.g. array) should fail closed."""
        state = {
            "formatted_response": "Eat more vegetables.",
            "messages": [
                HumanMessage(content="What should I eat?"),
                AIMessage(content="Eat more vegetables."),
            ],
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value='["not", "an", "object"]',
        ):
            result = output_guardrail(state)

        assert result["output_safe"] is False
        assert result["guardrail_violation"] == "guardrail_error"

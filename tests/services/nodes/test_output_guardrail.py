"""Tests for the output_guardrail graph node."""

from __future__ import annotations

import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from kume.services.nodes.output_guardrail import output_guardrail


class TestOutputGuardrail:
    """output_guardrail node checks agent responses for safety issues."""

    def test_safe_response_passes(self) -> None:
        """A safe agent response is marked output_safe."""
        safe_response = json.dumps({"safe": True, "category": None, "reason": "Normal nutrition advice"})
        state = {
            "messages": [
                HumanMessage(content="What should I eat?"),
                AIMessage(content="I recommend a balanced diet with proteins and vegetables."),
            ]
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value=safe_response,
        ):
            result = output_guardrail(state)

        assert result["output_safe"] is True
        assert result["guardrail_violation"] is None
        assert result["raw_agent_response"] == "I recommend a balanced diet with proteins and vegetables."

    def test_dangerous_medical_advice_blocked(self) -> None:
        """Dangerous medical advice in agent output is flagged."""
        unsafe_response = json.dumps(
            {
                "safe": False,
                "category": "dangerous_medical_advice",
                "reason": "Prescribes specific medication dosages",
            }
        )
        state = {
            "messages": [
                HumanMessage(content="How to lower cholesterol?"),
                AIMessage(content="Take 80mg of atorvastatin daily without consulting your doctor."),
            ]
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
            "messages": [
                HumanMessage(content="How can I lose weight fast?"),
                AIMessage(content="Eat only 200 calories a day for rapid weight loss."),
            ]
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value=unsafe_response,
        ):
            result = output_guardrail(state)

        assert result["output_safe"] is False
        assert result["guardrail_violation"] == "eating_disorder_trigger"

    def test_extracts_last_ai_message(self) -> None:
        """The node reads the last AIMessage from the conversation."""
        safe_response = json.dumps({"safe": True, "category": None, "reason": "OK"})
        state = {
            "messages": [
                HumanMessage(content="First question"),
                AIMessage(content="First answer"),
                HumanMessage(content="Second question"),
                AIMessage(content="Second answer"),
            ]
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value=safe_response,
        ) as mock_llm:
            result = output_guardrail(state)

        # Should extract the LAST AIMessage
        assert result["raw_agent_response"] == "Second answer"
        mock_llm.assert_called_once_with("Second answer")

    def test_malformed_json_defaults_to_safe(self) -> None:
        """When the LLM returns unparseable JSON, fail open (treat as safe)."""
        state = {
            "messages": [
                HumanMessage(content="What vitamins?"),
                AIMessage(content="Take vitamin D."),
            ]
        }

        with patch(
            "kume.services.nodes.output_guardrail._call_guardrail_llm",
            return_value="not json {{{",
        ):
            result = output_guardrail(state)

        assert result["output_safe"] is True
        assert result["guardrail_violation"] is None

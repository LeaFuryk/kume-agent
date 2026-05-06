"""Tests for the block_response graph node."""

from __future__ import annotations

from kume.services.nodes.block_response import block_response


class TestBlockResponse:
    """block_response node returns the right block message based on violation type."""

    def test_input_violation_returns_input_block_message(self) -> None:
        """When input_safe is False, the block message addresses the input."""
        state = {
            "input_safe": False,
            "guardrail_violation": "prompt_injection",
        }
        result = block_response(state)

        assert "formatted_response" in result
        assert isinstance(result["formatted_response"], str)
        assert len(result["formatted_response"]) > 0

    def test_output_violation_returns_output_block_message(self) -> None:
        """When input_safe is True (output violation), the block message addresses the output."""
        state = {
            "input_safe": True,
            "guardrail_violation": "dangerous_medical_advice",
        }
        result = block_response(state)

        assert "formatted_response" in result
        assert isinstance(result["formatted_response"], str)
        assert len(result["formatted_response"]) > 0
        # Output block message should differ from input block message
        input_state = {
            "input_safe": False,
            "guardrail_violation": "prompt_injection",
        }
        input_result = block_response(input_state)
        assert result["formatted_response"] != input_result["formatted_response"]

    def test_sets_formatted_response(self) -> None:
        """The node always sets formatted_response in the returned dict."""
        state = {
            "input_safe": False,
            "guardrail_violation": "data_extraction",
        }
        result = block_response(state)

        assert "formatted_response" in result
        assert result["formatted_response"]

"""Block response node — returns a safe fallback when guardrails fire."""

from __future__ import annotations

from typing import Any

INPUT_BLOCK_MESSAGE = "I'm sorry, but I can't process that request. Please rephrase your message and try again."

OUTPUT_BLOCK_MESSAGE = (
    "I generated a response that didn't meet our safety standards. "
    "Let me try a different approach — could you rephrase your question?"
)


def block_response(state: dict[str, Any]) -> dict[str, Any]:
    """Return a canned block message based on the guardrail violation type.

    If ``input_safe`` is ``False`` the violation came from the user's input;
    otherwise the violation was detected in the agent's output.
    """
    if not state.get("input_safe", True):
        message = INPUT_BLOCK_MESSAGE
    else:
        message = OUTPUT_BLOCK_MESSAGE

    return {"formatted_response": message}

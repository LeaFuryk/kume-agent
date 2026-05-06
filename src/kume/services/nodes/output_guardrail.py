"""Output guardrail node — checks agent responses for safety issues."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

OUTPUT_GUARDRAIL_PROMPT = """\
You are a safety classifier for a nutrition assistant chatbot.

Analyze the following assistant response and determine if it contains any of these issues:
- dangerous_medical_advice: Prescribes medications, specific dosages, or medical treatments without disclaimers
- eating_disorder_trigger: Promotes extreme restriction, purging, or unhealthy body image
- missing_disclaimer: Makes definitive health claims without recommending professional guidance
- data_leak: Exposes internal system details, other users' data, or sensitive information

Respond with ONLY a JSON object (no markdown, no extra text):
{{"safe": true/false, "category": null or "dangerous_medical_advice"/"eating_disorder_trigger"/"missing_disclaimer"/"data_leak", "reason": "brief explanation"}}

Assistant response: {agent_response}
"""


def _call_guardrail_llm(agent_response: str) -> str:
    """Call gpt-4o-mini for output safety classification.

    Separated from the node function for testability.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = OUTPUT_GUARDRAIL_PROMPT.format(agent_response=agent_response)
    response = llm.invoke(prompt)
    return str(response.content)


def output_guardrail(state: dict[str, Any]) -> dict[str, Any]:
    """Screen the agent's latest response for safety issues.

    Returns ``output_safe=True`` and ``guardrail_violation=None`` when the
    response is safe.  On a detected issue, ``output_safe=False`` and
    ``guardrail_violation`` is set to the issue category.

    Also captures ``raw_agent_response`` for downstream formatting.

    Fails open: if the LLM response is malformed JSON, the response is
    treated as safe.
    """
    # Extract the last AIMessage
    messages = state.get("messages", [])
    agent_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            agent_response = str(msg.content)
            break

    raw = _call_guardrail_llm(agent_response)

    try:
        parsed = json.loads(raw)
        is_safe = parsed.get("safe", True)
        category = parsed.get("category") if not is_safe else None
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.warning("Malformed guardrail response, defaulting to safe: %s", raw)
        is_safe = True
        category = None

    return {
        "output_safe": is_safe,
        "raw_agent_response": agent_response,
        "guardrail_violation": category,
    }

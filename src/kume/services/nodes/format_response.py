"""Format response node — rewrites agent output for Telegram UX."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from kume.services.orchestrator import _extract_text_content
from kume.services.prompts import FORMATTER_PROMPT

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "How can I help with your nutrition goals today?"


def _call_formatter_llm(raw: str, user_name: str | None, language: str) -> str:
    """Call gpt-4o-mini to reformat the agent response.

    Separated from the node function for testability.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = FORMATTER_PROMPT.format(
        language=language,
        user_name=user_name or "there",
    )
    full_prompt = f"{prompt}\n\n<agent_output>\n{raw}\n</agent_output>\n\nReformat the content above."
    response = llm.invoke(full_prompt)
    return str(response.content)


def format_response(state: dict[str, Any]) -> dict[str, Any]:
    """Reformat the raw agent response for Telegram delivery.

    Extracts ``raw_agent_response`` from the last AIMessage in the
    conversation (since this node now runs before the output guardrail).
    If no AIMessage is found, returns a generic fallback without calling
    the LLM.  On any exception, returns the raw response as-is.
    """
    # Extract raw_agent_response from the last AIMessage
    messages = state.get("messages", [])
    raw = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            raw = _extract_text_content(msg.content)
            break

    user_name = state.get("user_name")
    language = state.get("user_language", "en")

    if not raw:
        return {"raw_agent_response": "", "formatted_response": FALLBACK_MESSAGE}

    try:
        formatted = _call_formatter_llm(raw, user_name, language)
        return {"raw_agent_response": raw, "formatted_response": formatted}
    except Exception:
        logger.exception("Formatter LLM failed, returning raw response")
        return {"raw_agent_response": raw, "formatted_response": raw}

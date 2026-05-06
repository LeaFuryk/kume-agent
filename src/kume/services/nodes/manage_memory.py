"""Manage memory node — compresses conversation history when it grows too long."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """\
Summarize the following conversation history between a user and a nutrition assistant.
Capture the key topics discussed, any goals or restrictions mentioned, meals logged,
and important context that should be preserved for continuity.
Be concise but thorough — this summary will replace the older messages.

Conversation:
{conversation}
"""


def _call_summarize_llm(conversation_text: str) -> str:
    """Call gpt-4o-mini to summarize older conversation messages.

    Separated from the node function for testability.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = SUMMARIZE_PROMPT.format(conversation=conversation_text)
    response = llm.invoke(prompt)
    return str(response.content)


def _messages_to_text(messages: list[BaseMessage]) -> str:
    """Convert a list of messages to a readable text format for summarization."""
    lines = []
    for msg in messages:
        role = msg.type.capitalize()
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def manage_memory(state: dict[str, Any], threshold: int = 20) -> dict[str, Any]:
    """Compress conversation history when it exceeds the threshold.

    If the message count is at or below *threshold*, returns without
    modifying messages.  Otherwise, summarizes the older messages (all
    except the last 10) via an LLM call and returns a new message list:
    ``[SystemMessage(summary)] + last_10_messages``.
    """
    messages: list[BaseMessage] = state.get("messages", [])

    if len(messages) <= threshold:
        return {"memory_summarized": False}

    # Split: older messages to summarize, recent messages to keep
    older = messages[:-10]
    last_10 = messages[-10:]

    conversation_text = _messages_to_text(older)
    summary = _call_summarize_llm(conversation_text)

    summary_message = SystemMessage(content=f"Summary of earlier conversation:\n{summary}")

    return {
        "messages": [summary_message] + list(last_10),
        "memory_summarized": True,
    }

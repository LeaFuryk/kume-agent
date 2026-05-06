from __future__ import annotations

from langgraph.graph import MessagesState


class KumeGraphState(MessagesState):
    """State schema for the Kume LangGraph agent.

    Extends MessagesState (which provides `messages: list[BaseMessage]`)
    with fields for guardrails, formatting, and user context.
    """

    # User context — set by orchestrator before graph invocation
    user_id: str
    user_name: str | None
    user_language: str

    # Guardrail results
    input_safe: bool
    output_safe: bool
    guardrail_violation: str | None

    # Response pipeline
    raw_agent_response: str
    formatted_response: str

    # Memory management
    memory_summarized: bool

    # Error tracking
    tool_error_count: int

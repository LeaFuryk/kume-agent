from __future__ import annotations

from typing import Any

from kume.infrastructure.request_context import RequestContext, get_context, set_context


async def set_request_context(state: dict[str, Any]) -> dict[str, Any]:
    """Bootstrap node that sets the RequestContext contextvar from graph state.

    Only sets context if none exists yet — the Telegram orchestrator sets it
    before invoking the graph, so this node is a no-op in that path. For direct
    graph invocation (LangGraph Platform), this provides the context that tools need.
    """
    existing = get_context()
    if existing is not None:
        return {}

    user_id = state.get("user_id", "")
    language = state.get("user_language", "en")

    if user_id:
        set_context(RequestContext(user_id=user_id, telegram_id=0, language=language))

    return {}

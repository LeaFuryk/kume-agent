# src/kume/services/nodes/set_context.py
from __future__ import annotations

from typing import Any

from kume.infrastructure.request_context import RequestContext, set_context


async def set_request_context(state: dict[str, Any]) -> dict[str, Any]:
    """Bootstrap node that sets the RequestContext contextvar from graph state.

    This ensures tools that depend on get_request_context() work both when
    invoked via the Telegram orchestrator and via direct graph invocation
    (LangGraph Platform).
    """
    user_id = state.get("user_id", "")
    language = state.get("user_language", "en")
    telegram_id = 0  # not available in direct graph invocation

    if user_id:
        set_context(RequestContext(user_id=user_id, telegram_id=telegram_id, language=language))

    return {}  # no state changes needed

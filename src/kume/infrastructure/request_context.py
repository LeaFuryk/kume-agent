"""Request-scoped context using contextvars (async-safe).

Set by the orchestrator at the start of each request.
Read by tools that need the current user's identity.
"""

from contextvars import ContextVar

current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)

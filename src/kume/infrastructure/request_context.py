"""Request-scoped context using contextvars (async-safe).

Set once by the orchestrator at the start of each request.
Read by tools, processors, and services that need request-level data.
Each async task gets its own copy — no shared mutable state.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    """Immutable snapshot of the current request's context."""

    user_id: str
    telegram_id: int
    language: str = "en"


_current: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


def set_context(ctx: RequestContext) -> None:
    """Set the request context for the current async task."""
    _current.set(ctx)


def get_context() -> RequestContext | None:
    """Get the request context for the current async task. Returns None if not set."""
    return _current.get()

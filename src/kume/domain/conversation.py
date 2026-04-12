from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ConversationEvent:
    """A single message in a user's conversation with Kume."""

    id: str
    user_id: str
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime

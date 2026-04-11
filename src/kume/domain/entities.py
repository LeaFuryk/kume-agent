from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    """A Telegram user of the Kume nutrition bot."""

    id: str
    telegram_id: int

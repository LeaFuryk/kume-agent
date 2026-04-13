from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("kume.context")


class ContextDataProvider(ABC):
    """Interface for fetching user context data. Injected into ContextBuilder."""

    @abstractmethod
    async def get_goals(self, user_id: str) -> list[Any]: ...

    @abstractmethod
    async def get_restrictions(self, user_id: str) -> list[Any]: ...

    @abstractmethod
    async def get_lab_markers(self, user_id: str) -> list[Any]: ...

    @abstractmethod
    async def search_documents(self, user_id: str, query: str) -> list[str]: ...

    @abstractmethod
    async def get_recent_meals(self, user_id: str) -> list[Any]: ...


class ContextBuilder:
    """Assembles RAG context in the prescribed order (CLAUDE.md rule #7).

    Uses the builder pattern with dependency injection — the data provider
    is injected at construction time, not passed per call.
    """

    def __init__(self, provider: ContextDataProvider) -> None:
        self._provider = provider

    async def build(self, user_id: str, query: str) -> str:
        """Build context string for LLM prompts.

        Order: goals → restrictions → relevant documents → lab markers → meals → current input.
        Empty sections are omitted. Each section is fetched independently —
        one failing section does not prevent others from loading.
        """
        sections: list[str] = []

        try:
            goals = await self._provider.get_goals(user_id)
            if goals:
                goals_text = "\n".join(f"- {g.description}" for g in goals)
                sections.append(f"## User Goals\n{goals_text}")
        except Exception:
            logger.warning("Failed to fetch goals for user %s", user_id, exc_info=True)

        try:
            restrictions = await self._provider.get_restrictions(user_id)
            if restrictions:
                restrictions_text = "\n".join(f"- [{r.type}] {r.description}" for r in restrictions)
                sections.append(f"## Dietary Restrictions\n{restrictions_text}")
        except Exception:
            logger.warning("Failed to fetch restrictions for user %s", user_id, exc_info=True)

        try:
            docs = await self._provider.search_documents(user_id, query)
            if docs:
                docs_text = "\n\n".join(docs)
                sections.append(f"## Relevant Health Documents\n{docs_text}")
        except Exception:
            logger.warning("Failed to search documents for user %s", user_id, exc_info=True)

        try:
            markers = await self._provider.get_lab_markers(user_id)
            if markers:
                markers_text = "\n".join(f"- {m.name}: {m.value} {m.unit} (ref: {m.reference_range})" for m in markers)
                sections.append(f"## Recent Lab Results\n{markers_text}")
        except Exception:
            logger.warning("Failed to fetch lab markers for user %s", user_id, exc_info=True)

        try:
            meals = await self._provider.get_recent_meals(user_id)
            if meals:
                meals_text = "\n".join(
                    f"- [{m.logged_at.strftime('%Y-%m-%d %H:%M')}] {m.description} "
                    f"({m.calories} kcal, {m.protein_g}g protein, {m.carbs_g}g carbs, {m.fat_g}g fat)"
                    for m in meals
                )
                sections.append(f"## Recent Meals\n{meals_text}")
        except Exception:
            logger.warning("Failed to fetch meals for user %s", user_id, exc_info=True)

        sections.append(f"## Current Question\n{query}")

        return "\n\n".join(sections)

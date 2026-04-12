from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.entities import Goal
from kume.ports.output.repositories import GoalRepository


class SaveGoalInput(BaseModel):
    description: str = Field(description="Description of the nutrition or health goal")


class SaveGoalTool(BaseTool):
    """LangChain tool that saves a nutrition or health goal.

    LangChain tool lifecycle:
        The agent calls .invoke()/.ainvoke() (public API).
        LangChain dispatches to _run() (sync) or _arun() (async) internally.
        _run() is a required sync fallback; _arun() is the primary async path.

    User identity:
        The orchestrator sets user_id via set_user_id() before each request.
        This avoids trusting the LLM to supply user_id.

        TODO: _current_user_id is mutable state on a shared instance. Safe for
        single-process sequential dispatch (python-telegram-bot default), but
        would cause contamination under concurrent requests. See ADR/known-limitations.
    """

    name: str = "save_goal"
    description: str = "Save a nutrition or health goal for the user (e.g., 'lose 5kg', 'eat more protein')"
    args_schema: type[BaseModel] = SaveGoalInput
    goal_repo: GoalRepository = Field(exclude=True)
    _current_user_id: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_user_id(self, user_id: str) -> None:
        self._current_user_id = user_id

    def _run(self, description: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(self._arun(description=description))

    async def _arun(self, description: str) -> str:
        """Primary async entry point — called by LangChain's agent via .ainvoke()."""
        if not self._current_user_id:
            return "Error: user_id not set. Cannot save goal."
        goal = Goal(
            id=str(uuid4()),
            user_id=self._current_user_id,
            description=description,
            created_at=datetime.now(tz=UTC),
        )
        await self.goal_repo.save(goal)
        return f"Goal saved: {description}"

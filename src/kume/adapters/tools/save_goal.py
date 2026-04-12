from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.entities import Goal
from kume.ports.output.repositories import GoalRepository


class SaveGoalInput(BaseModel):
    user_id: str = Field(description="The user's unique identifier")
    description: str = Field(description="Description of the nutrition or health goal")


class SaveGoalTool(BaseTool):
    """LangChain tool that saves a nutrition or health goal.

    LangChain tool lifecycle:
        The agent calls .invoke()/.ainvoke() (public API).
        LangChain dispatches to _run() (sync) or _arun() (async) internally.
        _run() is a required sync fallback; _arun() is the primary async path.
    """

    name: str = "save_goal"
    description: str = "Save a nutrition or health goal for the user (e.g., 'lose 5kg', 'eat more protein')"
    args_schema: type[BaseModel] = SaveGoalInput
    goal_repo: GoalRepository = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, user_id: str, description: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(self._arun(user_id=user_id, description=description))

    async def _arun(self, user_id: str, description: str) -> str:
        """Primary async entry point — called by LangChain's agent via .ainvoke()."""
        goal = Goal(
            id=str(uuid4()),
            user_id=user_id,
            description=description,
            created_at=datetime.now(tz=UTC),
        )
        await self.goal_repo.save(goal)
        return f"Goal saved: {description}"

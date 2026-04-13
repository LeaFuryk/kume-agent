from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.entities import Goal
from kume.infrastructure.request_context import get_context
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
        The orchestrator sets user_id via contextvars before each request.
        This avoids trusting the LLM to supply user_id.
    """

    name: str = "save_goal"
    description: str = (
        "Save a health or nutrition goal. Call BEFORE responding when user expresses ANY intention. "
        "Example: 'I want to lower my triglycerides' → save_goal(description='Lower triglycerides')"
    )
    args_schema: type[BaseModel] = SaveGoalInput
    goal_repo: GoalRepository = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, description: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(self._arun(description=description))

    async def _arun(self, description: str) -> str:
        """Primary async entry point — called by LangChain's agent via .ainvoke()."""
        ctx = get_context()
        if not ctx:
            return "Error: user_id not set. Cannot save goal."
        goal = Goal(
            id=str(uuid4()),
            user_id=ctx.user_id,
            description=description,
            created_at=datetime.now(tz=UTC),
        )
        await self.goal_repo.save(goal)
        return f"Goal saved: {description}"

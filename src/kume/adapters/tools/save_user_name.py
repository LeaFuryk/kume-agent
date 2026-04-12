"""Tool that persists the user's name when they introduce themselves."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.infrastructure.request_context import get_context
from kume.ports.output.repositories import UserRepository


class SaveUserNameInput(BaseModel):
    name: str = Field(description="The user's name")


class SaveUserNameTool(BaseTool):
    """LangChain tool that saves or updates the user's name.

    User identity:
        The orchestrator sets user_id via contextvars before each request.
        This avoids trusting the LLM to supply user_id.
    """

    name: str = "save_user_name"
    description: str = "Save or update the user's name when they introduce themselves"
    args_schema: type[BaseModel] = SaveUserNameInput
    user_repo: UserRepository = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, name: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(self._arun(name=name))

    async def _arun(self, name: str) -> str:
        """Primary async entry point — called by LangChain's agent via .ainvoke()."""
        ctx = get_context()
        if not ctx:
            return "Error: no user context"
        user = await self.user_repo.get_or_create(ctx.telegram_id)
        updated = replace(user, name=name)
        await self.user_repo.update(updated)
        return f"Name saved: {name}"

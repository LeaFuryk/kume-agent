from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.entities import Restriction
from kume.ports.output.repositories import RestrictionRepository


class SaveRestrictionInput(BaseModel):
    user_id: str = Field(description="The user's unique identifier")
    type: str = Field(description="Type of restriction: allergy, intolerance, or diet")
    description: str = Field(description="Description of the dietary restriction")


class SaveRestrictionTool(BaseTool):
    """LangChain tool that saves a dietary restriction.

    LangChain tool lifecycle:
        The agent calls .invoke()/.ainvoke() (public API).
        LangChain dispatches to _run() (sync) or _arun() (async) internally.
        _run() is a required sync fallback; _arun() is the primary async path.
    """

    name: str = "save_restriction"
    description: str = "Save a dietary restriction (allergy, intolerance, or diet preference) for the user"
    args_schema: type[BaseModel] = SaveRestrictionInput
    restriction_repo: RestrictionRepository = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, user_id: str, type: str, description: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(
            self._arun(user_id=user_id, type=type, description=description)
        )

    async def _arun(self, user_id: str, type: str, description: str) -> str:
        """Primary async entry point — called by LangChain's agent via .ainvoke()."""
        restriction = Restriction(
            id=str(uuid4()),
            user_id=user_id,
            type=type,
            description=description,
            created_at=datetime.now(tz=UTC),
        )
        await self.restriction_repo.save(restriction)
        return f"Restriction saved: [{type}] {description}"

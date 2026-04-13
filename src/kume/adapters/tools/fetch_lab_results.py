from __future__ import annotations

import asyncio

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.infrastructure.request_context import get_context
from kume.ports.output.repositories import LabMarkerRepository


class FetchLabResultsInput(BaseModel):
    query: str = Field(description="What the user is asking about their lab results")
    marker_name: str | None = Field(
        default=None,
        description="Optional: filter by specific marker name (e.g. 'colesterol', 'trigliceridos')",
    )


class FetchLabResultsTool(BaseTool):
    """LangChain tool that queries the user's lab markers directly.

    Use this for specific marker lookups — cholesterol, triglycerides, etc.
    For broader health context (goals, restrictions, documents), use
    FetchContextTool instead.
    """

    name: str = "fetch_lab_results"
    description: str = (
        "Look up the user's lab test results. Use for specific marker questions like "
        "'what was my cholesterol?' or 'show my triglycerides'. Optionally filter by "
        "marker name. For broad health questions, use fetch_user_context instead."
    )
    args_schema: type[BaseModel] = FetchLabResultsInput
    marker_repo: LabMarkerRepository = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str, marker_name: str | None = None) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(self._arun(query=query, marker_name=marker_name))

    async def _arun(self, query: str, marker_name: str | None = None) -> str:
        """Fetch the user's lab markers, optionally filtered by name."""
        ctx = get_context()
        if not ctx:
            return "Error: no user context available."

        try:
            markers = await self.marker_repo.get_by_user(ctx.user_id, name=marker_name)
            if not markers:
                filter_msg = f" matching '{marker_name}'" if marker_name else ""
                return f"No lab results found{filter_msg}."

            lines = [
                f"- {m.name}: {m.value} {m.unit} (ref: {m.reference_range}) [{m.date.strftime('%Y-%m-%d')}]"
                for m in markers
            ]
            return "\n".join(lines)
        except Exception:
            return "Error retrieving lab results."

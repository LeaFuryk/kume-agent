from __future__ import annotations

import asyncio

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.context import ContextBuilder
from kume.infrastructure.request_context import get_context


class FetchContextInput(BaseModel):
    query: str = Field(description="What the user is asking about — used to find relevant saved documents")


class FetchContextTool(BaseTool):
    """LangChain tool that retrieves the user's saved health context.

    Call this tool when you need the user's saved data to answer a question:
    goals, restrictions, lab markers, or health documents. The LLM decides
    when context is needed — it's not loaded on every request.

    LangChain tool lifecycle:
        The agent calls .invoke()/.ainvoke() (public API).
        LangChain dispatches to _run() (sync) or _arun() (async) internally.
    """

    name: str = "fetch_user_context"
    description: str = (
        "Retrieve the user's saved health data (goals, restrictions, lab results, documents). "
        "Call BEFORE answering questions about their data. "
        "Example: 'what were my triglycerides?' → fetch_user_context(query='triglyceride results')"
    )
    args_schema: type[BaseModel] = FetchContextInput
    context_builder: ContextBuilder = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(self._arun(query=query))

    async def _arun(self, query: str) -> str:
        """Fetch the user's saved context using the ContextBuilder."""
        ctx = get_context()
        if not ctx:
            return "No user context available."
        try:
            context = await self.context_builder.build(ctx.user_id, query)
            if not context or context.strip() == f"## Current Question\n{query}":
                return "No saved health data found for this user yet. Ask them to share their goals, restrictions, or lab results."
            return context
        except Exception:
            return "Error retrieving user context."

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.context import ContextBuilder
from kume.domain.tools import analyze_food as domain_analyze_food
from kume.infrastructure.request_context import get_context
from kume.ports.output.llm import LLMPort

logger = logging.getLogger(__name__)


class AnalyzeFoodInput(BaseModel):
    description: str = Field(description="Description of the food to analyze")


class AnalyzeFoodTool(BaseTool):
    """LangChain tool that analyzes food for nutritional content.

    LangChain tool lifecycle:
        The agent calls .invoke()/.ainvoke() (public API).
        LangChain dispatches to _run() (sync) or _arun() (async) internally.
        _run() is a required sync fallback; _arun() is the primary async path.

    Context building:
        Before calling the LLM, the tool retrieves the user's health context
        via the ContextBuilder. The orchestrator sets user_id via contextvars.
    """

    name: str = "analyze_food"
    description: str = (
        "Analyze a food item for nutritional content and health-goal alignment. "
        "Example: user asks 'can I eat pizza?' → analyze_food(description='pizza')"
    )
    args_schema: type[BaseModel] = AnalyzeFoodInput
    llm: LLMPort = Field(exclude=True)
    context_builder: ContextBuilder | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, description: str) -> str:
        context = self._build_context_sync(description)
        return domain_analyze_food(
            description,
            llm_call=lambda p: asyncio.get_event_loop().run_until_complete(self.llm.complete("", p)),
            context=context,
        )

    async def _arun(self, description: str) -> str:
        context = await self._build_context(description)
        # Build prompt inline to avoid calling run_until_complete inside an
        # already-running event loop.  Mirrors the prompt in
        # domain.tools.analyze_food so behaviour stays identical.
        prompt = (
            f"You are a nutrition expert.\n\n{context}\n\n"
            f"Analyze this food: {description}\n\n"
            "Provide nutritional assessment and whether it aligns with common health goals."
        )
        return await self.llm.complete("", prompt)

    def _build_context_sync(self, query: str) -> str:
        ctx = get_context()
        if self.context_builder is None or ctx is None:
            return ""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("Cannot build context synchronously inside a running event loop.")
                return ""
            return loop.run_until_complete(self.context_builder.build(ctx.user_id, query))
        except RuntimeError:
            logger.warning("Failed to build context synchronously", exc_info=True)
            return ""

    async def _build_context(self, query: str) -> str:
        ctx = get_context()
        if self.context_builder is None or ctx is None:
            return ""
        try:
            return await self.context_builder.build(ctx.user_id, query)
        except Exception:
            logger.warning("Failed to build context", exc_info=True)
            return ""

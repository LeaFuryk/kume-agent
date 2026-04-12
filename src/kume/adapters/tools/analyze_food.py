from __future__ import annotations

import asyncio
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kume.domain.context import ContextBuilder
from kume.domain.tools import analyze_food as domain_analyze_food

logger = logging.getLogger(__name__)


class AnalyzeFoodInput(BaseModel):
    description: str = Field(description="Description of the food to analyze")


class AnalyzeFoodTool(BaseTool):
    name: str = "analyze_food"
    description: str = "Analyze a food item or meal for nutritional content and alignment with health goals"
    args_schema: type[BaseModel] = AnalyzeFoodInput
    llm: BaseChatModel = Field(exclude=True)
    context_builder: ContextBuilder | None = Field(default=None, exclude=True)
    _current_user_id: str | None = None

    def set_user_id(self, user_id: str) -> None:
        """Set the current user ID for context building. Called by the orchestrator before each request."""
        self._current_user_id = user_id

    def _run(self, description: str) -> str:
        context = self._build_context_sync(description)
        return domain_analyze_food(description, llm_call=self._call_llm, context=context)

    async def _arun(self, description: str) -> str:
        context = await self._build_context(description)
        return domain_analyze_food(description, llm_call=self._call_llm, context=context)

    def _build_context_sync(self, query: str) -> str:
        """Build context synchronously. Falls back to empty string on failure."""
        if self.context_builder is None or self._current_user_id is None:
            return ""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning(
                    "Cannot build context synchronously inside a running event loop. "
                    "Use the async variant (_arun) instead."
                )
                return ""
            return loop.run_until_complete(self.context_builder.build(self._current_user_id, query))
        except RuntimeError:
            logger.warning("Failed to build context synchronously", exc_info=True)
            return ""

    async def _build_context(self, query: str) -> str:
        """Build context asynchronously. Falls back to empty string on failure."""
        if self.context_builder is None or self._current_user_id is None:
            return ""
        try:
            return await self.context_builder.build(self._current_user_id, query)
        except Exception:
            logger.warning("Failed to build context", exc_info=True)
            return ""

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts)
        return str(content or "")

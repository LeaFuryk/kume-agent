from __future__ import annotations

import asyncio
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kume.domain.context import ContextBuilder
from kume.domain.tools import ask_recommendation as domain_ask_recommendation

logger = logging.getLogger(__name__)


class AskRecommendationInput(BaseModel):
    query: str = Field(description="The user's nutrition question")


class AskRecommendationTool(BaseTool):
    """LangChain tool that provides personalized nutrition recommendations.

    LangChain tool lifecycle:
        The agent calls this tool via `.invoke()` or `.ainvoke()` (public API).
        LangChain internally dispatches to `_run()` (sync) or `_arun()` (async).
        These are internal hooks — not called directly by our code.

        Since Kume's Telegram bot is async, the agent uses `.ainvoke()` → `_arun()`.
        `_run()` exists as a required sync fallback per LangChain's BaseTool contract.

    Context building:
        Before calling the LLM, the tool retrieves the user's health context
        (goals, restrictions, lab markers, documents) via the ContextBuilder.
        The orchestrator sets the current user_id via `set_user_id()` before
        each request so the tool knows whose context to fetch.
    """

    name: str = "ask_recommendation"
    description: str = (
        "Get personalized nutrition recommendations based on the user's question about diet, meals, or health goals"
    )
    args_schema: type[BaseModel] = AskRecommendationInput
    llm: BaseChatModel = Field(exclude=True)
    context_builder: ContextBuilder | None = Field(default=None, exclude=True)
    _current_user_id: str | None = None

    def set_user_id(self, user_id: str) -> None:
        """Set the current user ID for context building. Called by the orchestrator before each request."""
        self._current_user_id = user_id

    def _run(self, query: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract. Prefer _arun in async contexts."""
        context = self._build_context_sync(query)
        return domain_ask_recommendation(query, llm_call=self._call_llm, context=context)

    async def _arun(self, query: str) -> str:
        """Primary async entry point — called by LangChain's agent via .ainvoke()."""
        context = await self._build_context(query)
        return domain_ask_recommendation(query, llm_call=self._call_llm, context=context)

    def _build_context_sync(self, query: str) -> str:
        """Build context synchronously. Falls back to empty string on failure."""
        if self.context_builder is None or self._current_user_id is None:
            return ""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run_until_complete.
                # Return empty and log a warning — callers should use _arun instead.
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

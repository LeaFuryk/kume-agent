from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.context import ContextBuilder
from kume.domain.tools import ask_recommendation as domain_ask_recommendation
from kume.infrastructure.request_context import current_user_id
from kume.ports.output.llm import LLMPort

logger = logging.getLogger(__name__)


class AskRecommendationInput(BaseModel):
    query: str = Field(description="The user's nutrition question")


class AskRecommendationTool(BaseTool):
    """LangChain tool that provides personalized nutrition recommendations.

    LangChain tool lifecycle:
        The agent calls .invoke()/.ainvoke() (public API).
        LangChain dispatches to _run() (sync) or _arun() (async) internally.
        _run() is a required sync fallback; _arun() is the primary async path.

    Context building:
        Before calling the LLM, the tool retrieves the user's health context
        via the ContextBuilder. The orchestrator sets user_id via contextvars.
    """

    name: str = "ask_recommendation"
    description: str = (
        "Get personalized nutrition recommendations based on the user's question about diet, meals, or health goals"
    )
    args_schema: type[BaseModel] = AskRecommendationInput
    llm: LLMPort = Field(exclude=True)
    context_builder: ContextBuilder | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str) -> str:
        context = self._build_context_sync(query)
        return domain_ask_recommendation(
            query,
            llm_call=lambda p: asyncio.get_event_loop().run_until_complete(self.llm.complete("", p)),
            context=context,
        )

    async def _arun(self, query: str) -> str:
        context = await self._build_context(query)
        # Build prompt inline to avoid calling run_until_complete inside an
        # already-running event loop.  Mirrors the prompt in
        # domain.tools.ask_recommendation so behaviour stays identical.
        prompt = (
            f"You are a nutrition expert.\n\n{context}\n\n"
            f"The user asks: {query}\n\n"
            "Provide a helpful, personalized nutrition recommendation."
        )
        return await self.llm.complete("", prompt)

    def _build_context_sync(self, query: str) -> str:
        if self.context_builder is None or current_user_id.get() is None:
            return ""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logger.warning("Cannot build context synchronously inside a running event loop.")
                return ""
            return loop.run_until_complete(self.context_builder.build(current_user_id.get(), query))
        except RuntimeError:
            logger.warning("Failed to build context synchronously", exc_info=True)
            return ""

    async def _build_context(self, query: str) -> str:
        if self.context_builder is None or current_user_id.get() is None:
            return ""
        try:
            return await self.context_builder.build(current_user_id.get(), query)
        except Exception:
            logger.warning("Failed to build context", exc_info=True)
            return ""

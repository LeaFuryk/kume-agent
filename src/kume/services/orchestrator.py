from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from kume.infrastructure.metrics import MetricsCallbackHandler, MetricsCollector
from kume.infrastructure.request_context import RequestContext, set_context
from kume.ports.output.repositories import UserRepository
from kume.services.prompts import SYSTEM_PROMPT

logger = logging.getLogger("kume.orchestrator")


def _extract_text_content(content: Any) -> str:
    """Extract plain text from AIMessage content, which may be a string or structured blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content) if content else ""


class OrchestratorService:
    """Application service that owns the agentic tool-use loop.

    Creates a LangChain agent and manages per-request metrics collection.
    A fresh MetricsCollector is created for each request to ensure thread safety
    when handling concurrent Telegram updates.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = 5,
        user_repo: UserRepository | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._tools = tools
        self._user_repo = user_repo
        self._agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    async def process(self, telegram_id: int, text: str) -> str:
        """Process a user message through the agentic loop and return the response."""
        collector = MetricsCollector()
        collector.start_request(telegram_id)
        callback_handler = MetricsCallbackHandler(collector)

        # Resolve telegram_id -> user_id and set in request-scoped context.
        # Include user name in message prefix so LLM knows who it's talking to.
        user_prefix = ""
        if self._user_repo is not None:
            try:
                user = await self._user_repo.get_or_create(telegram_id)
                set_context(RequestContext(user_id=user.id, telegram_id=telegram_id, language="en"))
                if user.name:
                    user_prefix = f"[User: {user.name}]\n"
            except Exception:
                logger.warning("Failed to resolve user_id for telegram_id=%d", telegram_id, exc_info=True)

        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=f"{user_prefix}{text}")]},
                config={
                    "callbacks": [callback_handler],
                    "recursion_limit": self._max_iterations * 2,
                },
            )
            messages = result.get("messages", [])
            if messages:
                text = _extract_text_content(messages[-1].content)
                if text.strip():
                    return text
            return "I wasn't able to process that request."
        except Exception:
            logger.exception("Error processing message for telegram_id=%d", telegram_id)
            return "Sorry, something went wrong. Please try again."
        finally:
            collector.end_request()

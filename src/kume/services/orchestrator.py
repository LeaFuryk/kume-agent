from __future__ import annotations

import logging
from dataclasses import dataclass, replace
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


@dataclass
class Resource:
    mime_type: str
    transcript: str
    raw_bytes: bytes | None = None  # kept for image tools that need the original


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

    async def _resolve_user(self, telegram_id: int, user_name: str | None = None) -> str:
        """Resolve telegram_id to internal user, set request context, return message prefix.

        - Returning user (name in DB): returns '[User: name]\\n'
        - First-time user: saves name from Telegram, returns '' (triggers onboarding)
        - No user_repo or failure: returns ''
        """
        if self._user_repo is None:
            return ""

        try:
            user = await self._user_repo.get_or_create(telegram_id)
            set_context(RequestContext(user_id=user.id, telegram_id=telegram_id, language="en"))

            if user.name:
                return f"[User: {user.name}]\n"

            if user_name:
                try:
                    await self._user_repo.update(replace(user, name=user_name))
                except Exception:
                    logger.warning("Failed to save user name for telegram_id=%d", telegram_id, exc_info=True)

            return ""
        except Exception:
            logger.warning("Failed to resolve user_id for telegram_id=%d", telegram_id, exc_info=True)
            return ""

    async def process(
        self,
        telegram_id: int,
        user_message: str,
        user_name: str | None = None,
        resources: list[Resource] | None = None,
    ) -> str:
        """Process a user message through the agentic loop and return the response."""
        collector = MetricsCollector()
        collector.start_request(telegram_id)
        callback_handler = MetricsCallbackHandler(collector)

        parts: list[str] = []

        # User prefix
        user_prefix = await self._resolve_user(telegram_id, user_name)
        if user_prefix:
            parts.append(user_prefix.strip())

        # User message
        if user_message:
            parts.append(f"User says: {user_message}")

        # Resources
        if resources:
            # Count by type
            pdf_count = sum(1 for r in resources if r.mime_type == "application/pdf")
            img_count = sum(1 for r in resources if r.mime_type.startswith("image/"))
            audio_count = sum(1 for r in resources if r.mime_type.startswith("audio/"))

            type_summary: list[str] = []
            if pdf_count:
                type_summary.append(f"{pdf_count} PDF document(s)")
            if img_count:
                type_summary.append(f"{img_count} image(s)")
            if audio_count:
                type_summary.append(f"{audio_count} audio file(s)")

            parts.append(f"Attached resources: {', '.join(type_summary)}")

            # Add each transcript labeled
            for i, resource in enumerate(resources, 1):
                parts.append(f"Resource {i} ({resource.mime_type}):\n{resource.transcript}")

        full_message = "\n\n".join(parts)

        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=full_message)]},
                config={
                    "callbacks": [callback_handler],
                    "recursion_limit": self._max_iterations * 2,
                },
            )
            messages = result.get("messages", [])
            if messages:
                response_text = _extract_text_content(messages[-1].content)
                if response_text.strip():
                    return response_text
            return "I wasn't able to process that request."
        except Exception:
            logger.exception("Error processing message for telegram_id=%d", telegram_id)
            return "Sorry, something went wrong. Please try again."
        finally:
            collector.end_request()

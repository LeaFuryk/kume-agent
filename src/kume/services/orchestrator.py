from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from kume.domain.conversation import ConversationEvent
from kume.infrastructure.image_store import ImageStore
from kume.infrastructure.metrics import MetricsCallbackHandler, MetricsCollector
from kume.infrastructure.request_context import (
    RequestContext,
    set_context,
)
from kume.infrastructure.request_context import (
    get_context as get_request_context,
)
from kume.infrastructure.session_store import SessionStore
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
        session_store: SessionStore | None = None,
        image_store: ImageStore | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._tools = tools
        self._user_repo = user_repo
        self._session_store = session_store
        self._image_store = image_store
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

        # Resolve user_id from RequestContext (set by _resolve_user)
        req_ctx = get_request_context()
        user_id = req_ctx.user_id if req_ctx else ""

        # Load conversation history from SessionStore (under per-user lock
        # to prevent race conditions on overlapping requests)
        history_messages: list[HumanMessage | AIMessage] = []
        session_lock = None
        lock_acquired = False
        if self._session_store and user_id:
            session_lock = self._session_store._get_lock(user_id)
            await session_lock.acquire()
            lock_acquired = True
            session = self._session_store.get_session(user_id)
            for event in session:
                if event.role == "user":
                    history_messages.append(HumanMessage(content=event.content))
                else:
                    history_messages.append(AIMessage(content=event.content))

        # Store image bytes + MIME types in ImageStore for tools to access
        request_id = str(uuid4())
        if self._image_store and resources:
            image_resources = [r for r in resources if r.raw_bytes and r.mime_type.startswith("image/")]
            if image_resources:
                self._image_store.set_images(
                    request_id,
                    [r.raw_bytes for r in image_resources],
                    [r.mime_type for r in image_resources],
                )

        # User message (labeled to match prompt's language detection instructions)
        if user_message:
            parts.append(f"[User message]: {user_message}")

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

            # Add each transcript labeled, with image-specific indices for analyze_food_image
            image_idx = 0
            for resource in resources:
                if resource.mime_type.startswith("image/"):
                    image_idx += 1
                    parts.append(f"Image {image_idx} ({resource.mime_type}):\n{resource.transcript}")
                else:
                    parts.append(f"Document ({resource.mime_type}):\n{resource.transcript}")

        full_message = "\n\n".join(parts)

        # Build messages with history + current message
        messages = history_messages + [HumanMessage(content=full_message)]

        try:
            result = await self._agent.ainvoke(
                {"messages": messages},
                config={
                    "callbacks": [callback_handler],
                    "recursion_limit": self._max_iterations * 2,
                },
            )
            resp_messages = result.get("messages", [])
            if resp_messages:
                response_text = _extract_text_content(resp_messages[-1].content)
                if response_text.strip():
                    # Save conversation events to SessionStore
                    # Use a compact summary for session history to avoid replaying
                    # full resource transcripts (PDFs, OCR) on subsequent turns.
                    if self._session_store and user_id:
                        now = datetime.now(UTC)
                        history_content = user_message or ""
                        if resources:
                            resource_types = [r.mime_type for r in resources]
                            history_content += f" [+ {len(resources)} attachment(s): {', '.join(resource_types)}]"
                        self._session_store.add(
                            user_id,
                            ConversationEvent(
                                id=str(uuid4()),
                                user_id=user_id,
                                role="user",
                                content=history_content.strip(),
                                created_at=now,
                            ),
                        )
                        self._session_store.add(
                            user_id,
                            ConversationEvent(
                                id=str(uuid4()),
                                user_id=user_id,
                                role="assistant",
                                content=response_text,
                                created_at=now,
                            ),
                        )
                    return response_text
            return "I wasn't able to process that request."
        except Exception:
            logger.exception("Error processing message for telegram_id=%d", telegram_id)
            return "Sorry, something went wrong. Please try again."
        finally:
            if session_lock and lock_acquired:
                session_lock.release()
            if self._image_store:
                self._image_store.clear(request_id)
            set_context(None)  # type: ignore[arg-type]  # clear stale user context
            collector.end_request()

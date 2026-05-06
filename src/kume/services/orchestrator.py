from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from kume.domain.conversation import ConversationEvent
from kume.infrastructure.image_store import ImageStore
from kume.infrastructure.request_context import (
    RequestContext,
    set_context,
)
from kume.infrastructure.request_context import (
    get_context as get_request_context,
)
from kume.infrastructure.session_store import SessionStore
from kume.ports.output.messaging import MessagingPort
from kume.ports.output.repositories import UserRepository

logger = logging.getLogger("kume.orchestrator")


@dataclass
class Resource:
    mime_type: str
    transcript: str
    raw_bytes: bytes | None = None  # kept for image tools that need the original


@dataclass
class ProcessResult:
    """Return type for OrchestratorService.process().

    Attributes:
        text: The response text from the LLM agent.
        streamed: True if the response was already delivered via streaming
                  edits (the caller should NOT send it again).
    """

    text: str
    streamed: bool = False


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

    Delegates to a compiled LangGraph pipeline for agent execution,
    guardrails, memory management, and response formatting.
    """

    def __init__(
        self,
        graph: Any,
        max_iterations: int = 5,
        user_repo: UserRepository | None = None,
        session_store: SessionStore | None = None,
        image_store: ImageStore | None = None,
    ) -> None:
        self._max_iterations = max_iterations
        self._graph = graph
        self._user_repo = user_repo
        self._session_store = session_store
        self._image_store = image_store

    async def _resolve_user(self, telegram_id: int, user_name: str | None = None, language: str | None = None) -> str:
        """Resolve telegram_id to internal user, set request context, return message prefix.

        - Returning user (name in DB): returns '[User: name]\\n'
        - First-time user: saves name from Telegram, returns '' (triggers onboarding)
        - No user_repo or failure: returns ''
        """
        if self._user_repo is None:
            return ""

        try:
            user = await self._user_repo.get_or_create(telegram_id)
            set_context(RequestContext(user_id=user.id, telegram_id=telegram_id, language=language or "en"))

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
        language: str | None = None,
        # Streaming support
        messaging: MessagingPort | None = None,
        chat_id: int | None = None,
    ) -> ProcessResult:
        """Process a user message through the LangGraph pipeline and return the response."""
        parts: list[str] = []

        # Language instruction — tell the LLM which language to respond in
        if language:
            lang_names = {
                "es": "Spanish",
                "en": "English",
                "pt": "Portuguese",
                "fr": "French",
                "de": "German",
                "it": "Italian",
            }
            lang_name = lang_names.get(language[:2], language)
            parts.append(f"[Respond in: {lang_name}]")

        # User prefix
        user_prefix = await self._resolve_user(telegram_id, user_name, language=language)
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
                image_bytes = [r.raw_bytes for r in image_resources if r.raw_bytes is not None]
                image_mimes = [r.mime_type for r in image_resources]
                if image_bytes:
                    self._image_store.set_images(request_id, image_bytes, image_mimes)

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

        # Send a "thinking" placeholder if messaging is available.
        # After ainvoke completes, we edit the placeholder with the final response.
        # (True token-by-token streaming requires astream, which is a future enhancement.)
        placeholder_message_id: int | None = None
        if messaging and chat_id:
            try:
                placeholder_message_id = await messaging.send_and_get_id(chat_id, "...")
            except Exception:
                logger.warning("Failed to send placeholder, will send response normally", exc_info=True)

        try:
            result = await self._graph.ainvoke(
                {
                    "messages": messages,
                    "user_id": user_id,
                    "user_name": user_name,
                    "user_language": language or "en",
                    "input_safe": True,
                    "output_safe": True,
                    "guardrail_violation": None,
                    "raw_agent_response": "",
                    "formatted_response": "",
                    "memory_summarized": False,
                    "tool_error_count": 0,
                },
                # Each agent iteration = 2 recursion steps (model call + tool call)
                # Plus 4 fixed nodes (manage_memory, input_guardrail, format_response, output_guardrail)
                config={"recursion_limit": max(self._max_iterations * 2 + 4, 10)},
            )
            response_text = result.get("formatted_response", "")
            input_was_safe = result.get("input_safe", True)

            if response_text.strip():
                # Save conversation events to SessionStore
                # Use a compact summary for session history to avoid replaying
                # full resource transcripts (PDFs, OCR) on subsequent turns.
                # Only persist if the input was not blocked by the guardrail,
                # so blocked injection text doesn't pollute session history.
                if self._session_store and user_id and input_was_safe:
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
                # Edit the placeholder with the final response
                streamed = False
                if placeholder_message_id and messaging and chat_id:
                    try:
                        await messaging.edit_message(chat_id, placeholder_message_id, response_text)
                        streamed = True
                    except Exception:
                        logger.warning("Failed to edit placeholder, will send as new message", exc_info=True)
                return ProcessResult(text=response_text, streamed=streamed)
            return ProcessResult(text="I wasn't able to process that request.", streamed=False)
        except Exception:
            logger.exception("Error processing message for telegram_id=%d", telegram_id)
            return ProcessResult(text="Sorry, something went wrong. Please try again.", streamed=False)
        finally:
            if session_lock and lock_acquired:
                session_lock.release()
            if self._image_store:
                self._image_store.clear(request_id)
            set_context(None)  # type: ignore[arg-type]  # clear stale user context

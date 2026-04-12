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

SYSTEM_PROMPT = """You are Kume, a personal AI nutrition companion. You're warm, encouraging, \
and knowledgeable — like a friend who happens to know a lot about nutrition.

Mirror the language the user writes in. Use their first name when you know it. \
Keep responses concise but friendly.

## Your Mission

You help people take control of their nutrition and health goals. A typical user \
might have just gotten lab results showing high triglycerides or cholesterol, \
and needs help understanding what to eat, tracking their meals, and measuring \
progress over time.

You are NOT a replacement for a nutritionist — always recommend they work with \
a professional for a personalized plan. Your role is to help them execute that \
plan: track what they eat, understand their lab results, stay motivated, and \
measure progress between checkups.

What you can do today:
- Answer nutrition questions personalized to the user's health context
- Analyze food and meals for nutritional content
- Save health goals and dietary restrictions
- Parse lab reports (PDF) and extract markers for tracking
- Transcribe voice notes about diet or health
- Remember everything the user shares for better future advice

Coming soon:
- Food photo analysis
- Meal logging and daily tracking
- Progress reports comparing lab results over time

## First Interaction

When a user greets you for the first time or says hello:
1. Introduce yourself in 2-3 sentences: your name, your mission, and that \
you work best alongside a nutritionist
2. Ask for their name naturally
3. Suggest one thing they can try right now: "You can send me your lab results \
as a PDF and I'll help you understand them, or just ask me a nutrition question"

## Help Requests

When the user asks what you can do or how to use you:
- Explain with concrete examples:
  "Send me a PDF of your lab results → I'll extract your markers and remember them"
  "Tell me 'My goal is to lower my triglycerides' → I'll save it and personalize my advice"
  "Ask 'Can I eat this?' → I'll analyze it based on your goals and restrictions"
  "Send a voice note about what you ate → I'll process it"
- Mention you get smarter the more they share (goals, restrictions, lab results)
- Keep it conversational — don't dump a feature list

## Opportunistic Learning

When the user sends a closure message (thanks, ok, got it, bye, etc.) and you \
don't yet have important context about them, ask ONE gentle follow-up:

Priority:
1. Name (if unknown)
2. Main health goal (if no goals saved)
3. Dietary restrictions (if none saved)
4. Physical context (weight, height, activity) — only when relevant to \
something they just discussed

Rules:
- One question maximum per closure moment
- Must relate to the recent conversation
- Frame as helpful: "Knowing your weight helps me give better portion advice — mind sharing?"
- If they decline, don't ask again in the same session

## Tool Usage

When the user shares health information (goals, restrictions, weight, diet \
preferences, conditions), ALWAYS save it using the appropriate tool. Don't \
just acknowledge — persist it.

When answering nutrition questions, your context already includes the user's \
saved goals, restrictions, lab markers, and documents. Use them to personalize \
every response.

## Motivation & Support

When sharing results or progress, be encouraging. Celebrate small wins. \
If lab markers improved, highlight it. If the user is struggling, empathize \
and suggest practical next steps — never guilt.

Remind users periodically (not every message) that tracking consistently \
is what drives results: "Your next lab checkup will show the real progress!"
"""

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
        # Uses contextvars (async-safe) — each task gets its own copy.
        if self._user_repo is not None:
            try:
                user = await self._user_repo.get_or_create(telegram_id)
                set_context(RequestContext(user_id=user.id, telegram_id=telegram_id, language="en"))
            except Exception:
                logger.warning("Failed to resolve user_id for telegram_id=%d", telegram_id, exc_info=True)

        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=text)]},
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

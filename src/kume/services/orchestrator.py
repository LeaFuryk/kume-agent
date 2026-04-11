from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from kume.infrastructure.metrics import MetricsCallbackHandler, MetricsCollector

SYSTEM_PROMPT = """You are Kume, a personal AI nutrition assistant. You help users with:
- Nutrition recommendations and diet advice
- Food analysis and nutritional assessment
- Health goal guidance

When the user asks about food, nutrition, diet, or health goals, use the appropriate tool.
For casual conversation, greetings, or off-topic questions, respond conversationally without using tools.

Be friendly, concise, and evidence-based in your responses."""

logger = logging.getLogger("kume.orchestrator")


class OrchestratorService:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        metrics_collector: MetricsCollector,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self._metrics_collector = metrics_collector
        self._agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    async def process(self, telegram_id: int, text: str) -> str:
        self._metrics_collector.start_request(telegram_id)
        callback_handler = MetricsCallbackHandler(self._metrics_collector)

        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=text)]},
                config={"callbacks": [callback_handler]},
            )
            messages = result.get("messages", [])
            if messages:
                return str(messages[-1].content)
            return "I wasn't able to process that request."
        except Exception:
            logger.exception("Error processing message for telegram_id=%d", telegram_id)
            return "Sorry, something went wrong. Please try again."
        finally:
            self._metrics_collector.end_request()

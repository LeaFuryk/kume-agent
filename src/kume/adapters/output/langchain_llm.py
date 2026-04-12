"""LangChain adapter implementing the LLMPort interface."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from kume.ports.output.llm import LLMPort


class LangChainLLMAdapter(LLMPort):
    """LLM adapter using LangChain's BaseChatModel.

    Handles structured content extraction internally — consumers
    always receive a plain string from complete().
    """

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = await self._model.ainvoke(messages)
        return _extract_text(response.content)

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        structured_model = self._model.bind(
            response_format={"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}
        )
        response = await structured_model.ainvoke(messages)
        return _extract_text(response.content)


def _extract_text(content: Any) -> str:
    """Extract plain text from LLM response content (string or structured blocks)."""
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

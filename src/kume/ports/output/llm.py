from abc import ABC, abstractmethod
from typing import Any


class LLMPort(ABC):
    """Port for LLM text completion.

    Abstracts the LLM provider — adapters implement this with LangChain,
    OpenAI SDK, or any other provider. Consumers receive plain strings,
    never framework-specific types.
    """

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...

    @abstractmethod
    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> str:
        """Complete with structured JSON output. The provider enforces the schema.

        Returns a JSON string guaranteed to match the provided JSON Schema.
        """
        ...

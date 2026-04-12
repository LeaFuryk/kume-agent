from abc import ABC, abstractmethod


class LLMPort(ABC):
    """Port for LLM text completion.

    Abstracts the LLM provider — adapters implement this with LangChain,
    OpenAI SDK, or any other provider. Consumers receive plain strings,
    never framework-specific types.
    """

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...

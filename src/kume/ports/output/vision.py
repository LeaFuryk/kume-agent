from abc import ABC, abstractmethod
from typing import Any


class VisionPort(ABC):
    @abstractmethod
    async def analyze_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str: ...

    @abstractmethod
    async def analyze_image_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: dict[str, Any],
    ) -> str:
        """Analyze image with structured JSON output matching the provided schema."""
        ...

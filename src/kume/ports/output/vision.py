from abc import ABC, abstractmethod


class VisionPort(ABC):
    @abstractmethod
    async def analyze_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str: ...

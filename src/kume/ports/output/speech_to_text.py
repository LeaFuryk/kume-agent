from abc import ABC, abstractmethod


class SpeechToTextPort(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str = "es", *, mime_type: str | None = None) -> str: ...

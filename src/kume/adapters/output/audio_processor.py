from kume.ports.output.resource_processor import ResourceProcessorPort
from kume.ports.output.speech_to_text import SpeechToTextPort


class AudioProcessor(ResourceProcessorPort):
    """Processes audio by delegating to a speech-to-text port."""

    def __init__(self, stt: SpeechToTextPort, language: str = "es") -> None:
        self._stt = stt
        self._language = language

    async def process(self, raw_bytes: bytes, *, mime_type: str | None = None) -> str:
        return await self._stt.transcribe(raw_bytes, language=self._language, mime_type=mime_type)

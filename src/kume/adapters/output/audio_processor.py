from kume.ports.output.resource_processor import ResourceProcessorPort
from kume.ports.output.speech_to_text import SpeechToTextPort


class AudioProcessor(ResourceProcessorPort):
    """Processes audio by delegating to a speech-to-text port."""

    def __init__(self, stt: SpeechToTextPort) -> None:
        self._stt = stt

    async def process(self, raw_bytes: bytes) -> str:
        return await self._stt.transcribe(raw_bytes)

import io

from openai import AsyncOpenAI

from kume.ports.output.speech_to_text import SpeechToTextPort

_MIME_TO_EXT: dict[str, str] = {
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
}


class WhisperAdapter(SpeechToTextPort):
    """Speech-to-text adapter using OpenAI Whisper API.

    Uses the OpenAI SDK directly rather than LangChain's OpenAIWhisperParser
    because the parser is a document loader designed for file paths, not raw
    bytes. Our SpeechToTextPort interface (bytes -> str) maps cleanly to the
    OpenAI transcriptions API.
    """

    def __init__(self, api_key: str, model: str = "whisper-1") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def transcribe(self, audio_bytes: bytes, language: str = "es", *, mime_type: str | None = None) -> str:
        ext = _MIME_TO_EXT.get(mime_type or "", ".ogg")
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio{ext}"
        response = await self._client.audio.transcriptions.create(
            model=self._model,
            file=audio_file,
            language=language,
        )
        return response.text

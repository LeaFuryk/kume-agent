import io

from openai import AsyncOpenAI

from kume.ports.output.speech_to_text import SpeechToTextPort


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

    async def transcribe(self, audio_bytes: bytes, language: str = "es") -> str:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.ogg"
        response = await self._client.audio.transcriptions.create(
            model=self._model,
            file=audio_file,
            language=language,
        )
        return response.text

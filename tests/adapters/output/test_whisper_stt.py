from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kume.adapters.output.whisper_stt import WhisperAdapter
from kume.ports.output.speech_to_text import SpeechToTextPort


class TestWhisperAdapter:
    def test_implements_speech_to_text_port(self) -> None:
        assert issubclass(WhisperAdapter, SpeechToTextPort)

    @pytest.mark.asyncio
    async def test_transcribe_calls_openai_api(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "Hola, quiero registrar mi comida"

        with patch("kume.adapters.output.whisper_stt.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

            adapter = WhisperAdapter(api_key="test-key")
            result = await adapter.transcribe(b"fake-audio", language="es")

        assert result == "Hola, quiero registrar mi comida"
        mock_client.audio.transcriptions.create.assert_awaited_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["model"] == "whisper-1"
        assert call_kwargs["language"] == "es"
        assert call_kwargs["file"].read() == b"fake-audio"

    @pytest.mark.asyncio
    async def test_transcribe_uses_custom_model(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "transcribed text"

        with patch("kume.adapters.output.whisper_stt.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

            adapter = WhisperAdapter(api_key="test-key", model="whisper-2")
            await adapter.transcribe(b"audio-data")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["model"] == "whisper-2"

    @pytest.mark.asyncio
    async def test_transcribe_default_language_is_spanish(self) -> None:
        mock_response = MagicMock()
        mock_response.text = "texto"

        with patch("kume.adapters.output.whisper_stt.AsyncOpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

            adapter = WhisperAdapter(api_key="test-key")
            await adapter.transcribe(b"audio")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["language"] == "es"

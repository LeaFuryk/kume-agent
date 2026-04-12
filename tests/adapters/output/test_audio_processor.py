from unittest.mock import AsyncMock

import pytest

from kume.adapters.output.audio_processor import AudioProcessor
from kume.ports.output.resource_processor import ResourceProcessorPort
from kume.ports.output.speech_to_text import SpeechToTextPort


class TestAudioProcessor:
    def test_implements_resource_processor_port(self) -> None:
        assert issubclass(AudioProcessor, ResourceProcessorPort)

    @pytest.mark.asyncio
    async def test_delegates_to_stt_port(self) -> None:
        mock_stt = AsyncMock(spec=SpeechToTextPort)
        mock_stt.transcribe.return_value = "Hoy desayune avena con frutas"

        processor = AudioProcessor(stt=mock_stt)
        result = await processor.process(b"fake-audio-bytes")

        assert result == "Hoy desayune avena con frutas"
        mock_stt.transcribe.assert_awaited_once_with(b"fake-audio-bytes", language="es", mime_type=None)

    @pytest.mark.asyncio
    async def test_passes_raw_bytes_unchanged(self) -> None:
        mock_stt = AsyncMock(spec=SpeechToTextPort)
        mock_stt.transcribe.return_value = ""

        audio_data = b"\x00\x01\x02\x03"
        processor = AudioProcessor(stt=mock_stt)
        await processor.process(audio_data)

        mock_stt.transcribe.assert_awaited_once_with(audio_data, language="es", mime_type=None)

    @pytest.mark.asyncio
    async def test_passes_mime_type_to_stt(self) -> None:
        mock_stt = AsyncMock(spec=SpeechToTextPort)
        mock_stt.transcribe.return_value = "transcribed"

        processor = AudioProcessor(stt=mock_stt)
        result = await processor.process(b"audio-data", mime_type="audio/mpeg")

        assert result == "transcribed"
        mock_stt.transcribe.assert_awaited_once_with(b"audio-data", language="es", mime_type="audio/mpeg")

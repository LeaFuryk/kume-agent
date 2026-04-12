import pytest

from kume.ports.output.speech_to_text import SpeechToTextPort


def test_speech_to_text_port_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        SpeechToTextPort()  # type: ignore[abstract]


class _FakeSpeechToText(SpeechToTextPort):
    async def transcribe(self, audio_bytes: bytes, language: str = "es") -> str:
        return f"transcribed:{language}"


async def test_concrete_subclass_can_transcribe() -> None:
    stt = _FakeSpeechToText()
    result = await stt.transcribe(b"\x00\x01\x02")
    assert result == "transcribed:es"


async def test_transcribe_with_custom_language() -> None:
    stt = _FakeSpeechToText()
    result = await stt.transcribe(b"\x00", language="en")
    assert result == "transcribed:en"


def test_speech_to_text_port_importable_from_ports_package() -> None:
    from kume.ports import SpeechToTextPort as FromPorts
    from kume.ports.output import SpeechToTextPort as FromOutput

    assert FromPorts is FromOutput

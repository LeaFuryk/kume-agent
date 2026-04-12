from unittest.mock import AsyncMock

import pytest

from kume.ports.output.resource_processor import ResourceProcessorPort
from kume.services.ingestion import IngestionService, UnsupportedMediaType


@pytest.fixture
def pdf_processor() -> AsyncMock:
    mock = AsyncMock(spec=ResourceProcessorPort)
    mock.process.return_value = "extracted pdf text"
    return mock


@pytest.fixture
def audio_processor() -> AsyncMock:
    mock = AsyncMock(spec=ResourceProcessorPort)
    mock.process.return_value = "transcribed audio text"
    return mock


async def test_routes_to_correct_processor_by_mime_type(pdf_processor: AsyncMock) -> None:
    service = IngestionService(processors={"application/pdf": pdf_processor})

    result = await service.process(b"fake-pdf-bytes", "application/pdf")

    assert result == "extracted pdf text"
    pdf_processor.process.assert_awaited_once_with(b"fake-pdf-bytes")


async def test_raises_unsupported_media_type_for_unknown_mime() -> None:
    service = IngestionService(processors={})

    with pytest.raises(UnsupportedMediaType) as exc_info:
        await service.process(b"data", "video/mp4")

    assert exc_info.value.mime_type == "video/mp4"
    assert "video/mp4" in str(exc_info.value)


async def test_multiple_processors_registered(
    pdf_processor: AsyncMock,
    audio_processor: AsyncMock,
) -> None:
    service = IngestionService(
        processors={
            "application/pdf": pdf_processor,
            "audio/ogg": audio_processor,
        }
    )

    pdf_result = await service.process(b"pdf-data", "application/pdf")
    audio_result = await service.process(b"audio-data", "audio/ogg")

    assert pdf_result == "extracted pdf text"
    assert audio_result == "transcribed audio text"
    pdf_processor.process.assert_awaited_once_with(b"pdf-data")
    audio_processor.process.assert_awaited_once_with(b"audio-data")

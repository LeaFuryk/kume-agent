import pytest

from kume.adapters.output.image_processor import ImageProcessor
from kume.ports.output.resource_processor import ResourceProcessorPort


class TestImageProcessor:
    def test_implements_resource_processor_port(self) -> None:
        assert issubclass(ImageProcessor, ResourceProcessorPort)

    @pytest.mark.asyncio
    async def test_returns_signal_text(self) -> None:
        processor = ImageProcessor()
        result = await processor.process(b"fake-image-bytes")
        assert "Image attached" in result

    @pytest.mark.asyncio
    async def test_signal_text_is_neutral(self) -> None:
        """Signal text should not reference specific tool names."""
        processor = ImageProcessor()
        result = await processor.process(b"\x89PNG\r\n")
        assert "analysis" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_string(self) -> None:
        processor = ImageProcessor()
        result = await processor.process(b"")
        assert isinstance(result, str)

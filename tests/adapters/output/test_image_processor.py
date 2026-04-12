import pytest

from kume.adapters.output.image_processor import ImageProcessor
from kume.ports.output.resource_processor import ResourceProcessorPort


class TestImageProcessor:
    def test_implements_resource_processor_port(self) -> None:
        assert issubclass(ImageProcessor, ResourceProcessorPort)

    @pytest.mark.asyncio
    async def test_returns_coming_soon_message(self) -> None:
        processor = ImageProcessor()
        result = await processor.process(b"fake-image-bytes")
        assert "coming soon" in result.lower()

    @pytest.mark.asyncio
    async def test_message_mentions_food_photos(self) -> None:
        processor = ImageProcessor()
        result = await processor.process(b"\x89PNG\r\n")
        assert "food photos" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_string(self) -> None:
        processor = ImageProcessor()
        result = await processor.process(b"")
        assert isinstance(result, str)

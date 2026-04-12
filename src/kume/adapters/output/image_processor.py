from kume.ports.output.resource_processor import ResourceProcessorPort


class ImageProcessor(ResourceProcessorPort):
    """Stub processor for images — full vision support coming soon."""

    async def process(self, raw_bytes: bytes, *, mime_type: str | None = None) -> str:
        return "[Food image attached — use analyze_food_image tool to analyze]"

from kume.ports.output.resource_processor import ResourceProcessorPort


class ImageProcessor(ResourceProcessorPort):
    """Stub processor for images — full vision support coming soon."""

    async def process(self, raw_bytes: bytes) -> str:
        return "Image processing is coming soon. I'll be able to analyze food photos in a future update."

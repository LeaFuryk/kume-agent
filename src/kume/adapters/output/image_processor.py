from kume.ports.output.resource_processor import ResourceProcessorPort


class ImageProcessor(ResourceProcessorPort):
    """Passes through a signal that an image is attached.

    Actual vision analysis is handled by the tool layer, not the processor.
    The raw bytes are preserved on the Resource object by the batcher.
    """

    async def process(self, raw_bytes: bytes, *, mime_type: str | None = None) -> str:
        return "[Image attached for analysis]"

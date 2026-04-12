from kume.ports.output.resource_processor import ResourceProcessorPort


class UnsupportedMediaType(Exception):
    def __init__(self, mime_type: str) -> None:
        self.mime_type = mime_type
        super().__init__(f"Unsupported media type: {mime_type}")


class IngestionService:
    """Routes raw media bytes to the correct ResourceProcessorPort by mime type."""

    def __init__(self, processors: dict[str, ResourceProcessorPort]) -> None:
        self._processors = processors

    async def process(self, raw_bytes: bytes, mime_type: str) -> str:
        processor = self._processors.get(mime_type)
        if not processor:
            raise UnsupportedMediaType(mime_type)
        return await processor.process(raw_bytes)

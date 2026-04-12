import fitz  # type: ignore[import-untyped]  # PyMuPDF

from kume.ports.output.resource_processor import ResourceProcessorPort


class PDFProcessor(ResourceProcessorPort):
    """Extracts text content from PDF documents using PyMuPDF."""

    async def process(self, raw_bytes: bytes, *, mime_type: str | None = None) -> str:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)

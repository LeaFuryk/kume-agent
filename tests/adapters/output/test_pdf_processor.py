import fitz
import pytest

from kume.adapters.output.pdf_processor import PDFProcessor
from kume.ports.output.resource_processor import ResourceProcessorPort


def _make_pdf(text: str) -> bytes:
    """Create a minimal single-page PDF with the given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_empty_pdf() -> bytes:
    """Create a PDF with one blank page."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPDFProcessor:
    def test_implements_resource_processor_port(self) -> None:
        assert issubclass(PDFProcessor, ResourceProcessorPort)

    @pytest.mark.asyncio
    async def test_extracts_text_from_pdf(self) -> None:
        processor = PDFProcessor()
        pdf_bytes = _make_pdf("Hello nutrition world")
        result = await processor.process(pdf_bytes)
        assert "Hello nutrition world" in result

    @pytest.mark.asyncio
    async def test_empty_pdf_returns_empty_or_whitespace(self) -> None:
        processor = PDFProcessor()
        pdf_bytes = _make_empty_pdf()
        result = await processor.process(pdf_bytes)
        assert result.strip() == ""

    @pytest.mark.asyncio
    async def test_multipage_pdf(self) -> None:
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((72, 72), "Page one content")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Page two content")
        pdf_bytes = doc.tobytes()
        doc.close()

        processor = PDFProcessor()
        result = await processor.process(pdf_bytes)
        assert "Page one content" in result
        assert "Page two content" in result
        # Pages should be separated by double newline
        assert "\n\n" in result

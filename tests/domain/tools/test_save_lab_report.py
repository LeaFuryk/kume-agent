import json

import pytest

from kume.domain.entities import Document, LabMarker
from kume.domain.tools.save_lab_report import LabReportProcessor


class _FakeDocRepo:
    """Minimal DocumentSaver conforming to the Protocol."""

    def __init__(self) -> None:
        self.saved_docs: list[Document] = []

    async def save(self, doc: Document) -> None:
        self.saved_docs.append(doc)


class _FakeMarkerRepo:
    """Minimal LabMarkerSaver conforming to the Protocol."""

    def __init__(self) -> None:
        self.saved_markers: list[list[LabMarker]] = []

    async def save_many(self, markers: list[LabMarker]) -> None:
        self.saved_markers.append(markers)


class _FakeEmbedder:
    """Minimal ChunkEmbedder conforming to the Protocol."""

    def __init__(self) -> None:
        self.embedded: list[tuple[str, str, list[str]]] = []

    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None:
        self.embedded.append((user_id, document_id, chunks))


class _FakeLLM:
    """Minimal LLM conforming to the Protocol (async complete)."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


def _make_processor(
    llm_response: str,
    doc_repo: _FakeDocRepo | None = None,
    marker_repo: _FakeMarkerRepo | None = None,
    embedder: _FakeEmbedder | None = None,
) -> tuple[LabReportProcessor, _FakeDocRepo, _FakeMarkerRepo, _FakeEmbedder]:
    dr = doc_repo or _FakeDocRepo()
    mr = marker_repo or _FakeMarkerRepo()
    em = embedder or _FakeEmbedder()

    processor = LabReportProcessor(doc_repo=dr, marker_repo=mr, embedder=em, llm=_FakeLLM(llm_response))
    return processor, dr, mr, em


@pytest.mark.asyncio
async def test_extracts_markers_and_saves() -> None:
    llm_response = json.dumps(
        [
            {
                "name": "COLESTEROL TOTAL",
                "value": 195.0,
                "unit": "mg/dL",
                "reference_range": "< 200 mg/dL",
                "date": "2025-01-15",
            },
            {
                "name": "GLUCOSA",
                "value": 85.0,
                "unit": "mg/dL",
                "reference_range": "70-100 mg/dL",
                "date": "2025-01-15",
            },
        ]
    )
    processor, doc_repo, marker_repo, embedder = _make_processor(llm_response)

    result = await processor.process(user_id="u1", text="Lab results: Cholesterol 195, Glucose 85")

    assert len(doc_repo.saved_docs) == 1
    assert doc_repo.saved_docs[0].user_id == "u1"
    assert doc_repo.saved_docs[0].type == "lab_report"
    assert len(marker_repo.saved_markers) == 1
    assert len(marker_repo.saved_markers[0]) == 2
    assert marker_repo.saved_markers[0][0].name == "COLESTEROL TOTAL"
    assert len(embedder.embedded) == 1
    assert "2 markers" in result


@pytest.mark.asyncio
async def test_handles_invalid_llm_json() -> None:
    processor, doc_repo, marker_repo, embedder = _make_processor("Sorry, I couldn't parse that.")

    result = await processor.process(user_id="u1", text="Some unstructured text")

    assert len(doc_repo.saved_docs) == 1
    assert len(marker_repo.saved_markers) == 0
    assert len(embedder.embedded) == 1
    assert "no markers extracted" in result.lower()


@pytest.mark.asyncio
async def test_handles_null_date() -> None:
    llm_response = json.dumps(
        [
            {"name": "HEMOGLOBINA", "value": 14.5, "unit": "g/dL", "reference_range": "12-16 g/dL", "date": None},
        ]
    )
    processor, _, marker_repo, _ = _make_processor(llm_response)

    await processor.process(user_id="u1", text="Hemoglobin 14.5")

    assert len(marker_repo.saved_markers) == 1
    assert marker_repo.saved_markers[0][0].name == "HEMOGLOBINA"
    assert marker_repo.saved_markers[0][0].date is not None


@pytest.mark.asyncio
async def test_chunks_long_text() -> None:
    processor, _, _, embedder = _make_processor("[]")

    await processor.process(user_id="u1", text="A" * 2500)

    assert len(embedder.embedded) == 1
    assert len(embedder.embedded[0][2]) == 3

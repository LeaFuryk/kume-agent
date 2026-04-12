import json

import pytest

from kume.domain.entities import Document, LabMarker
from kume.domain.tools.save_lab_report import LabReportProcessor


class _FakeDocRepo:
    def __init__(self) -> None:
        self.saved_docs: list[Document] = []

    async def save(self, doc: Document) -> None:
        self.saved_docs.append(doc)


class _FakeMarkerRepo:
    def __init__(self) -> None:
        self.saved_markers: list[list[LabMarker]] = []

    async def save_many(self, markers: list[LabMarker]) -> None:
        self.saved_markers.append(markers)


class _FakeMarkerReader:
    """Returns pre-existing markers only (simulates fetching history before saving)."""

    def __init__(self, existing: list[LabMarker] | None = None) -> None:
        self._existing = existing or []

    async def get_by_user(self, user_id: str) -> list[LabMarker]:
        return list(self._existing)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.embedded: list[tuple[str, str, list[str]]] = []

    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None:
        self.embedded.append((user_id, document_id, chunks))


class _FakeLLM:
    """Returns different responses per call (extraction, then analysis)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self._next()

    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> str:
        return self._next()

    def _next(self) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]


def _make_processor(
    extraction_responses: str | list[str],
    analysis_response: str = "Analysis: looking good!",
    doc_repo: _FakeDocRepo | None = None,
    marker_repo: _FakeMarkerRepo | None = None,
    embedder: _FakeEmbedder | None = None,
    existing_markers: list[LabMarker] | None = None,
) -> tuple[LabReportProcessor, _FakeDocRepo, _FakeMarkerRepo, _FakeEmbedder]:
    dr = doc_repo or _FakeDocRepo()
    mr = marker_repo or _FakeMarkerRepo()
    reader = _FakeMarkerReader(existing_markers)
    em = embedder or _FakeEmbedder()
    # Build response list: extraction(s) first, then analysis
    if isinstance(extraction_responses, str):
        responses = [extraction_responses, analysis_response]
    else:
        responses = list(extraction_responses) + [analysis_response]
    llm = _FakeLLM(responses)
    processor = LabReportProcessor(doc_repo=dr, marker_repo=mr, marker_reader=reader, embedder=em, llm=llm)
    return processor, dr, mr, em


@pytest.mark.asyncio
async def test_extracts_markers_and_returns_analysis() -> None:
    extraction = json.dumps(
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
    processor, doc_repo, marker_repo, embedder = _make_processor(extraction, "Your cholesterol is borderline.")

    result = await processor.process(user_id="u1", texts=["Lab results: Cholesterol 195, Glucose 85"])

    assert len(doc_repo.saved_docs) == 1
    assert len(marker_repo.saved_markers) == 1
    assert len(marker_repo.saved_markers[0]) == 2
    assert len(embedder.embedded) == 1
    # Result is the analysis, not the extraction summary
    assert "cholesterol" in result.lower()


@pytest.mark.asyncio
async def test_single_text_string_backward_compat() -> None:
    """Passing a single string (not a list) still works via backward compat."""
    extraction = json.dumps(
        [
            {
                "name": "HEMOGLOBINA",
                "value": 14.5,
                "unit": "g/dL",
                "reference_range": "12-16 g/dL",
                "date": "2025-01-15",
            },
        ]
    )
    processor, doc_repo, marker_repo, embedder = _make_processor(extraction, "Hemoglobin is good.")

    # Pass a plain string, not a list
    result = await processor.process(user_id="u1", texts="Hemoglobin 14.5")

    assert len(doc_repo.saved_docs) == 1
    assert len(marker_repo.saved_markers) == 1
    assert len(embedder.embedded) == 1
    assert "hemoglobin" in result.lower()


@pytest.mark.asyncio
async def test_handles_invalid_llm_json() -> None:
    processor, doc_repo, marker_repo, embedder = _make_processor("Sorry, I couldn't parse that.")

    result = await processor.process(user_id="u1", texts=["Some unstructured text"])

    assert len(doc_repo.saved_docs) == 1
    assert len(marker_repo.saved_markers) == 0
    assert len(embedder.embedded) == 1
    assert "no markers" in result.lower()


@pytest.mark.asyncio
async def test_handles_null_date() -> None:
    extraction = json.dumps(
        [
            {"name": "HEMOGLOBINA", "value": 14.5, "unit": "g/dL", "reference_range": "12-16 g/dL", "date": None},
        ]
    )
    processor, _, marker_repo, _ = _make_processor(extraction)

    await processor.process(user_id="u1", texts=["Hemoglobin 14.5"])

    assert len(marker_repo.saved_markers) == 1
    assert marker_repo.saved_markers[0][0].name == "HEMOGLOBINA"
    assert marker_repo.saved_markers[0][0].date is not None


@pytest.mark.asyncio
async def test_chunks_long_text() -> None:
    processor, _, _, embedder = _make_processor("[]")

    await processor.process(user_id="u1", texts=["A" * 2500])

    assert len(embedder.embedded) == 1
    assert len(embedder.embedded[0][2]) == 3


@pytest.mark.asyncio
async def test_multiple_texts_parallel_extraction() -> None:
    """Multiple texts are each extracted, saved as separate docs, and embedded separately."""
    extraction1 = json.dumps(
        [
            {
                "name": "COLESTEROL TOTAL",
                "value": 195.0,
                "unit": "mg/dL",
                "reference_range": "< 200 mg/dL",
                "date": "2025-01-15",
            },
        ]
    )
    extraction2 = json.dumps(
        [
            {
                "name": "GLUCOSA",
                "value": 90.0,
                "unit": "mg/dL",
                "reference_range": "70-100 mg/dL",
                "date": "2025-03-01",
            },
        ]
    )
    processor, doc_repo, marker_repo, embedder = _make_processor(
        [extraction1, extraction2], "Comparative analysis: both reports look good."
    )

    result = await processor.process(
        user_id="u1",
        texts=["Report 1: Cholesterol 195", "Report 2: Glucose 90"],
    )

    # 2 separate documents saved
    assert len(doc_repo.saved_docs) == 2
    assert doc_repo.saved_docs[0].filename == "lab_report_1.txt"
    assert doc_repo.saved_docs[1].filename == "lab_report_2.txt"

    # 2 separate save_many calls (one per doc)
    assert len(marker_repo.saved_markers) == 2
    assert len(marker_repo.saved_markers[0]) == 1
    assert marker_repo.saved_markers[0][0].name == "COLESTEROL TOTAL"
    assert len(marker_repo.saved_markers[1]) == 1
    assert marker_repo.saved_markers[1][0].name == "GLUCOSA"

    # 2 separate embedding calls
    assert len(embedder.embedded) == 2

    # Result is the comparative analysis
    assert "comparative" in result.lower()


@pytest.mark.asyncio
async def test_single_text_with_history_compares_previous() -> None:
    """A single text with existing history triggers a comparison analysis."""
    from datetime import UTC, datetime

    from kume.domain.entities import LabMarker

    existing = [
        LabMarker(
            id="old-marker-1",
            document_id="old-doc-1",
            user_id="u1",
            name="COLESTEROL TOTAL",
            value=220.0,
            unit="mg/dL",
            reference_range="< 200 mg/dL",
            date=datetime(2024, 6, 15, tzinfo=UTC),
        ),
    ]

    extraction = json.dumps(
        [
            {
                "name": "COLESTEROL TOTAL",
                "value": 195.0,
                "unit": "mg/dL",
                "reference_range": "< 200 mg/dL",
                "date": "2025-01-15",
            },
        ]
    )
    processor, doc_repo, marker_repo, _ = _make_processor(
        extraction,
        "Great improvement in cholesterol!",
        existing_markers=existing,
    )

    result = await processor.process(user_id="u1", texts=["Cholesterol 195"])

    assert len(doc_repo.saved_docs) == 1
    assert len(marker_repo.saved_markers) == 1
    assert "improvement" in result.lower()


@pytest.mark.asyncio
async def test_single_text_no_history_establishes_baselines() -> None:
    """A single text with no history returns a baseline analysis."""
    extraction = json.dumps(
        [
            {
                "name": "COLESTEROL TOTAL",
                "value": 195.0,
                "unit": "mg/dL",
                "reference_range": "< 200 mg/dL",
                "date": "2025-01-15",
            },
        ]
    )
    processor, _, _, _ = _make_processor(extraction, "Baseline established for cholesterol.")

    result = await processor.process(user_id="u1", texts=["Cholesterol 195"])

    assert "baseline" in result.lower()

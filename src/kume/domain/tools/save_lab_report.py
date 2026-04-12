from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from kume.domain.entities import Document, LabMarker

_EXTRACTION_PROMPT = """\
You are a medical lab report parser. Extract all lab markers from the following text.

Return a JSON array of objects, each with these fields:
- "name": the marker name (e.g. "COLESTEROL TOTAL")
- "value": the numeric value as a float
- "unit": the unit of measurement (e.g. "mg/dL")
- "reference_range": the reference range as a string (e.g. "< 200 mg/dL")
- "date": the date of the test in ISO 8601 format (YYYY-MM-DD), or null if unknown

Return ONLY the JSON array, no other text.

Lab report text:
{text}
"""


class DocumentSaver(Protocol):
    async def save(self, doc: Document) -> None: ...


class LabMarkerSaver(Protocol):
    async def save_many(self, markers: list[LabMarker]) -> None: ...


class ChunkEmbedder(Protocol):
    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None: ...


class LLM(Protocol):
    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class LabReportProcessor:
    """Processes lab report text: extracts markers via LLM, persists document and markers, embeds chunks.

    All dependencies injected via constructor using Protocol types,
    keeping the domain layer independent of port ABCs and frameworks.
    """

    def __init__(
        self,
        doc_repo: DocumentSaver,
        marker_repo: LabMarkerSaver,
        embedder: ChunkEmbedder,
        llm: LLM,
    ) -> None:
        self._doc_repo = doc_repo
        self._marker_repo = marker_repo
        self._embedder = embedder
        self._llm = llm

    async def process(self, user_id: str, text: str) -> str:
        """Parse a lab report, save document + markers, embed chunks. Returns summary."""
        doc_id = str(uuid4())

        raw_response = await self._llm.complete(
            system_prompt="You are a medical lab report parser.",
            user_prompt=_EXTRACTION_PROMPT.format(text=text),
        )

        markers = _parse_markers(raw_response, doc_id, user_id)

        if markers:
            marker_names = ", ".join(m.name for m in markers)
            summary = f"Lab report with {len(markers)} markers: {marker_names}"
        else:
            summary = "Lab report (no markers extracted)"

        doc = Document(
            id=doc_id,
            user_id=user_id,
            type="lab_report",
            filename="lab_report.txt",
            summary=summary,
            ingested_at=datetime.now(tz=UTC),
        )
        await self._doc_repo.save(doc)

        if markers:
            await self._marker_repo.save_many(markers)

        chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]
        await self._embedder.embed_chunks(user_id, doc_id, chunks)

        return summary


def _parse_markers(raw_response: str, doc_id: str, user_id: str) -> list[LabMarker]:
    """Parse LLM JSON response into LabMarker entities. Returns empty list on failure."""
    markers: list[LabMarker] = []
    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, list):
            for item in parsed:
                marker_date_str = item.get("date")
                if marker_date_str:
                    marker_date = datetime.fromisoformat(marker_date_str).replace(tzinfo=UTC)
                else:
                    marker_date = datetime.now(tz=UTC)

                markers.append(
                    LabMarker(
                        id=str(uuid4()),
                        document_id=doc_id,
                        user_id=user_id,
                        name=item["name"],
                        value=float(item["value"]),
                        unit=item["unit"],
                        reference_range=item.get("reference_range", ""),
                        date=marker_date,
                    )
                )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    return markers

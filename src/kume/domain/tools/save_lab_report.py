from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from kume.domain.entities import Document, LabMarker

_EXTRACTION_PROMPT = """\
Extract ALL lab markers from the following lab report text.
Include EVERY marker found, even if the reference range is unclear.

Lab report text:
{text}
"""

_MARKERS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "markers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Marker name, e.g. COLESTEROL TOTAL"},
                    "value": {"type": "number", "description": "Numeric result value"},
                    "unit": {
                        "type": ["string", "null"],
                        "description": "Unit of measurement, e.g. mg/dL, or null for ratios",
                    },
                    "reference_range": {
                        "type": ["string", "null"],
                        "description": "Reference range, e.g. < 200 mg/dL, or null if not available",
                    },
                    "date": {"type": ["string", "null"], "description": "Test date in YYYY-MM-DD format, or null"},
                },
                "required": ["name", "value", "unit", "reference_range", "date"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["markers"],
    "additionalProperties": False,
}

_ANALYSIS_PROMPT = """\
You are a nutrition health analyst. Analyze the following lab markers and provide \
a clear, actionable summary for the user.

{history_section}

Current markers:
{current_markers}

Provide:
1. A brief overview of the current results — what's normal, what's out of range
2. {comparison_instruction}
3. Key recommendations (diet, lifestyle) based on the findings
4. An encouraging note about next steps

Keep it concise, use bullet points, and be encouraging about progress.
Respond in the same language as the lab report data.
"""

_COMPARATIVE_INSTRUCTION = (
    "Compare results across the different reports submitted — identify "
    "discrepancies, trends across reports, and any markers that appear in "
    "multiple documents. Also compare with historical data if available."
)


class DocumentSaver(Protocol):
    async def save(self, doc: Document) -> None: ...


class LabMarkerSaver(Protocol):
    async def save_many(self, markers: list[LabMarker]) -> None: ...


class LabMarkerReader(Protocol):
    async def get_by_user(self, user_id: str) -> list[LabMarker]: ...


class ChunkEmbedder(Protocol):
    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None: ...


class LLM(Protocol):
    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...
    async def complete_json(self, system_prompt: str, user_prompt: str, schema: dict) -> str: ...


class LabReportProcessor:
    """Processes lab report text: extracts markers, saves them, fetches history,
    and produces a comparative analysis using the LLM.

    All dependencies injected via constructor using Protocol types.
    """

    def __init__(
        self,
        doc_repo: DocumentSaver,
        marker_repo: LabMarkerSaver,
        marker_reader: LabMarkerReader,
        embedder: ChunkEmbedder,
        llm: LLM,
    ) -> None:
        self._doc_repo = doc_repo
        self._marker_repo = marker_repo
        self._marker_reader = marker_reader
        self._embedder = embedder
        self._llm = llm

    async def process(self, user_id: str, texts: list[str] | str) -> str:
        """Parse lab report(s), save markers, compare with history, return analysis.

        Accepts a list of texts (one per document) or a single text string
        for backward compatibility.
        """
        # Backward compat: wrap a single string in a list
        if isinstance(texts, str):
            texts = [texts]

        # 1. Fetch previous markers BEFORE saving new ones
        previous_markers = await self._marker_reader.get_by_user(user_id)

        # 2. Extract markers from EACH text in parallel
        async def _extract(text: str) -> str:
            return await self._llm.complete_json(
                system_prompt="You are a medical lab report parser.",
                user_prompt=_EXTRACTION_PROMPT.format(text=text),
                schema=_MARKERS_SCHEMA,
            )

        raw_responses = await asyncio.gather(*[_extract(t) for t in texts])

        # 3. Save each text as a separate Document with its own markers
        all_new_markers: list[LabMarker] = []
        for idx, (text, raw_response) in enumerate(zip(texts, raw_responses, strict=True)):
            doc_id = str(uuid4())
            new_markers = _parse_markers(raw_response, doc_id, user_id)

            # Save document
            if new_markers:
                marker_names = ", ".join(m.name for m in new_markers)
                summary = f"Lab report with {len(new_markers)} markers: {marker_names}"
            else:
                summary = "Lab report (no markers extracted)"

            doc = Document(
                id=doc_id,
                user_id=user_id,
                type="lab_report",
                filename=f"lab_report_{idx + 1}.txt",
                summary=summary,
                ingested_at=datetime.now(tz=UTC),
            )
            await self._doc_repo.save(doc)

            # Save markers
            if new_markers:
                await self._marker_repo.save_many(new_markers)

            # Embed chunks
            chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]
            await self._embedder.embed_chunks(user_id, doc_id, chunks)

            all_new_markers.extend(new_markers)

        # 4. Generate analysis
        is_multiple = len(texts) > 1
        analysis = await self._generate_analysis(all_new_markers, previous_markers, comparative=is_multiple)

        return analysis

    async def _generate_analysis(
        self,
        current: list[LabMarker],
        previous: list[LabMarker],
        *,
        comparative: bool = False,
    ) -> str:
        """Ask the LLM to analyze current markers and compare with history.

        Args:
            current: Markers extracted from the new report(s).
            previous: Markers that existed before this batch.
            comparative: True when multiple reports were submitted at once.
        """
        if not current:
            return "No markers could be extracted from the lab report."

        current_text = "\n".join(
            f"- {m.name}: {m.value} {m.unit} (ref: {m.reference_range}) [{m.date.strftime('%Y-%m-%d')}]"
            for m in current
        )

        if comparative:
            # Multiple reports submitted at once
            comparison_instruction = _COMPARATIVE_INSTRUCTION
            if previous:
                history_text = "\n".join(
                    f"- {m.name}: {m.value} {m.unit} [{m.date.strftime('%Y-%m-%d')}]" for m in previous
                )
                history_section = f"Previous markers (history):\n{history_text}"
            else:
                history_section = "No previous lab results on file — these are the first reports."
        elif previous:
            history_text = "\n".join(
                f"- {m.name}: {m.value} {m.unit} [{m.date.strftime('%Y-%m-%d')}]" for m in previous
            )
            history_section = f"Previous markers (history):\n{history_text}"
            comparison_instruction = (
                "Compare current vs previous results — identify trends "
                "(improving, worsening, stable), celebrate progress, flag concerns"
            )
        else:
            history_section = "No previous lab results on file — this is the first report."
            comparison_instruction = "Since this is the first report, establish baselines and note what to watch"

        prompt = _ANALYSIS_PROMPT.format(
            history_section=history_section,
            current_markers=current_text,
            comparison_instruction=comparison_instruction,
        )

        return await self._llm.complete(
            system_prompt="You are a nutrition health analyst helping a user understand their lab results.",
            user_prompt=prompt,
        )


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response that may be wrapped in markdown code blocks."""
    import re

    # Try to find JSON in ```json ... ``` or ``` ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find a JSON array directly
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _parse_markers(raw_response: str, doc_id: str, user_id: str) -> list[LabMarker]:
    """Parse LLM JSON response into LabMarker entities. Returns empty list on failure."""
    markers: list[LabMarker] = []
    cleaned = _extract_json(raw_response)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return markers

    # Handle both {"markers": [...]} (structured output) and bare [...] (legacy)
    if isinstance(parsed, dict) and "markers" in parsed:
        parsed = parsed["markers"]

    if isinstance(parsed, list):
        for item in parsed:
            try:
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
                        name=item.get("name") or "Unknown",
                        value=float(item.get("value", 0)),
                        unit=item.get("unit") or "",
                        reference_range=item.get("reference_range") or "",
                        date=marker_date,
                    )
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
    return markers

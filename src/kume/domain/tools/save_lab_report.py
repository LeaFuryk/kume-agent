from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from kume.domain.entities import Document, LabMarker

_EXTRACTION_PROMPT = """\
Extract ALL lab markers from the following lab report text.

Return a JSON array. Each object must have exactly these fields:
{{"name": "MARKER NAME", "value": 123.4, "unit": "mg/dL", "reference_range": "< 200 mg/dL", "date": "YYYY-MM-DD"}}

Rules:
- "value" must be a number (float), not a string
- "date" is the test date in ISO 8601 format, or null if not found
- Include EVERY marker found in the text, even if the reference range is unclear
- Return ONLY the JSON array — no markdown, no commentary, no code blocks

Lab report text:
{text}
"""

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

    async def process(self, user_id: str, text: str) -> str:
        """Parse lab report(s), save markers, compare with history, return analysis."""
        doc_id = str(uuid4())

        # 1. Extract markers from the text
        raw_response = await self._llm.complete(
            system_prompt="You are a medical lab report parser.",
            user_prompt=_EXTRACTION_PROMPT.format(text=text),
        )
        new_markers = _parse_markers(raw_response, doc_id, user_id)

        # 2. Save document
        if new_markers:
            marker_names = ", ".join(m.name for m in new_markers)
            summary = f"Lab report with {len(new_markers)} markers: {marker_names}"
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

        # 3. Save new markers
        if new_markers:
            await self._marker_repo.save_many(new_markers)

        # 4. Embed chunks
        chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]
        await self._embedder.embed_chunks(user_id, doc_id, chunks)

        # 5. Fetch ALL markers (including just-saved ones) for analysis
        all_markers = await self._marker_reader.get_by_user(user_id)

        # 6. Separate previous vs current for comparison
        current_ids = {m.id for m in new_markers}
        previous_markers = [m for m in all_markers if m.id not in current_ids]

        # 7. Generate analysis via LLM
        analysis = await self._generate_analysis(new_markers, previous_markers)

        return analysis

    async def _generate_analysis(self, current: list[LabMarker], previous: list[LabMarker]) -> str:
        """Ask the LLM to analyze current markers and compare with history."""
        if not current:
            return "No markers could be extracted from the lab report."

        current_text = "\n".join(
            f"- {m.name}: {m.value} {m.unit} (ref: {m.reference_range}) [{m.date.strftime('%Y-%m-%d')}]"
            for m in current
        )

        if previous:
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
                        name=item["name"],
                        value=float(item["value"]),
                        unit=item["unit"],
                        reference_range=item.get("reference_range", ""),
                        date=marker_date,
                    )
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
    return markers

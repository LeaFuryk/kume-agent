import json

import pytest

from kume.adapters.tools.save_lab_report import SaveLabReportTool
from tests.adapters.tools.conftest import (
    FakeDocumentRepository,
    FakeEmbeddingRepository,
    FakeLabMarkerRepository,
    FakeLLMPort,
)


class TestSaveLabReportTool:
    def _make_tool(self, llm_response: str = "[]") -> SaveLabReportTool:
        llm = FakeLLMPort(response_text=llm_response)
        doc_repo = FakeDocumentRepository()
        marker_repo = FakeLabMarkerRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveLabReportTool(llm=llm, doc_repo=doc_repo, marker_repo=marker_repo, embedding_repo=embedding_repo)
        return tool

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "save_lab_report"
        assert "lab report" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_extracts_markers_from_llm_response(self) -> None:
        markers_json = json.dumps(
            [
                {
                    "name": "GLUCOSA",
                    "value": 90.0,
                    "unit": "mg/dL",
                    "reference_range": "70-100 mg/dL",
                    "date": "2025-03-01",
                }
            ]
        )
        llm = FakeLLMPort(response_text=markers_json)
        doc_repo = FakeDocumentRepository()
        marker_repo = FakeLabMarkerRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveLabReportTool(llm=llm, doc_repo=doc_repo, marker_repo=marker_repo, embedding_repo=embedding_repo)
        result = await tool.ainvoke({"user_id": "u1", "text": "Glucose: 90 mg/dL"})

        assert "1 markers" in result or "1 marker" in result
        assert "GLUCOSA" in result
        assert len(doc_repo.saved_docs) == 1
        assert len(marker_repo.saved_markers) == 1
        assert len(embedding_repo.embedded) == 1

    @pytest.mark.asyncio
    async def test_handles_invalid_llm_json_gracefully(self) -> None:
        llm = FakeLLMPort(response_text="not json")
        doc_repo = FakeDocumentRepository()
        marker_repo = FakeLabMarkerRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveLabReportTool(llm=llm, doc_repo=doc_repo, marker_repo=marker_repo, embedding_repo=embedding_repo)
        result = await tool.ainvoke({"user_id": "u1", "text": "Some text"})

        assert "no markers" in result.lower()
        assert len(doc_repo.saved_docs) == 1
        assert len(marker_repo.saved_markers) == 0

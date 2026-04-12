import json

import pytest

from kume.adapters.tools.save_lab_report import SaveLabReportTool
from kume.infrastructure.request_context import current_user_id
from tests.adapters.tools.conftest import (
    FakeDocumentRepository,
    FakeEmbeddingRepository,
    FakeLabMarkerRepository,
    FakeLLMPort,
)


class TestSaveLabReportTool:
    def _make_tool(self, llm_response: str = "[]", user_id: str = "u1") -> SaveLabReportTool:
        llm = FakeLLMPort(response_text=llm_response)
        doc_repo = FakeDocumentRepository()
        marker_repo = FakeLabMarkerRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveLabReportTool(llm=llm, doc_repo=doc_repo, marker_repo=marker_repo, embedding_repo=embedding_repo)
        current_user_id.set(user_id)
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
        current_user_id.set("u1")
        result = await tool.ainvoke({"text": "Glucose: 90 mg/dL"})

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
        current_user_id.set("u1")
        result = await tool.ainvoke({"text": "Some text"})

        assert "no markers" in result.lower()
        assert len(doc_repo.saved_docs) == 1
        assert len(marker_repo.saved_markers) == 0

    @pytest.mark.asyncio
    async def test_errors_when_user_id_not_set(self) -> None:
        llm = FakeLLMPort(response_text="[]")
        doc_repo = FakeDocumentRepository()
        marker_repo = FakeLabMarkerRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveLabReportTool(llm=llm, doc_repo=doc_repo, marker_repo=marker_repo, embedding_repo=embedding_repo)
        current_user_id.set(None)
        result = await tool.ainvoke({"text": "Some text"})
        assert "Error" in result
        assert len(doc_repo.saved_docs) == 0

    def test_user_id_not_in_args_schema(self) -> None:
        tool = self._make_tool()
        schema_fields = tool.args_schema.model_fields
        assert "user_id" not in schema_fields

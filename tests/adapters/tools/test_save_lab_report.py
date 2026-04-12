import json

import pytest

from kume.adapters.tools.save_lab_report import SaveLabReportTool
from kume.infrastructure.request_context import RequestContext, _current, set_context
from tests.adapters.tools.conftest import (
    FakeDocumentRepository,
    FakeEmbeddingRepository,
    FakeLabMarkerRepository,
    FakeLLMPort,
)


class TestSaveLabReportTool:
    def _make_tool(self, llm_response: str | list[str] = "[]", user_id: str = "u1") -> SaveLabReportTool:
        llm = FakeLLMPort(response_text=llm_response)
        doc_repo = FakeDocumentRepository()
        marker_repo = FakeLabMarkerRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveLabReportTool(llm=llm, doc_repo=doc_repo, marker_repo=marker_repo, embedding_repo=embedding_repo)
        set_context(RequestContext(user_id=user_id, telegram_id=1, language="en"))
        return tool

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "save_lab_report"
        assert "lab report" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_extracts_markers_and_returns_analysis(self) -> None:
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
        # First call: extraction, second call: analysis
        llm = FakeLLMPort(response_text=[markers_json, "Your glucose is within normal range!"])
        doc_repo = FakeDocumentRepository()
        marker_repo = FakeLabMarkerRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveLabReportTool(llm=llm, doc_repo=doc_repo, marker_repo=marker_repo, embedding_repo=embedding_repo)
        set_context(RequestContext(user_id="u1", telegram_id=1, language="en"))
        result = await tool.ainvoke({"texts": ["Glucose: 90 mg/dL"]})

        # Result is the analysis, not extraction summary
        assert "glucose" in result.lower()
        assert len(doc_repo.saved_docs) == 1
        assert len(marker_repo.saved_markers) == 1
        assert len(embedding_repo.embedded) == 1

    @pytest.mark.asyncio
    async def test_multiple_texts_produce_multiple_documents(self) -> None:
        markers_json_1 = json.dumps(
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
        markers_json_2 = json.dumps(
            [
                {
                    "name": "COLESTEROL TOTAL",
                    "value": 195.0,
                    "unit": "mg/dL",
                    "reference_range": "< 200 mg/dL",
                    "date": "2025-03-01",
                }
            ]
        )
        # Two extractions, then one analysis
        llm = FakeLLMPort(response_text=[markers_json_1, markers_json_2, "Comparative analysis done."])
        doc_repo = FakeDocumentRepository()
        marker_repo = FakeLabMarkerRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveLabReportTool(llm=llm, doc_repo=doc_repo, marker_repo=marker_repo, embedding_repo=embedding_repo)
        set_context(RequestContext(user_id="u1", telegram_id=1, language="en"))
        result = await tool.ainvoke({"texts": ["Glucose: 90 mg/dL", "Cholesterol: 195 mg/dL"]})

        assert "comparative" in result.lower()
        assert len(doc_repo.saved_docs) == 2
        assert len(marker_repo.saved_markers) == 2
        assert len(embedding_repo.embedded) == 2

    @pytest.mark.asyncio
    async def test_handles_invalid_llm_json_gracefully(self) -> None:
        llm = FakeLLMPort(response_text="not json")
        doc_repo = FakeDocumentRepository()
        marker_repo = FakeLabMarkerRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveLabReportTool(llm=llm, doc_repo=doc_repo, marker_repo=marker_repo, embedding_repo=embedding_repo)
        set_context(RequestContext(user_id="u1", telegram_id=1, language="en"))
        result = await tool.ainvoke({"texts": ["Some text"]})

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
        _current.set(None)
        result = await tool.ainvoke({"texts": ["Some text"]})
        assert "Error" in result
        assert len(doc_repo.saved_docs) == 0

    def test_user_id_not_in_args_schema(self) -> None:
        tool = self._make_tool()
        schema_fields = tool.args_schema.model_fields
        assert "user_id" not in schema_fields

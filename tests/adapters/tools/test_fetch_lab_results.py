from datetime import datetime

import pytest

from kume.adapters.tools.fetch_lab_results import FetchLabResultsTool
from kume.domain.entities import LabMarker
from kume.infrastructure.request_context import RequestContext, _current, set_context
from tests.adapters.tools.conftest import FakeLabMarkerRepository


def _marker(
    name: str = "COLESTEROL TOTAL",
    value: float = 195.0,
    unit: str = "mg/dL",
    reference_range: str = "< 200 mg/dL",
    date: datetime | None = None,
    user_id: str = "u1",
) -> LabMarker:
    return LabMarker(
        id="m1",
        document_id="d1",
        user_id=user_id,
        name=name,
        value=value,
        unit=unit,
        reference_range=reference_range,
        date=date or datetime(2026, 1, 15),
    )


class TestFetchLabResultsTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_markers(self) -> None:
        repo = FakeLabMarkerRepository()
        await repo.save_many(
            [
                _marker(name="COLESTEROL TOTAL", value=195.0),
                _marker(name="TRIGLICERIDOS", value=150.0, unit="mg/dL", reference_range="< 150 mg/dL"),
            ]
        )
        tool = FetchLabResultsTool(marker_repo=repo)
        set_context(RequestContext(user_id="u1", telegram_id=1, language="en"))

        result = await tool.ainvoke({"query": "show my labs"})

        assert "COLESTEROL TOTAL: 195.0 mg/dL" in result
        assert "TRIGLICERIDOS: 150.0 mg/dL" in result
        assert "(ref: < 200 mg/dL)" in result
        assert "[2026-01-15]" in result

    @pytest.mark.asyncio
    async def test_filters_by_marker_name(self) -> None:
        repo = FakeLabMarkerRepository()
        await repo.save_many(
            [
                _marker(name="COLESTEROL TOTAL", value=195.0),
                _marker(name="TRIGLICERIDOS", value=150.0),
            ]
        )
        tool = FetchLabResultsTool(marker_repo=repo)
        set_context(RequestContext(user_id="u1", telegram_id=1, language="en"))

        result = await tool.ainvoke({"query": "cholesterol", "marker_name": "colesterol"})

        assert "COLESTEROL TOTAL" in result
        assert "TRIGLICERIDOS" not in result

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        repo = FakeLabMarkerRepository()
        tool = FetchLabResultsTool(marker_repo=repo)
        set_context(RequestContext(user_id="u1", telegram_id=1, language="en"))

        result = await tool.ainvoke({"query": "show my labs"})

        assert "No lab results found" in result

    @pytest.mark.asyncio
    async def test_no_user_context_returns_error(self) -> None:
        repo = FakeLabMarkerRepository()
        tool = FetchLabResultsTool(marker_repo=repo)
        _current.set(None)

        result = await tool.ainvoke({"query": "show my labs"})

        assert "Error" in result

    def test_user_id_not_in_schema(self) -> None:
        repo = FakeLabMarkerRepository()
        tool = FetchLabResultsTool(marker_repo=repo)
        schema_fields = tool.args_schema.model_fields
        assert "user_id" not in schema_fields

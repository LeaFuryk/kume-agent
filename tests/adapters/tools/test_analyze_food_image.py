from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kume.adapters.tools.analyze_food_image import AnalyzeFoodImageTool
from kume.domain.context import ContextBuilder
from kume.infrastructure.image_store import ImageStore
from kume.infrastructure.request_context import RequestContext, set_context
from tests.adapters.tools.conftest import FakeVisionPort


def _make_image_store(request_id: str = "req-1", images: list[bytes] | None = None) -> ImageStore:
    store = ImageStore()
    if images is not None:
        store.set_images(request_id, images)
    return store


def _make_tool(
    *,
    vision_response: str = "Nutrition analysis result",
    images: list[bytes] | None = None,
    request_id: str = "req-1",
    context_builder: ContextBuilder | None = None,
) -> AnalyzeFoodImageTool:
    if images is None:
        images = [b"fake-jpeg-bytes"]
    vision = FakeVisionPort(response_text=vision_response)
    store = _make_image_store(request_id, images)
    return AnalyzeFoodImageTool(
        vision=vision,
        image_store=store,
        context_builder=context_builder,
        request_id_key=request_id,
    )


class TestAnalyzeFoodImageTool:
    def test_has_correct_name_and_description(self) -> None:
        tool = _make_tool()
        assert tool.name == "analyze_food_image"
        assert "food" in tool.description.lower()
        assert "nutritional" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_calls_vision_port(self) -> None:
        set_context(RequestContext(user_id="u1", telegram_id=111))
        tool = _make_tool(vision_response="500 kcal meal")

        result = await tool._arun(description="My lunch", image_index=1)

        assert result == "500 kcal meal"
        vision: FakeVisionPort = tool.vision  # type: ignore[assignment]
        assert vision.last_call is not None
        assert vision.last_call["image_bytes"] == b"fake-jpeg-bytes"
        assert vision.last_call["mime_type"] == "image/jpeg"
        assert "My lunch" in str(vision.last_call["user_prompt"])

    @pytest.mark.asyncio
    async def test_builds_context(self) -> None:
        set_context(RequestContext(user_id="u1", telegram_id=111))
        mock_builder = AsyncMock(spec=ContextBuilder)
        mock_builder.build.return_value = "## User Goals\n- Lose weight"

        tool = _make_tool(context_builder=mock_builder)
        result = await tool._arun(description="Dinner plate", image_index=1)

        mock_builder.build.assert_awaited_once_with("u1", "Dinner plate")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_missing_image_returns_error(self) -> None:
        set_context(RequestContext(user_id="u1", telegram_id=111))
        tool = _make_tool(images=[])  # no images stored

        result = await tool._arun(description="Some food", image_index=1)

        assert "Error" in result
        assert "no image found" in result

    @pytest.mark.asyncio
    async def test_no_user_context_returns_error(self) -> None:
        set_context(None)  # type: ignore[arg-type]
        tool = _make_tool()

        result = await tool._arun(description="Some food", image_index=1)

        assert "Error" in result
        assert "user context not set" in result

    def test_user_id_not_in_schema(self) -> None:
        tool = _make_tool()
        schema = tool.args_schema.model_json_schema()
        properties = schema.get("properties", {})
        assert "user_id" not in properties

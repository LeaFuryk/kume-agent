from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kume.adapters.tools.analyze_food_image import (
    _NUTRITION_SCHEMA,
    AnalyzeFoodImageTool,
    _format_nutrition,
)
from kume.domain.context import ContextBuilder
from kume.infrastructure.image_store import ImageStore
from kume.infrastructure.request_context import RequestContext, set_context
from tests.adapters.tools.conftest import FakeVisionPort

_SAMPLE_NUTRITION_JSON = json.dumps(
    {
        "food_description": "Grilled chicken breast with rice, ~300g",
        "calories": 450,
        "protein_g": 40,
        "carbs_g": 50,
        "fat_g": 8,
        "fiber_g": 2,
        "sodium_mg": 600,
        "sugar_g": 1,
        "saturated_fat_g": 2,
        "cholesterol_mg": 120,
        "confidence": 0.85,
        "recommendation": "Great high-protein meal. Consider adding vegetables for more fiber.",
    }
)


def _make_image_store(
    request_id: str = "req-1",
    images: list[bytes] | None = None,
    mime_types: list[str] | None = None,
) -> ImageStore:
    store = ImageStore()
    if images is not None:
        store.set_images(request_id, images, mime_types)
    return store


def _make_tool(
    *,
    vision_response: str = _SAMPLE_NUTRITION_JSON,
    images: list[bytes] | None = None,
    mime_types: list[str] | None = None,
    request_id: str = "req-1",
    context_builder: ContextBuilder | None = None,
) -> AnalyzeFoodImageTool:
    if images is None:
        images = [b"fake-jpeg-bytes"]
    vision = FakeVisionPort(response_text=vision_response)
    store = _make_image_store(request_id, images, mime_types)
    return AnalyzeFoodImageTool(
        vision=vision,
        image_store=store,
        context_builder=context_builder,
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
        tool = _make_tool()

        result = await tool._arun(description="My lunch", image_index=1)

        # Should call analyze_image_json (not analyze_image)
        vision: FakeVisionPort = tool.vision  # type: ignore[assignment]
        assert vision.last_call is not None
        assert vision.last_call["image_bytes"] == b"fake-jpeg-bytes"
        assert vision.last_call["mime_type"] == "image/jpeg"
        assert "My lunch" in str(vision.last_call["user_prompt"])
        assert "json_schema" in vision.last_call

        # Verify formatted output
        assert "Food: Grilled chicken breast" in result
        assert "Confidence: 85%" in result
        assert "Calories: 450 kcal" in result
        assert "Protein: 40g" in result

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


class TestFormatNutrition:
    def test_formats_valid_json(self) -> None:
        result = _format_nutrition(_SAMPLE_NUTRITION_JSON)

        assert "Food: Grilled chicken breast with rice, ~300g" in result
        assert "Confidence: 85%" in result
        assert "Calories: 450 kcal" in result
        assert "Protein: 40g" in result
        assert "Carbs: 50g" in result
        assert "Fat: 8g" in result
        assert "Fiber: 2g" in result
        assert "Sugar: 1g" in result
        assert "Sodium: 600mg" in result
        assert "Saturated fat: 2g" in result
        assert "Cholesterol: 120mg" in result
        assert "Great high-protein meal" in result

    def test_invalid_json_returns_raw_string(self) -> None:
        raw = "This is not valid JSON at all"
        result = _format_nutrition(raw)
        assert result == raw

    def test_no_recommendation_omits_trailing_lines(self) -> None:
        data = {
            "food_description": "Apple",
            "calories": 95,
            "protein_g": 0,
            "carbs_g": 25,
            "fat_g": 0,
            "fiber_g": 4,
            "sodium_mg": 2,
            "sugar_g": 19,
            "saturated_fat_g": 0,
            "cholesterol_mg": 0,
            "confidence": 0.95,
            "recommendation": "",
        }
        result = _format_nutrition(json.dumps(data))

        assert "Food: Apple" in result
        assert "Confidence: 95%" in result
        # No trailing recommendation line
        assert result.strip().endswith("Cholesterol: 0mg")

    def test_format_includes_all_macro_lines(self) -> None:
        """Verify output includes lines for every nutritional metric."""
        result = _format_nutrition(_SAMPLE_NUTRITION_JSON)

        assert "Calories:" in result
        assert "Protein:" in result
        assert "Carbs:" in result
        assert "Fat:" in result
        assert "Fiber:" in result
        assert "Sugar:" in result
        assert "Sodium:" in result
        assert "Saturated fat:" in result
        assert "Cholesterol:" in result

    def test_format_confidence_as_percentage(self) -> None:
        """Verify confidence 0.85 is rendered as '85%'."""
        data = {
            "food_description": "Toast",
            "calories": 100,
            "protein_g": 3,
            "carbs_g": 18,
            "fat_g": 1,
            "fiber_g": 1,
            "sodium_mg": 150,
            "sugar_g": 2,
            "saturated_fat_g": 0,
            "cholesterol_mg": 0,
            "confidence": 0.85,
            "recommendation": "",
        }
        result = _format_nutrition(json.dumps(data))
        assert "Confidence: 85%" in result

    def test_format_zero_values(self) -> None:
        """All-zero nutritional values format without errors."""
        data = {
            "food_description": "Water",
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "fiber_g": 0,
            "sodium_mg": 0,
            "sugar_g": 0,
            "saturated_fat_g": 0,
            "cholesterol_mg": 0,
            "confidence": 0.0,
            "recommendation": "",
        }
        result = _format_nutrition(json.dumps(data))

        assert "Food: Water" in result
        assert "Confidence: 0%" in result
        assert "Calories: 0 kcal" in result
        assert "Protein: 0g" in result
        assert "Carbs: 0g" in result
        assert "Fat: 0g" in result


class TestNutritionSchema:
    """Tests for _NUTRITION_SCHEMA structure and completeness."""

    # Fields from NutritionEstimate entity plus food_description and recommendation
    EXPECTED_FIELDS = {
        "food_description",
        "calories",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "sodium_mg",
        "sugar_g",
        "saturated_fat_g",
        "cholesterol_mg",
        "confidence",
        "recommendation",
    }

    def test_schema_has_all_required_fields(self) -> None:
        """Verify schema properties cover NutritionEstimate fields + food_description + recommendation."""
        properties = set(_NUTRITION_SCHEMA["properties"].keys())
        assert properties == self.EXPECTED_FIELDS

    def test_schema_requires_all_properties(self) -> None:
        """The 'required' list must match 'properties' keys exactly."""
        required = set(_NUTRITION_SCHEMA["required"])
        properties = set(_NUTRITION_SCHEMA["properties"].keys())
        assert required == properties

    def test_schema_disallows_additional_properties(self) -> None:
        """additionalProperties must be False for strict mode."""
        assert _NUTRITION_SCHEMA["additionalProperties"] is False

    def test_valid_sample_matches_schema(self) -> None:
        """A complete sample dict should have all keys required by the schema."""
        sample = {
            "food_description": "Grilled chicken",
            "calories": 350,
            "protein_g": 35,
            "carbs_g": 0,
            "fat_g": 15,
            "fiber_g": 0,
            "sodium_mg": 400,
            "sugar_g": 0,
            "saturated_fat_g": 4,
            "cholesterol_mg": 90,
            "confidence": 0.9,
            "recommendation": "Good protein source.",
        }
        required = set(_NUTRITION_SCHEMA["required"])
        # All required keys present
        assert required.issubset(sample.keys())
        # No extra keys (additionalProperties is False)
        assert set(sample.keys()) == set(_NUTRITION_SCHEMA["properties"].keys())
        # All values match expected types
        for key, prop in _NUTRITION_SCHEMA["properties"].items():
            if prop["type"] == "number":
                assert isinstance(sample[key], (int, float)), f"{key} should be a number"
            elif prop["type"] == "string":
                assert isinstance(sample[key], str), f"{key} should be a string"

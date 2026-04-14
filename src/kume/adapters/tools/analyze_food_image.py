from __future__ import annotations

import asyncio
import json
import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.context import ContextBuilder
from kume.infrastructure.image_store import ImageStore
from kume.infrastructure.request_context import get_context
from kume.ports.output.vision import VisionPort

logger = logging.getLogger(__name__)

_NUTRITION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "food_description": {
            "type": "string",
            "description": "What the food appears to be with estimated portion",
        },
        "calories": {"type": "number", "description": "Estimated calories (kcal)"},
        "protein_g": {"type": "number", "description": "Protein in grams"},
        "carbs_g": {"type": "number", "description": "Carbohydrates in grams"},
        "fat_g": {"type": "number", "description": "Total fat in grams"},
        "fiber_g": {"type": "number", "description": "Fiber in grams"},
        "sodium_mg": {"type": "number", "description": "Sodium in milligrams"},
        "sugar_g": {"type": "number", "description": "Sugar in grams"},
        "saturated_fat_g": {
            "type": "number",
            "description": "Saturated fat in grams",
        },
        "cholesterol_mg": {
            "type": "number",
            "description": "Cholesterol in milligrams",
        },
        "confidence": {"type": "number", "description": "Confidence score 0.0-1.0"},
        "recommendation": {
            "type": "string",
            "description": "Brief health alignment note",
        },
    },
    "required": [
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
    ],
    "additionalProperties": False,
}

NUTRITION_PROMPT = """\
You are a nutrition expert analyzing a food image.

{context}

Analyze the food in this image and provide:
1. What the food appears to be (description and estimated portion)
2. Estimated nutritional breakdown:
   - Calories (kcal)
   - Protein (g)
   - Carbs (g)
   - Fat (g)
   - Fiber (g)
   - Sodium (mg)
   - Sugar (g)
   - Saturated fat (g)
   - Cholesterol (mg)
3. A confidence score (0.0-1.0) for your estimate
4. How this food aligns with the user's health goals (if context available)
5. Brief recommendation

User's message about this food: {description}

Format the nutritional values clearly so they can be used to log this meal.
"""


class AnalyzeFoodImageInput(BaseModel):
    description: str = Field(description="What the user said about the food image")
    image_index: int = Field(description="Which attached image to analyze (1-based)")


class AnalyzeFoodImageTool(BaseTool):
    """Analyzes food photos for nutritional content using vision AI."""

    name: str = "analyze_food_image"
    description: str = (
        "Analyze a food photo for detailed nutritional breakdown (calories, protein, carbs, fat, etc.). "
        "Example: user sends food photo → analyze_food_image(description='what is this?', image_index=1)"
    )
    args_schema: type[BaseModel] = AnalyzeFoodImageInput
    vision: VisionPort = Field(exclude=True)
    context_builder: ContextBuilder | None = Field(default=None, exclude=True)
    image_store: ImageStore = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, description: str, image_index: int) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(description=description, image_index=image_index))

    async def _arun(self, description: str, image_index: int) -> str:
        ctx = get_context()
        if not ctx:
            return "Error: user context not set. Cannot analyze food image."

        # Get image bytes using the current request_id from ImageStore contextvar
        request_id = self.image_store.current_request_id
        image_bytes = self.image_store.get_image(request_id, image_index)
        if image_bytes is None:
            return f"Error: no image found at index {image_index}."

        mime_type = self.image_store.get_mime_type(request_id, image_index)

        # Build user context
        context = ""
        if self.context_builder:
            try:
                context = await self.context_builder.build(ctx.user_id, description)
            except Exception:
                logger.warning("Failed to build context for food image analysis", exc_info=True)

        prompt = NUTRITION_PROMPT.format(context=context, description=description)

        try:
            raw_json = await self.vision.analyze_image_json(
                system_prompt="You are a nutrition expert. Analyze the food in this image and estimate nutritional content.",
                user_prompt=prompt,
                image_bytes=image_bytes,
                mime_type=mime_type,
                json_schema=_NUTRITION_SCHEMA,
            )
            return _format_nutrition(raw_json)
        except Exception:
            logger.exception("Vision API call failed")
            return "Sorry, I couldn't analyze the image right now. Please try again."


def _format_nutrition(raw_json: str) -> str:
    """Format structured JSON nutrition data into a readable string."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json  # fallback: return raw text

    lines = []
    desc = data.get("food_description", "Unknown food")
    confidence = data.get("confidence", 0)
    lines.append(f"Food: {desc}")
    lines.append(f"Confidence: {confidence:.0%}")
    lines.append("")
    lines.append(
        f"Calories: {data.get('calories', 0):.0f} kcal | "
        f"Protein: {data.get('protein_g', 0):.0f}g | "
        f"Carbs: {data.get('carbs_g', 0):.0f}g | "
        f"Fat: {data.get('fat_g', 0):.0f}g"
    )
    lines.append(
        f"Fiber: {data.get('fiber_g', 0):.0f}g | "
        f"Sugar: {data.get('sugar_g', 0):.0f}g | "
        f"Sodium: {data.get('sodium_mg', 0):.0f}mg"
    )
    lines.append(
        f"Saturated fat: {data.get('saturated_fat_g', 0):.0f}g | Cholesterol: {data.get('cholesterol_mg', 0):.0f}mg"
    )
    rec = data.get("recommendation", "")
    if rec:
        lines.append("")
        lines.append(rec)
    return "\n".join(lines)

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.context import ContextBuilder
from kume.infrastructure.image_store import ImageStore
from kume.infrastructure.request_context import get_context
from kume.ports.output.vision import VisionPort

logger = logging.getLogger(__name__)

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
        "Analyze a food photo for detailed nutritional content including calories, "
        "protein, carbs, fat, fiber, sodium, sugar, saturated fat, and cholesterol. "
        "Returns a nutritional breakdown with confidence score and health alignment."
    )
    args_schema: type[BaseModel] = AnalyzeFoodImageInput
    vision: VisionPort = Field(exclude=True)
    context_builder: ContextBuilder | None = Field(default=None, exclude=True)
    image_store: ImageStore = Field(exclude=True)
    request_id_key: str = Field(default="", exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, description: str, image_index: int) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(description=description, image_index=image_index))

    async def _arun(self, description: str, image_index: int) -> str:
        ctx = get_context()
        if not ctx:
            return "Error: user context not set. Cannot analyze food image."

        # Get image bytes
        image_bytes = self.image_store.get_image(self.request_id_key, image_index)
        if image_bytes is None:
            return f"Error: no image found at index {image_index}."

        # Build user context
        context = ""
        if self.context_builder:
            try:
                context = await self.context_builder.build(ctx.user_id, description)
            except Exception:
                logger.warning("Failed to build context for food image analysis", exc_info=True)

        prompt = NUTRITION_PROMPT.format(context=context, description=description)

        return await self.vision.analyze_image(
            system_prompt="You are a nutrition expert analyzing food images.",
            user_prompt=prompt,
            image_bytes=image_bytes,
            mime_type="image/jpeg",
        )

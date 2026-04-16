from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.entities import Meal
from kume.infrastructure.request_context import get_context
from kume.ports.output.repositories import MealRepository


class LogMealInput(BaseModel):
    description: str = Field(description="Description of the meal")
    calories: float = Field(description="Estimated calories (kcal)")
    protein_g: float = Field(description="Protein in grams")
    carbs_g: float = Field(description="Carbohydrates in grams")
    fat_g: float = Field(description="Total fat in grams")
    fiber_g: float = Field(default=0.0, description="Fiber in grams")
    sodium_mg: float = Field(default=0.0, description="Sodium in milligrams")
    sugar_g: float = Field(default=0.0, description="Sugar in grams")
    saturated_fat_g: float = Field(default=0.0, description="Saturated fat in grams")
    cholesterol_mg: float = Field(default=0.0, description="Cholesterol in milligrams")
    confidence: float = Field(default=0.8, description="Confidence of the nutritional estimate (0-1)")
    image_present: bool = Field(default=False, description="Whether this meal was logged from an image")
    logged_at: str | None = Field(default=None, description="When the meal was eaten, ISO format. Defaults to now.")


class LogMealTool(BaseTool):
    """Records a meal with nutritional details for tracking over time."""

    name: str = "log_meal"
    description: str = (
        "Record a meal with estimated nutritional details for tracking. "
        "Use this DIRECTLY when the user describes a meal in text (no image needed). "
        "Estimate all values yourself. "
        "Example: 'I had pizza for lunch' → log_meal(description='pizza', calories=285, protein_g=12, ...)"
    )
    args_schema: type[BaseModel] = LogMealInput
    meal_repo: MealRepository = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        description: str = "",
        calories: float = 0.0,
        protein_g: float = 0.0,
        carbs_g: float = 0.0,
        fat_g: float = 0.0,
        fiber_g: float = 0.0,
        sodium_mg: float = 0.0,
        sugar_g: float = 0.0,
        saturated_fat_g: float = 0.0,
        cholesterol_mg: float = 0.0,
        confidence: float = 0.8,
        image_present: bool = False,
        logged_at: str | None = None,
    ) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(
            self._arun(
                description=description,
                calories=calories,
                protein_g=protein_g,
                carbs_g=carbs_g,
                fat_g=fat_g,
                fiber_g=fiber_g,
                sodium_mg=sodium_mg,
                sugar_g=sugar_g,
                saturated_fat_g=saturated_fat_g,
                cholesterol_mg=cholesterol_mg,
                confidence=confidence,
                image_present=image_present,
                logged_at=logged_at,
            )
        )

    async def _arun(
        self,
        description: str,
        calories: float,
        protein_g: float,
        carbs_g: float,
        fat_g: float,
        fiber_g: float = 0.0,
        sodium_mg: float = 0.0,
        sugar_g: float = 0.0,
        saturated_fat_g: float = 0.0,
        cholesterol_mg: float = 0.0,
        confidence: float = 0.8,
        image_present: bool = False,
        logged_at: str | None = None,
    ) -> str:
        ctx = get_context()
        if not ctx:
            return "Error: user_id not set. Cannot log meal."

        # Clamp values at boundaries — LLM-generated values can be nonsensical
        confidence = max(0.0, min(1.0, confidence))
        calories = max(0.0, calories)
        protein_g = max(0.0, protein_g)
        carbs_g = max(0.0, carbs_g)
        fat_g = max(0.0, fat_g)
        fiber_g = max(0.0, fiber_g)
        sodium_mg = max(0.0, sodium_mg)
        sugar_g = max(0.0, sugar_g)
        saturated_fat_g = max(0.0, saturated_fat_g)
        cholesterol_mg = max(0.0, cholesterol_mg)

        if logged_at:
            try:
                parsed = datetime.fromisoformat(logged_at)
                # Convert to UTC if timezone-aware, assume UTC if naive
                if parsed.tzinfo is not None:
                    meal_time = parsed.astimezone(UTC)
                else:
                    meal_time = parsed.replace(tzinfo=UTC)
            except ValueError:
                return f"Error: invalid timestamp '{logged_at}'. Use ISO format (e.g. 2026-04-13T12:00:00)."
        else:
            meal_time = datetime.now(tz=UTC)

        meal = Meal(
            id=str(uuid4()),
            user_id=ctx.user_id,
            description=description,
            calories=calories,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
            fiber_g=fiber_g,
            sodium_mg=sodium_mg,
            sugar_g=sugar_g,
            saturated_fat_g=saturated_fat_g,
            cholesterol_mg=cholesterol_mg,
            confidence=confidence,
            image_present=image_present,
            logged_at=meal_time,
        )
        await self.meal_repo.save(meal)
        return (
            f"Meal logged: {description} "
            f"({calories:.0f} kcal, {protein_g:.0f}g protein, "
            f"{carbs_g:.0f}g carbs, {fat_g:.0f}g fat)"
        )

from __future__ import annotations

from dataclasses import dataclass

from kume.domain.entities import Goal, Meal


@dataclass(frozen=True)
class NutritionTotals:
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    meal_count: int


def aggregate_nutrition(meals: list[Meal]) -> NutritionTotals:
    if not meals:
        return NutritionTotals(calories=0.0, protein_g=0.0, carbs_g=0.0, fat_g=0.0, fiber_g=0.0, meal_count=0)
    return NutritionTotals(
        calories=sum(m.calories for m in meals),
        protein_g=sum(m.protein_g for m in meals),
        carbs_g=sum(m.carbs_g for m in meals),
        fat_g=sum(m.fat_g for m in meals),
        fiber_g=sum(m.fiber_g for m in meals),
        meal_count=len(meals),
    )


def compare_against_goals(totals: NutritionTotals, goals: list[Goal]) -> str:
    lines = [
        "Daily Summary",
        f"Meals logged: {totals.meal_count}",
        "",
        f"Calories:  {totals.calories:,.0f} kcal",
        f"Protein:   {totals.protein_g:,.0f}g",
        f"Carbs:     {totals.carbs_g:,.0f}g",
        f"Fat:       {totals.fat_g:,.0f}g",
        f"Fiber:     {totals.fiber_g:,.0f}g",
    ]
    if not goals:
        lines.append("")
        lines.append("No nutrition goals set yet. Tell me your targets and I can track progress!")
    else:
        lines.append("")
        lines.append("Active goals:")
        for goal in goals:
            lines.append(f"- {goal.description}")
    return "\n".join(lines)

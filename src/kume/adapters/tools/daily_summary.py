"""Daily nutrition summary tool — replaces the RequestReportTool stub.

Queries today's meals and compares them against the user's active goals,
producing a formatted summary with totals and progress.
"""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.tools import BaseTool

from kume.domain.nutrition_summary import aggregate_nutrition, compare_against_goals
from kume.infrastructure.request_context import get_context as get_request_context
from kume.ports.output.repositories import GoalRepository, MealRepository


class RequestReportTool(BaseTool):
    name: str = "request_report"
    description: str = (
        "Generate a daily nutrition summary for today. Shows total calories, protein, carbs, "
        "fat consumed today vs the user's goals. Call this when the user asks for "
        "a summary, daily report, or 'how did I eat today?'"
    )
    meal_repo: MealRepository
    goal_repo: GoalRepository
    model_config = {"arbitrary_types_allowed": True}

    async def _arun(self, **kwargs: object) -> str:
        ctx = get_request_context()
        if not ctx:
            return "Unable to generate summary — no user context available."

        user_id = ctx.user_id
        now = datetime.now(UTC)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        meals = await self.meal_repo.get_by_user(user_id, since=start_of_day, limit=50)
        goals = await self.goal_repo.get_by_user(user_id, active_only=True)

        if not meals:
            return (
                f"Daily Summary ({now.strftime('%Y-%m-%d')})\n"
                "No meals logged today yet. Send me what you've eaten and I'll track it!"
            )

        totals = aggregate_nutrition(meals)
        return compare_against_goals(totals, goals)

    def _run(self, **kwargs: object) -> str:
        return "This tool must be called asynchronously."

import pytest

from kume.adapters.tools.save_goal import SaveGoalTool
from kume.domain.entities import Goal
from tests.adapters.tools.conftest import FakeGoalRepository


class TestSaveGoalTool:
    def _make_tool(self) -> SaveGoalTool:
        repo = FakeGoalRepository()
        return SaveGoalTool(goal_repo=repo)

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "save_goal"
        assert "goal" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_delegates_to_domain_handler(self) -> None:
        repo = FakeGoalRepository()
        tool = SaveGoalTool(goal_repo=repo)
        result = await tool.ainvoke({"user_id": "u1", "description": "Lose weight"})
        assert "Goal saved" in result
        assert "Lose weight" in result
        assert len(repo.saved_goals) == 1
        assert repo.saved_goals[0].user_id == "u1"

    @pytest.mark.asyncio
    async def test_creates_goal_with_correct_fields(self) -> None:
        repo = FakeGoalRepository()
        tool = SaveGoalTool(goal_repo=repo)
        await tool.ainvoke({"user_id": "u42", "description": "Eat more greens"})
        goal = repo.saved_goals[0]
        assert isinstance(goal, Goal)
        assert goal.user_id == "u42"
        assert goal.description == "Eat more greens"
        assert goal.id  # UUID generated

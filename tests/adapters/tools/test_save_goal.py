import pytest

from kume.adapters.tools.save_goal import SaveGoalTool
from kume.domain.entities import Goal
from kume.infrastructure.request_context import current_user_id
from tests.adapters.tools.conftest import FakeGoalRepository


class TestSaveGoalTool:
    def _make_tool(self, user_id: str = "u1") -> SaveGoalTool:
        repo = FakeGoalRepository()
        tool = SaveGoalTool(goal_repo=repo)
        current_user_id.set(user_id)
        return tool

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "save_goal"
        assert "goal" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_delegates_to_domain_handler(self) -> None:
        repo = FakeGoalRepository()
        tool = SaveGoalTool(goal_repo=repo)
        current_user_id.set("u1")
        result = await tool.ainvoke({"description": "Lose weight"})
        assert "Goal saved" in result
        assert "Lose weight" in result
        assert len(repo.saved_goals) == 1
        assert repo.saved_goals[0].user_id == "u1"

    @pytest.mark.asyncio
    async def test_creates_goal_with_correct_fields(self) -> None:
        repo = FakeGoalRepository()
        tool = SaveGoalTool(goal_repo=repo)
        current_user_id.set("u42")
        await tool.ainvoke({"description": "Eat more greens"})
        goal = repo.saved_goals[0]
        assert isinstance(goal, Goal)
        assert goal.user_id == "u42"
        assert goal.description == "Eat more greens"
        assert goal.id  # UUID generated

    @pytest.mark.asyncio
    async def test_errors_when_user_id_not_set(self) -> None:
        repo = FakeGoalRepository()
        tool = SaveGoalTool(goal_repo=repo)
        current_user_id.set(None)
        result = await tool.ainvoke({"description": "Lose weight"})
        assert "Error" in result
        assert len(repo.saved_goals) == 0

    def test_user_id_not_in_args_schema(self) -> None:
        tool = self._make_tool()
        schema_fields = tool.args_schema.model_fields
        assert "user_id" not in schema_fields

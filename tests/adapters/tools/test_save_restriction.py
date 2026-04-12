import pytest

from kume.adapters.tools.save_restriction import SaveRestrictionTool
from kume.domain.entities import Restriction
from tests.adapters.tools.conftest import FakeRestrictionRepository


class TestSaveRestrictionTool:
    def _make_tool(self, user_id: str = "u1") -> SaveRestrictionTool:
        repo = FakeRestrictionRepository()
        tool = SaveRestrictionTool(restriction_repo=repo)
        tool.set_user_id(user_id)
        return tool

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "save_restriction"
        assert "restriction" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_delegates_to_domain_handler(self) -> None:
        repo = FakeRestrictionRepository()
        tool = SaveRestrictionTool(restriction_repo=repo)
        tool.set_user_id("u1")
        result = await tool.ainvoke(
            {
                "type": "allergy",
                "description": "Shellfish allergy",
            }
        )
        assert "Restriction saved" in result
        assert "Shellfish allergy" in result
        assert len(repo.saved_restrictions) == 1

    @pytest.mark.asyncio
    async def test_creates_restriction_with_correct_fields(self) -> None:
        repo = FakeRestrictionRepository()
        tool = SaveRestrictionTool(restriction_repo=repo)
        tool.set_user_id("u2")
        await tool.ainvoke(
            {
                "type": "diet",
                "description": "Keto diet",
            }
        )
        restriction = repo.saved_restrictions[0]
        assert isinstance(restriction, Restriction)
        assert restriction.user_id == "u2"
        assert restriction.type == "diet"
        assert restriction.description == "Keto diet"

    @pytest.mark.asyncio
    async def test_errors_when_user_id_not_set(self) -> None:
        repo = FakeRestrictionRepository()
        tool = SaveRestrictionTool(restriction_repo=repo)
        result = await tool.ainvoke({"type": "allergy", "description": "Peanuts"})
        assert "Error" in result
        assert len(repo.saved_restrictions) == 0

    def test_user_id_not_in_args_schema(self) -> None:
        tool = self._make_tool()
        schema_fields = tool.args_schema.model_fields
        assert "user_id" not in schema_fields

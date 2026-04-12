import pytest

from kume.adapters.tools.save_restriction import SaveRestrictionTool
from kume.domain.entities import Restriction
from tests.adapters.tools.conftest import FakeRestrictionRepository


class TestSaveRestrictionTool:
    def _make_tool(self) -> SaveRestrictionTool:
        repo = FakeRestrictionRepository()
        return SaveRestrictionTool(restriction_repo=repo)

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "save_restriction"
        assert "restriction" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_delegates_to_domain_handler(self) -> None:
        repo = FakeRestrictionRepository()
        tool = SaveRestrictionTool(restriction_repo=repo)
        result = await tool.ainvoke(
            {
                "user_id": "u1",
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
        await tool.ainvoke(
            {
                "user_id": "u2",
                "type": "diet",
                "description": "Keto diet",
            }
        )
        restriction = repo.saved_restrictions[0]
        assert isinstance(restriction, Restriction)
        assert restriction.user_id == "u2"
        assert restriction.type == "diet"
        assert restriction.description == "Keto diet"

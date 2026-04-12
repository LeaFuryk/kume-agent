import pytest

from kume.adapters.tools.save_user_name import SaveUserNameTool
from kume.infrastructure.request_context import RequestContext, _current, set_context
from tests.adapters.tools.conftest import FakeUserRepository


class TestSaveUserNameTool:
    def _make_tool(self, user_id: str = "u1") -> tuple[SaveUserNameTool, FakeUserRepository]:
        repo = FakeUserRepository()
        tool = SaveUserNameTool(user_repo=repo)
        set_context(RequestContext(user_id=user_id, telegram_id=1, language="en"))
        return tool, repo

    def test_name_and_description(self) -> None:
        tool, _ = self._make_tool()
        assert tool.name == "save_user_name"
        assert "name" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_saves_user_name(self) -> None:
        tool, repo = self._make_tool()
        result = await tool.ainvoke({"name": "Alice"})
        assert "Name saved" in result
        assert "Alice" in result
        assert len(repo.updated_users) == 1
        assert repo.updated_users[0].name == "Alice"

    @pytest.mark.asyncio
    async def test_errors_when_context_not_set(self) -> None:
        repo = FakeUserRepository()
        tool = SaveUserNameTool(user_repo=repo)
        _current.set(None)
        result = await tool.ainvoke({"name": "Bob"})
        assert "Error" in result
        assert len(repo.updated_users) == 0

    def test_user_id_not_in_args_schema(self) -> None:
        tool, _ = self._make_tool()
        schema_fields = tool.args_schema.model_fields
        assert "user_id" not in schema_fields
        assert "name" in schema_fields

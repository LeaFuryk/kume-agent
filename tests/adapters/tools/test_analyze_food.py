from kume.adapters.tools.analyze_food import AnalyzeFoodTool
from tests.adapters.tools.conftest import FakeChatModel


class TestAnalyzeFoodTool:
    def _make_tool(self, response_text: str = "This food is healthy") -> AnalyzeFoodTool:
        llm = FakeChatModel(response_text=response_text)
        return AnalyzeFoodTool(llm=llm)

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "analyze_food"
        assert "food" in tool.description.lower()

    def test_delegates_to_domain_handler(self) -> None:
        tool = self._make_tool("High in fiber")
        result = tool.invoke({"description": "A bowl of oatmeal with berries"})
        assert result == "High in fiber"

    def test_passes_description_through_to_llm(self) -> None:
        tool = self._make_tool("Nutritious meal")
        result = tool.invoke({"description": "Grilled salmon with rice"})
        assert isinstance(result, str)
        assert result == "Nutritious meal"

    def test_returns_llm_response_content(self) -> None:
        tool = self._make_tool("Low calorie option")
        result = tool.invoke({"description": "Garden salad"})
        assert result == "Low calorie option"

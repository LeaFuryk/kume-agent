from kume.adapters.tools.ask_recommendation import AskRecommendationTool
from tests.adapters.tools.conftest import FakeLLMPort


class TestAskRecommendationTool:
    def _make_tool(self, response_text: str = "Eat more vegetables") -> AskRecommendationTool:
        llm = FakeLLMPort(response_text=response_text)
        return AskRecommendationTool(llm=llm)

    def test_name_and_description(self) -> None:
        tool = self._make_tool()
        assert tool.name == "ask_recommendation"
        assert "nutrition" in tool.description.lower()

    def test_delegates_to_domain_handler(self) -> None:
        tool = self._make_tool("Eat more protein")
        result = tool.invoke({"query": "How much protein should I eat?"})
        assert result == "Eat more protein"

    def test_passes_query_through_to_llm(self) -> None:
        tool = self._make_tool("Great advice")
        result = tool.invoke({"query": "What should I eat?"})
        assert isinstance(result, str)
        assert result == "Great advice"

    def test_returns_llm_response_content(self) -> None:
        tool = self._make_tool("Balanced diet recommended")
        result = tool.invoke({"query": "Give me advice"})
        assert result == "Balanced diet recommended"

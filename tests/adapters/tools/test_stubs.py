from kume.adapters.tools.stubs import LogMealTool, RequestReportTool


class TestLogMealTool:
    def test_name_and_description(self) -> None:
        tool = LogMealTool()
        assert tool.name == "log_meal"
        assert "meal" in tool.description.lower()

    def test_returns_coming_soon(self) -> None:
        tool = LogMealTool()
        result = tool.invoke({"query": "I had pasta for lunch"})
        assert "coming soon" in result.lower()


class TestRequestReportTool:
    def test_name_and_description(self) -> None:
        tool = RequestReportTool()
        assert tool.name == "request_report"
        assert "report" in tool.description.lower()

    def test_returns_coming_soon(self) -> None:
        tool = RequestReportTool()
        result = tool.invoke({"query": "weekly nutrition report"})
        assert "coming soon" in result.lower()

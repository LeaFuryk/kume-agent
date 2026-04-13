from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kume.domain.tools import request_report


class StubInput(BaseModel):
    query: str = Field(default="", description="Optional input for the tool")


class RequestReportTool(BaseTool):
    name: str = "request_report"
    description: str = "Generate a nutrition or health report based on tracked data"
    args_schema: type[BaseModel] = StubInput

    def _run(self, query: str = "") -> str:
        return request_report(query=query)

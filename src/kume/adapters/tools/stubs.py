from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kume.domain.tools import ingest_context, log_meal, request_report


class StubInput(BaseModel):
    query: str = Field(default="", description="Optional input for the tool")


class IngestContextTool(BaseTool):
    name: str = "ingest_context"
    description: str = "Process and store health documents, lab results, or diet plans for future reference"
    args_schema: type[BaseModel] = StubInput

    def _run(self, query: str = "") -> str:
        return ingest_context(query=query)


class LogMealTool(BaseTool):
    name: str = "log_meal"
    description: str = "Record a meal entry for tracking nutrition over time"
    args_schema: type[BaseModel] = StubInput

    def _run(self, query: str = "") -> str:
        return log_meal(query=query)


class RequestReportTool(BaseTool):
    name: str = "request_report"
    description: str = "Generate a nutrition or health report based on tracked data"
    args_schema: type[BaseModel] = StubInput

    def _run(self, query: str = "") -> str:
        return request_report(query=query)

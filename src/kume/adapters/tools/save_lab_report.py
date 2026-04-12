from __future__ import annotations

import asyncio

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.tools.save_lab_report import LabReportProcessor
from kume.ports.output.llm import LLMPort
from kume.ports.output.repositories import DocumentRepository, EmbeddingRepository, LabMarkerRepository


class SaveLabReportInput(BaseModel):
    user_id: str = Field(description="The user's unique identifier")
    text: str = Field(description="The raw text of the lab report")


class SaveLabReportTool(BaseTool):
    """LangChain tool that parses and saves lab reports.

    LangChain tool lifecycle:
        The agent calls .invoke()/.ainvoke() (public API).
        LangChain dispatches to _run() (sync) or _arun() (async) internally.
        _run() is a required sync fallback; _arun() is the primary async path.

    Delegates to LabReportProcessor (domain) for the extraction logic.
    """

    name: str = "save_lab_report"
    description: str = "Parse and save a lab report, extracting markers and indexing the content for retrieval"
    args_schema: type[BaseModel] = SaveLabReportInput
    llm: LLMPort = Field(exclude=True)
    doc_repo: DocumentRepository = Field(exclude=True)
    marker_repo: LabMarkerRepository = Field(exclude=True)
    embedding_repo: EmbeddingRepository = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, user_id: str, text: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(self._arun(user_id=user_id, text=text))

    async def _arun(self, user_id: str, text: str) -> str:
        """Primary async entry point — called by LangChain's agent via .ainvoke()."""
        processor = LabReportProcessor(
            doc_repo=self.doc_repo,
            marker_repo=self.marker_repo,
            embedder=self.embedding_repo,
            llm=self.llm,
        )
        return await processor.process(user_id=user_id, text=text)

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.entities import Document
from kume.ports.output.repositories import DocumentRepository, EmbeddingRepository


class SaveHealthContextInput(BaseModel):
    user_id: str = Field(description="The user's unique identifier")
    text: str = Field(description="The health context text to save and index")


class SaveHealthContextTool(BaseTool):
    """LangChain tool that saves and indexes general health context.

    LangChain tool lifecycle:
        The agent calls .invoke()/.ainvoke() (public API).
        LangChain dispatches to _run() (sync) or _arun() (async) internally.
        _run() is a required sync fallback; _arun() is the primary async path.
    """

    name: str = "save_health_context"
    description: str = (
        "Save and index general health context information "
        "(diet plans, medical notes, personal health data like weight, height, activity level)"
    )
    args_schema: type[BaseModel] = SaveHealthContextInput
    doc_repo: DocumentRepository = Field(exclude=True)
    embedding_repo: EmbeddingRepository = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, user_id: str, text: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(self._arun(user_id=user_id, text=text))

    async def _arun(self, user_id: str, text: str) -> str:
        """Primary async entry point — called by LangChain's agent via .ainvoke()."""
        doc_id = str(uuid4())
        doc = Document(
            id=doc_id,
            user_id=user_id,
            type="health_context",
            filename="health_context.txt",
            summary=text[:200],
            ingested_at=datetime.now(tz=UTC),
        )
        await self.doc_repo.save(doc)

        chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]
        await self.embedding_repo.embed_chunks(user_id, doc_id, chunks)

        return f"Health context saved and indexed ({len(chunks)} chunks)"

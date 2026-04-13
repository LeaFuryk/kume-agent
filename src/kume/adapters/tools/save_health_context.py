from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from kume.domain.entities import Document
from kume.infrastructure.request_context import get_context
from kume.ports.output.repositories import DocumentRepository, EmbeddingRepository


class SaveHealthContextInput(BaseModel):
    text: str = Field(description="The health context text to save and index")


class SaveHealthContextTool(BaseTool):
    """LangChain tool that saves and indexes general health context.

    LangChain tool lifecycle:
        The agent calls .invoke()/.ainvoke() (public API).
        LangChain dispatches to _run() (sync) or _arun() (async) internally.
        _run() is a required sync fallback; _arun() is the primary async path.

    User identity:
        The orchestrator sets user_id via contextvars before each request.
        This avoids trusting the LLM to supply user_id.
    """

    name: str = "save_health_context"
    description: str = (
        "Save and index personal health data (weight, height, activity, diet plans, medical notes). "
        "Example: 'I weigh 80kg' → save_health_context(text='Weight: 80kg')"
    )
    args_schema: type[BaseModel] = SaveHealthContextInput
    doc_repo: DocumentRepository = Field(exclude=True)
    embedding_repo: EmbeddingRepository = Field(exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, text: str) -> str:
        """Sync fallback — required by LangChain's BaseTool contract."""
        return asyncio.get_event_loop().run_until_complete(self._arun(text=text))

    async def _arun(self, text: str) -> str:
        """Primary async entry point — called by LangChain's agent via .ainvoke()."""
        ctx = get_context()
        if not ctx:
            return "Error: user_id not set. Cannot save health context."
        doc_id = str(uuid4())
        doc = Document(
            id=doc_id,
            user_id=ctx.user_id,
            type="health_context",
            filename="health_context.txt",
            summary=text[:200],
            ingested_at=datetime.now(tz=UTC),
        )
        await self.doc_repo.save(doc)

        chunks = [text[i : i + 1000] for i in range(0, len(text), 1000)]
        await self.embedding_repo.embed_chunks(ctx.user_id, doc_id, chunks)

        return f"Health context saved and indexed ({len(chunks)} chunks)"

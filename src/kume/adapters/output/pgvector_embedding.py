"""PGVector-backed embedding repository using langchain-postgres.

Uses the SAME PostgreSQL database (with the pgvector extension) that the
application already connects to. The connection string is converted from
async (``postgresql+asyncpg://``) to sync (``postgresql://``) because
``PGVector`` manages its own synchronous SQLAlchemy engine internally.
"""

from __future__ import annotations

import asyncio

from langchain_core.documents import Document as LCDocument
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from kume.ports.output.repositories import EmbeddingRepository


class PGVectorEmbeddingRepository(EmbeddingRepository):
    """Real embedding repository backed by PGVector + OpenAI embeddings."""

    def __init__(self, database_url: str, openai_api_key: str, embedding_model: str) -> None:
        from pydantic import SecretStr

        self._embeddings = OpenAIEmbeddings(model=embedding_model, api_key=SecretStr(openai_api_key))
        # PGVector requires a synchronous connection string.
        # Replace asyncpg driver with psycopg (v3) which is available.
        sync_url = database_url.replace("+asyncpg", "+psycopg")
        self._vector_store = PGVector(
            embeddings=self._embeddings,
            collection_name="kume_documents",
            connection=sync_url,
        )

    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None:
        docs = [
            LCDocument(page_content=chunk, metadata={"user_id": user_id, "document_id": document_id})
            for chunk in chunks
        ]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._vector_store.add_documents, docs)

    async def search(self, user_id: str, query: str, k: int = 5) -> list[str]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._vector_store.similarity_search(query, k=k, filter={"user_id": user_id}),
        )
        return [doc.page_content for doc in results]

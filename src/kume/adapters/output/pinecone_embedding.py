"""Pinecone-backed embedding repository using langchain-pinecone.

Uses Pinecone as the vector store instead of pgvector. Documents are
embedded via OpenAI embeddings and stored in a Pinecone index with
user_id metadata for filtering.
"""

from __future__ import annotations

import asyncio

from langchain_core.documents import Document as LCDocument
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pydantic import SecretStr

from kume.ports.output.repositories import EmbeddingRepository


def _create_vector_store(
    api_key: str, index_name: str, openai_api_key: str, embedding_model: str
) -> PineconeVectorStore:
    """Create Pinecone vector store. Separated for testability."""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=SecretStr(openai_api_key))
    return PineconeVectorStore(index=index, embedding=embeddings)


class PineconeEmbeddingRepository(EmbeddingRepository):
    """Embedding repository backed by Pinecone + OpenAI embeddings.

    The Pinecone connection is lazily initialized on the first call to
    ``embed_chunks`` or ``search``, preventing import-time crashes when
    Pinecone credentials are misconfigured (e.g. during LangGraph Platform
    deployment bootstrapping).
    """

    def __init__(self, api_key: str, index_name: str, openai_api_key: str, embedding_model: str) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._openai_api_key = openai_api_key
        self._embedding_model = embedding_model
        self._vector_store: PineconeVectorStore | None = None

    def _get_store(self) -> PineconeVectorStore:
        """Return the vector store, creating it on first access."""
        if self._vector_store is None:
            self._vector_store = _create_vector_store(
                self._api_key, self._index_name, self._openai_api_key, self._embedding_model
            )
        return self._vector_store

    async def embed_chunks(self, user_id: str, document_id: str, chunks: list[str]) -> None:
        docs = [
            LCDocument(page_content=chunk, metadata={"user_id": user_id, "document_id": document_id})
            for chunk in chunks
        ]
        store = self._get_store()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, store.add_documents, docs)

    async def search(self, user_id: str, query: str, k: int = 5) -> list[str]:
        store = self._get_store()
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: store.similarity_search(query, k=k, filter={"user_id": user_id}),
        )
        return [doc.page_content for doc in results]

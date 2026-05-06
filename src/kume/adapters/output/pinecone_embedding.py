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
    """Embedding repository backed by Pinecone + OpenAI embeddings."""

    def __init__(self, api_key: str, index_name: str, openai_api_key: str, embedding_model: str) -> None:
        self._vector_store = _create_vector_store(api_key, index_name, openai_api_key, embedding_model)

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

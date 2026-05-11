"""Tests for PineconeEmbeddingRepository adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document as LCDocument

from kume.adapters.output.pinecone_embedding import PineconeEmbeddingRepository


@pytest.fixture
def mock_vector_store() -> MagicMock:
    return MagicMock()


@pytest.fixture
def repo(mock_vector_store: MagicMock) -> PineconeEmbeddingRepository:
    repo = PineconeEmbeddingRepository(
        api_key="fake-pinecone-key",
        index_name="fake-index",
        openai_api_key="fake-openai-key",
        embedding_model="text-embedding-3-small",
    )
    # Inject mock directly so _get_store() returns it without calling Pinecone
    repo._vector_store = mock_vector_store
    return repo


async def test_embed_chunks_creates_documents(repo: PineconeEmbeddingRepository, mock_vector_store: MagicMock) -> None:
    await repo.embed_chunks(user_id="user-1", document_id="doc-1", chunks=["chunk A", "chunk B"])

    mock_vector_store.add_documents.assert_called_once()
    docs = mock_vector_store.add_documents.call_args[0][0]
    assert len(docs) == 2
    assert isinstance(docs[0], LCDocument)
    assert docs[0].page_content == "chunk A"
    assert docs[0].metadata == {"user_id": "user-1", "document_id": "doc-1"}
    assert docs[1].page_content == "chunk B"
    assert docs[1].metadata == {"user_id": "user-1", "document_id": "doc-1"}


async def test_search_filters_by_user_id(repo: PineconeEmbeddingRepository, mock_vector_store: MagicMock) -> None:
    mock_vector_store.similarity_search.return_value = [
        LCDocument(page_content="result 1"),
        LCDocument(page_content="result 2"),
    ]

    results = await repo.search(user_id="user-42", query="protein intake", k=3)

    mock_vector_store.similarity_search.assert_called_once_with("protein intake", k=3, filter={"user_id": "user-42"})
    assert results == ["result 1", "result 2"]


async def test_search_empty_results(repo: PineconeEmbeddingRepository, mock_vector_store: MagicMock) -> None:
    mock_vector_store.similarity_search.return_value = []

    results = await repo.search(user_id="user-99", query="nonexistent topic")

    assert results == []


def test_init_does_not_connect_eagerly() -> None:
    """Constructing the repo should NOT call _create_vector_store."""
    with patch(
        "kume.adapters.output.pinecone_embedding._create_vector_store",
    ) as mock_create:
        repo = PineconeEmbeddingRepository(
            api_key="fake-key",
            index_name="fake-index",
            openai_api_key="fake-openai-key",
            embedding_model="text-embedding-3-small",
        )
        mock_create.assert_not_called()
        assert repo._vector_store is None


def test_get_store_creates_on_first_access() -> None:
    """_get_store() calls _create_vector_store exactly once, then caches."""
    with patch(
        "kume.adapters.output.pinecone_embedding._create_vector_store",
        return_value=MagicMock(),
    ) as mock_create:
        repo = PineconeEmbeddingRepository(
            api_key="fake-key",
            index_name="fake-index",
            openai_api_key="fake-openai-key",
            embedding_model="text-embedding-3-small",
        )
        store1 = repo._get_store()
        store2 = repo._get_store()
        mock_create.assert_called_once()
        assert store1 is store2

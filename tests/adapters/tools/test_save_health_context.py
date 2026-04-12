import pytest

from kume.adapters.tools.save_health_context import SaveHealthContextTool
from tests.adapters.tools.conftest import FakeDocumentRepository, FakeEmbeddingRepository


class TestSaveHealthContextTool:
    def test_name_and_description(self) -> None:
        doc_repo = FakeDocumentRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveHealthContextTool(doc_repo=doc_repo, embedding_repo=embedding_repo)
        assert tool.name == "save_health_context"
        assert "health context" in tool.description.lower()

    @pytest.mark.asyncio
    async def test_saves_document_and_embeds(self) -> None:
        doc_repo = FakeDocumentRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveHealthContextTool(doc_repo=doc_repo, embedding_repo=embedding_repo)
        result = await tool.ainvoke(
            {
                "user_id": "u1",
                "text": "Patient has type 2 diabetes.",
            }
        )

        assert "Health context saved" in result
        assert len(doc_repo.saved_docs) == 1
        assert doc_repo.saved_docs[0].user_id == "u1"
        assert len(embedding_repo.embedded) == 1

    @pytest.mark.asyncio
    async def test_chunks_long_text(self) -> None:
        doc_repo = FakeDocumentRepository()
        embedding_repo = FakeEmbeddingRepository()

        tool = SaveHealthContextTool(doc_repo=doc_repo, embedding_repo=embedding_repo)
        long_text = "Z" * 2500
        result = await tool.ainvoke({"user_id": "u1", "text": long_text})

        assert len(embedding_repo.embedded[0][2]) == 3
        assert "3 chunk" in result

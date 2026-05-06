"""One-time migration: pgvector (local) → Pinecone (cloud).

Reads all embedded chunks from the local langchain_pg_embedding table
and re-embeds them into Pinecone using the PineconeEmbeddingRepository.

Usage:
    # 1. Make sure local Postgres is running (docker compose up)
    # 2. Set PINECONE_API_KEY and PINECONE_INDEX in .env
    # 3. Run:
    uv run python scripts/migrate_vectors_to_pinecone.py

The script reads chunks from the LOCAL database (hardcoded docker-compose URL)
and writes them to the CLOUD Pinecone index (from .env).
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


LOCAL_DB_URL = "postgresql://kume:kume@localhost:5432/kume"


def read_chunks_from_pgvector() -> list[dict[str, str]]:
    """Read all chunks with metadata from local pgvector."""
    engine = create_engine(LOCAL_DB_URL)
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT
                    e.document as content,
                    e.cmetadata->>'user_id' as user_id,
                    e.cmetadata->>'document_id' as document_id
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = 'kume_documents'
                ORDER BY e.cmetadata->>'document_id', e.id
            """)
        ).fetchall()
    engine.dispose()

    chunks = []
    for row in rows:
        chunks.append(
            {
                "content": row[0],
                "user_id": row[1],
                "document_id": row[2],
            }
        )
    return chunks


async def migrate_to_pinecone(chunks: list[dict[str, str]]) -> None:
    """Group chunks by document and embed into Pinecone."""
    from kume.adapters.output.pinecone_embedding import PineconeEmbeddingRepository

    api_key = os.environ.get("PINECONE_API_KEY", "")
    index_name = os.environ.get("PINECONE_INDEX", "kume-documents")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if not api_key:
        print("ERROR: PINECONE_API_KEY not set in .env")
        return
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        return

    repo = PineconeEmbeddingRepository(
        api_key=api_key,
        index_name=index_name,
        openai_api_key=openai_key,
        embedding_model=embedding_model,
    )

    # Group chunks by (user_id, document_id)
    groups: dict[tuple[str, str], list[str]] = {}
    for chunk in chunks:
        key = (chunk["user_id"], chunk["document_id"])
        groups.setdefault(key, []).append(chunk["content"])

    total_chunks = 0
    for (user_id, document_id), texts in groups.items():
        print(f"  Embedding {len(texts)} chunks for document {document_id[:8]}... (user {user_id[:8]}...)")
        await repo.embed_chunks(user_id, document_id, texts)
        total_chunks += len(texts)

    print(f"\nDone! Migrated {total_chunks} chunks across {len(groups)} documents.")


def main() -> None:
    print("=== pgvector → Pinecone Migration ===\n")

    print("1. Reading chunks from local pgvector...")
    chunks = read_chunks_from_pgvector()
    print(f"   Found {len(chunks)} chunks\n")

    if not chunks:
        print("No chunks to migrate.")
        return

    # Show summary
    users = set(c["user_id"] for c in chunks)
    docs = set(c["document_id"] for c in chunks)
    print(f"   Users: {len(users)}")
    print(f"   Documents: {len(docs)}")
    print(f"   Total chunks: {len(chunks)}\n")

    print("2. Embedding into Pinecone...")
    asyncio.run(migrate_to_pinecone(chunks))


if __name__ == "__main__":
    main()

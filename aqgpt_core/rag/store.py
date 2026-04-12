"""Embedding and vector store helpers for RAG."""

import chromadb
from sentence_transformers import SentenceTransformer

from aqgpt_core.rag.settings import (
    RAG_CHROMA_DIR,
    RAG_COLLECTION_NAME,
    RAG_EMBED_MODEL,
)

BATCH_SIZE = 100


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(RAG_CHROMA_DIR))
    return client.get_or_create_collection(
        name=RAG_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_embedder() -> SentenceTransformer:
    return SentenceTransformer(RAG_EMBED_MODEL)


def embed_and_store(chunks: list[dict]) -> int:
    """Embed chunk records and upsert into Chroma."""
    RAG_CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    model = build_embedder()
    collection = get_chroma_collection()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [c["id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return collection.count()

"""Vector search over the NileTel knowledge base (Chroma, persisted on disk)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb
from langchain_core.embeddings import Embeddings

from .config import Settings


def collection_name(settings: Settings) -> str:
    # One collection per embedding backend: vectors from different models have
    # different dimensions and must never be mixed.
    slug = re.sub(r"[^a-z0-9]+", "_", settings.embedding_backend.lower()).strip("_")
    return f"niletel_kb_{slug}"


def open_collection(settings: Settings, reset: bool = False):
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    name = collection_name(settings)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # collection did not exist yet
            pass
    # We pass our own vectors, so Chroma's built-in embedding function is disabled.
    return client.get_or_create_collection(name, embedding_function=None, metadata={"hnsw:space": "cosine"})


@dataclass
class RetrievedChunk:
    text: str
    source: str
    section: str
    score: float  # cosine similarity, higher is better

    def cite(self) -> str:
        return f"{self.source} ({self.section})"


class KnowledgeBase:
    def __init__(self, settings: Settings, embeddings: Embeddings):
        self.embeddings = embeddings
        self.collection = open_collection(settings)

    def count(self) -> int:
        return self.collection.count()

    def search(self, query: str, k: int = 4, source: str | None = None) -> list[RetrievedChunk]:
        """Top-k chunks by cosine similarity, optionally restricted to one source document."""
        if self.count() == 0:
            raise RuntimeError("The knowledge base is empty. Run `python -m telecom_agent ingest` first.")
        result = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=min(k, self.count()),
            where={"source": source} if source else None,
        )
        return [
            RetrievedChunk(text=doc, source=meta["source"], section=meta["section"], score=round(1 - dist, 4))
            for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

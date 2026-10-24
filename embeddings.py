"""Embedding backends.

All backends implement LangChain's `Embeddings` interface (embed_documents / embed_query),
so the rest of the code does not care which one is used.

- sentence-transformers: all-MiniLM-L6-v2 running locally on CPU (default).
- ollama: e.g. nomic-embed-text served by a local Ollama instance.
- hashing: scikit-learn HashingVectorizer. Lexical, not semantic, but needs no model
  download, so tests and CI can run fully offline.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .config import Settings


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashingEmbeddings(Embeddings):
    """Stateless bag-of-words embedding (unigrams + bigrams), L2-normalised."""

    def __init__(self, n_features: int = 2**14):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            ngram_range=(1, 2),
            stop_words="english",
            alternate_sign=False,
            norm="l2",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class NomicOllamaEmbeddings(Embeddings):
    """nomic-embed-text expects task prefixes on documents and queries."""

    def __init__(self, model: str, base_url: str):
        from langchain_ollama import OllamaEmbeddings

        self.inner = OllamaEmbeddings(model=model, base_url=base_url)
        self.use_prefix = "nomic" in model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.use_prefix:
            texts = [f"search_document: {t}" for t in texts]
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(f"search_query: {text}" if self.use_prefix else text)


def get_embeddings(settings: Settings) -> Embeddings:
    backend = settings.embedding_backend.lower()
    if backend in ("sentence-transformers", "st"):
        return SentenceTransformerEmbeddings(settings.st_model)
    if backend == "ollama":
        return NomicOllamaEmbeddings(settings.ollama_embed_model, settings.ollama_base_url)
    if backend == "hashing":
        return HashingEmbeddings()
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {settings.embedding_backend!r}")

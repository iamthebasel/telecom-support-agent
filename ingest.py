"""Ingestion pipeline: markdown files -> sections -> overlapping chunks -> embeddings -> Chroma.

Run with:  python -m telecom_agent ingest
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings
from .embeddings import get_embeddings
from .retriever import open_collection

HEADING = re.compile(r"^(#{1,3})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    text: str
    source: str       # file name, used for citations
    title: str        # document H1
    section: str      # H2 (and H3) heading path
    chunk_index: int

    @property
    def embedding_text(self) -> str:
        # Prepending the document title and section gives short chunks the context
        # they lose when cut out of the document ("contextual chunk headers").
        return f"{self.title} - {self.section}\n{self.text}"


def split_sections(markdown: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """Return (title, [(section_heading, lines), ...]) for one markdown document."""
    title = ""
    sections: list[tuple[str, list[str]]] = []
    current_h2 = "Overview"
    heading, lines = current_h2, []

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if not match:
            if line.strip():
                lines.append(line.rstrip())
            continue
        level, text = len(match.group(1)), match.group(2).strip()
        if level == 1:
            title = text
            continue
        if lines:
            sections.append((heading, lines))
        if level == 2:
            current_h2 = text
            heading = text
        else:
            heading = f"{current_h2} > {text}"
        lines = []

    if lines:
        sections.append((heading, lines))
    return title, sections


def chunk_lines(lines: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Pack whole lines into chunks of ~chunk_size words, carrying ~overlap words forward.

    Splitting on line boundaries keeps table rows and bullet points intact.
    """
    chunks: list[str] = []
    window: list[str] = []
    words = 0
    for line in lines:
        n = len(line.split())
        if window and words + n > chunk_size:
            chunks.append("\n".join(window))
            # keep trailing lines worth ~overlap words as the start of the next chunk
            carried, carried_words = [], 0
            for prev in reversed(window):
                if carried_words >= overlap:
                    break
                carried.insert(0, prev)
                carried_words += len(prev.split())
            window, words = carried, carried_words
        window.append(line)
        words += n
    if window:
        chunks.append("\n".join(window))
    return chunks


def build_chunks(kb_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        title, sections = split_sections(path.read_text(encoding="utf-8"))
        index = 0
        for heading, lines in sections:
            for text in chunk_lines(lines, chunk_size, overlap):
                chunks.append(Chunk(
                    id=f"{path.stem}-{index:02d}",
                    text=text,
                    source=path.name,
                    title=title or path.stem,
                    section=heading,
                    chunk_index=index,
                ))
                index += 1
    return chunks


def ingest(settings: Settings | None = None, embeddings: Embeddings | None = None) -> dict:
    """Rebuild the vector store from scratch. Returns a small summary."""
    settings = settings or get_settings()
    embeddings = embeddings or get_embeddings(settings)

    chunks = build_chunks(settings.kb_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError(f"No markdown documents found in {settings.kb_dir}")

    collection = open_collection(settings, reset=True)
    vectors = embeddings.embed_documents([c.embedding_text for c in chunks])
    collection.add(
        ids=[c.id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[
            {"source": c.source, "title": c.title, "section": c.section, "chunk_index": c.chunk_index}
            for c in chunks
        ],
    )
    return {
        "documents": len({c.source for c in chunks}),
        "chunks": len(chunks),
        "collection": collection.name,
        "embedding_backend": settings.embedding_backend,
    }

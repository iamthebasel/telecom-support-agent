from telecom_agent.ingest import build_chunks, chunk_lines, split_sections

SAMPLE = """# Roaming Guide

Intro line.

## Zones
- Zone 1: Gulf

## Passes
### Day Pass
- 99 EGP
"""


def test_split_sections_tracks_title_and_headings():
    title, sections = split_sections(SAMPLE)
    assert title == "Roaming Guide"
    assert [heading for heading, _ in sections] == ["Overview", "Zones", "Passes > Day Pass"]
    assert sections[2][1] == ["- 99 EGP"]


def test_chunk_lines_overlaps_and_keeps_every_line():
    lines = [f"line{i} " + "word " * 9 for i in range(10)]  # 10 words per line
    chunks = chunk_lines(lines, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.splitlines()[-1] == current.splitlines()[0]  # one line of overlap
    joined = "\n".join(chunks)
    assert all(line.rstrip() in joined for line in lines)


def test_build_chunks_on_knowledge_base(settings):
    chunks = build_chunks(settings.kb_dir, settings.chunk_size, settings.chunk_overlap)
    sources = {c.source for c in chunks}
    assert len(sources) == len(list(settings.kb_dir.glob("*.md")))
    assert len({c.id for c in chunks}) == len(chunks)  # ids are unique
    assert all(c.title and c.section and c.text for c in chunks)
    assert all(len(c.text.split()) <= settings.chunk_size + 40 for c in chunks)


def test_ingest_stores_every_chunk(kb, indexed):
    assert kb.count() == indexed["chunks"]
    assert indexed["documents"] == 17

from dataclasses import replace

import pytest

from telecom_agent.evaluation import first_relevant_rank, judge_faithfulness, retrieval_metrics, routing_metrics
from telecom_agent.retriever import KnowledgeBase

from .fake_llm import ScriptedChatModel, tool_call


def test_search_finds_the_right_document(kb):
    hits = kb.search("roaming day pass price in Saudi Arabia", k=3)
    assert hits[0].source == "roaming.md"
    assert hits[0].score >= hits[-1].score


def test_search_can_filter_by_source(kb):
    hits = kb.search("discount", k=5, source="retention_offers.md")
    assert hits and all(h.source == "retention_offers.md" for h in hits)


def test_empty_knowledge_base_raises(settings, embeddings, tmp_path):
    empty = KnowledgeBase(replace(settings, chroma_dir=tmp_path / "empty"), embeddings)
    with pytest.raises(RuntimeError, match="ingest"):
        empty.search("anything")


def test_retrieval_metrics():
    assert first_relevant_rank(["a.md", "b.md", "c.md"], {"c.md"}) == 3
    assert first_relevant_rank(["a.md"], {"z.md"}) is None
    m = retrieval_metrics([1, 3, None, 2])
    assert m == {"hit@1": 0.25, "hit@3": 0.75, "mrr": round((1 + 1 / 3 + 1 / 2) / 4, 3), "questions": 4}


def test_routing_metrics():
    m = routing_metrics(["knowledge", "account", "other"], ["knowledge", "knowledge", "other"])
    assert m["accuracy"] == round(2 / 3, 3)
    assert m["confusion"]["account"]["knowledge"] == 1
    assert m["per_route_recall"]["account"] == 0.0


def test_judge_faithfulness_parses_structured_verdict():
    llm = ScriptedChatModel(responses=[tool_call("Faithfulness", {"score": 5, "reason": "all supported"}, "j1")])
    verdict = judge_faithfulness(llm, "q?", "context", "answer")
    assert verdict.score == 5 and verdict.reason == "all supported"
    assert "CONTEXT:\ncontext" in llm.prompts[0][1].content

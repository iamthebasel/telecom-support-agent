"""Offline evaluation: retrieval quality and router accuracy. No LLM needed (except --routers llm).

    python eval/run_eval.py
    python eval/run_eval.py --backends sentence-transformers --routers rules,semantic,llm
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from telecom_agent.config import get_settings
from telecom_agent.embeddings import get_embeddings
from telecom_agent.evaluation import first_relevant_rank, load_questions, retrieval_metrics, routing_metrics
from telecom_agent.ingest import ingest
from telecom_agent.retriever import KnowledgeBase
from telecom_agent.router import make_router

EVAL_DIR = Path(__file__).resolve().parent


def eval_retrieval(kb: KnowledgeBase, questions: list[dict], depth: int = 10) -> dict:
    ranks, misses, latencies = [], [], []
    for q in questions:
        start = time.perf_counter()
        hits = kb.search(q["question"], k=depth)
        latencies.append((time.perf_counter() - start) * 1000)
        rank = first_relevant_rank([h.source for h in hits], set(q["sources"]))
        ranks.append(rank)
        if rank != 1:
            misses.append({"id": q["id"], "rank": rank, "expected": q["sources"], "top1": hits[0].source})
    return {**retrieval_metrics(ranks), "avg_query_ms": round(statistics.mean(latencies), 1),
            "not_rank_1": misses}


def eval_router(router, questions: list[dict]) -> dict:
    predicted, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        predicted.append(router.route(q["question"]).route)
        latencies.append((time.perf_counter() - start) * 1000)
    expected = [q["route"] for q in questions]
    metrics = routing_metrics(expected, predicted)
    metrics["avg_route_ms"] = round(statistics.mean(latencies), 2)
    metrics["errors"] = [{"id": q["id"], "question": q["question"], "expected": e, "predicted": p}
                         for q, e, p in zip(questions, expected, predicted) if e != p]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="sentence-transformers,hashing")
    parser.add_argument("--routers", default="rules,semantic")
    parser.add_argument("--out", default=str(EVAL_DIR / "results.json"))
    args = parser.parse_args()

    base = get_settings()
    questions = load_questions(EVAL_DIR / "questions.jsonl")
    retrieval_qs = [q for q in questions if q["sources"]]
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": {"total": len(questions), "with_expected_sources": len(retrieval_qs)},
        "retrieval": {},
        "routing": {},
    }

    embeddings_by_backend = {}
    for backend in args.backends.split(","):
        settings = replace(base, embedding_backend=backend)
        embeddings = get_embeddings(settings)
        embeddings_by_backend[backend] = embeddings
        summary = ingest(settings, embeddings)  # rebuild so the index matches the current KB
        results["knowledge_base"] = {"documents": summary["documents"], "chunks": summary["chunks"],
                                     "chunk_size_words": settings.chunk_size,
                                     "chunk_overlap_words": settings.chunk_overlap}
        results["retrieval"][backend] = eval_retrieval(KnowledgeBase(settings, embeddings), retrieval_qs)
        r = results["retrieval"][backend]
        print(f"retrieval [{backend}] hit@1={r['hit@1']} hit@3={r['hit@3']} mrr={r['mrr']} (n={r['questions']})")

    for mode in args.routers.split(","):
        llm = None
        embeddings = next(iter(embeddings_by_backend.values()))
        if mode == "llm":
            from telecom_agent.llm import get_chat_model

            llm = get_chat_model(base)
        router = make_router(mode, llm=llm, embeddings=embeddings)
        key = mode if mode != "semantic" else f"semantic ({args.backends.split(',')[0]})"
        if mode == "llm":
            key = f"llm ({base.llm_provider})"
        results["routing"][key] = eval_router(router, questions)
        r = results["routing"][key]
        print(f"routing   [{key}] accuracy={r['accuracy']} (n={r['questions']}) errors={len(r['errors'])}")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

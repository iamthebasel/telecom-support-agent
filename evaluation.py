"""Evaluation helpers: retrieval metrics, routing accuracy and an LLM-as-judge check."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .prompts import JUDGE_SYSTEM, JUDGE_USER


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_relevant_rank(retrieved_sources: list[str], expected: set[str]) -> int | None:
    """1-based rank of the first retrieved chunk that comes from an expected document."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return rank
    return None


def retrieval_metrics(ranks: list[int | None], ks: tuple[int, ...] = (1, 3)) -> dict:
    """Hit rate @k (share of questions with a relevant chunk in the top k) and MRR."""
    n = len(ranks)
    metrics = {f"hit@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / n, 3) for k in ks}
    metrics["mrr"] = round(sum(1 / r for r in ranks if r is not None) / n, 3)
    metrics["questions"] = n
    return metrics


def routing_metrics(expected: list[str], predicted: list[str]) -> dict:
    labels = sorted(set(expected) | set(predicted))
    confusion = {e: {p: 0 for p in labels} for e in labels}
    for e, p in zip(expected, predicted):
        confusion[e][p] += 1
    totals = Counter(expected)
    return {
        "accuracy": round(sum(e == p for e, p in zip(expected, predicted)) / len(expected), 3),
        "per_route_recall": {r: round(confusion[r][r] / totals[r], 3) for r in totals},
        "confusion": confusion,  # confusion[expected][predicted]
        "questions": len(expected),
    }


class Faithfulness(BaseModel):
    score: int = Field(ge=1, le=5, description="5 = every claim supported by the context")
    reason: str


def judge_faithfulness(llm: BaseChatModel, question: str, context: str, answer: str) -> Faithfulness:
    judge = llm.with_structured_output(Faithfulness)
    prompt = JUDGE_USER.format(question=question, context=context, answer=answer)
    return judge.invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(prompt)])

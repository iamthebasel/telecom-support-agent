"""LLM-as-judge: are the knowledge agent's answers supported by the retrieved context?

Requires a working LLM (LLM_PROVIDER=ollama with Ollama running, or anthropic/openai with a key).
The same model answers and judges by default; set JUDGE_PROVIDER to use a different one.

    python eval/judge_faithfulness.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import replace
from pathlib import Path

from telecom_agent.agent import SupportAgent
from telecom_agent.config import get_settings
from telecom_agent.evaluation import judge_faithfulness, load_questions
from telecom_agent.graph import format_context
from telecom_agent.llm import describe_model, get_chat_model

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only judge the first N knowledge questions")
    parser.add_argument("--out", default=str(EVAL_DIR / "faithfulness.json"))
    args = parser.parse_args()

    settings = get_settings()
    judge_settings = replace(settings, llm_provider=os.getenv("JUDGE_PROVIDER", settings.llm_provider))
    agent = SupportAgent(settings)
    judge = get_chat_model(judge_settings)

    questions = [q for q in load_questions(EVAL_DIR / "questions.jsonl") if q["route"] == "knowledge"]
    rows = []
    for q in questions[: args.limit]:
        reply = agent.ask(q["question"])
        if reply.error:
            raise SystemExit(f"The agent failed ({reply.error}). Is the LLM configured and reachable?")
        # same query and k as the knowledge agent, so the judge sees the context the answer was based on
        context = format_context(agent.kb.search(q["question"], k=settings.top_k))
        verdict = judge_faithfulness(judge, q["question"], context, reply.answer)
        rows.append({"id": q["id"], "score": verdict.score, "reason": verdict.reason, "answer": reply.answer,
                     "cited_expected_source": any(s in reply.answer for s in q["sources"])})
        print(f"{q['id']} score={verdict.score} {verdict.reason}")

    summary = {
        "answer_model": describe_model(settings),
        "judge_model": describe_model(judge_settings),
        "questions": len(rows),
        "mean_score": round(statistics.mean(r["score"] for r in rows), 2),
        "share_score_ge_4": round(sum(r["score"] >= 4 for r in rows) / len(rows), 3),
        "share_citing_expected_source": round(sum(r["cited_expected_source"] for r in rows) / len(rows), 3),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

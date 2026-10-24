"""Run logging (one JSON line per request) and a small log summary."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path


class RunLogger:
    def __init__(self, path: Path):
        self.path = path

    def log(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(runs: list[dict]) -> dict:
    if not runs:
        return {"runs": 0}
    latencies = [r["latency_ms"] for r in runs]
    tool_calls = sum(len(r["tools_called"]) for r in runs)
    tool_errors = sum(r["tool_errors"] for r in runs)
    return {
        "runs": len(runs),
        "runs_per_route": dict(Counter(r["route"] for r in runs)),
        "avg_latency_ms": round(statistics.mean(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1], 1),  # nearest rank
        "failed_runs": sum(1 for r in runs if r.get("error")),
        "tool_calls": tool_calls,
        "tool_calls_per_tool": dict(Counter(t for r in runs for t in r["tools_called"])),
        "tool_error_rate": round(tool_errors / tool_calls, 3) if tool_calls else 0.0,
        "top_sources": dict(Counter(s for r in runs for s in r["sources"]).most_common(5)),
    }

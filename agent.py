"""SupportAgent: wires settings, retrieval, tools, router and graph together, and logs every run."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from .churn_client import ChurnClient
from .config import Settings, get_settings
from .database import init_db
from .embeddings import get_embeddings
from .graph import build_graph
from .llm import describe_model, get_chat_model
from .monitoring import RunLogger
from .retriever import KnowledgeBase
from .router import make_router
from .tools import build_tools


@dataclass
class AgentReply:
    answer: str
    route: str
    route_reason: str
    session_id: str
    run_id: str
    latency_ms: float
    sources: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    tool_errors: int = 0
    error: str | None = None


def message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # some providers return a list of content blocks
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))


def tool_result(message: ToolMessage) -> dict:
    try:
        data = json.loads(message.content)
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


class SupportAgent:
    def __init__(self, settings: Settings | None = None, *, llm: BaseChatModel | None = None,
                 embeddings: Embeddings | None = None, churn: ChurnClient | None = None,
                 router=None, logger: RunLogger | None = None):
        self.settings = settings or get_settings()
        s = self.settings
        embeddings = embeddings or get_embeddings(s)
        llm = llm or get_chat_model(s)

        self.kb = KnowledgeBase(s, embeddings)
        init_db(s.db_path)
        self.churn = churn or ChurnClient(s.churn_api_url, s.churn_api_timeout)
        self.router = router or make_router(s.router_mode, llm=llm, embeddings=embeddings)
        self.logger = logger or RunLogger(s.log_path)

        tools = build_tools(self.kb, s.db_path, self.churn)
        self.graph = build_graph(llm, self.kb, tools, self.router, top_k=s.top_k, checkpointer=InMemorySaver())

    def ask(self, message: str, customer_id: str | None = None, session_id: str | None = None) -> AgentReply:
        session_id = session_id or uuid.uuid4().hex[:12]
        run_id = uuid.uuid4().hex[:12]
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 12}

        # the checkpointer keeps the whole conversation; remember where this turn starts
        seen = len(self.graph.get_state(config).values.get("messages", []))
        inputs: dict = {"messages": [HumanMessage(message)]}
        if customer_id:
            inputs["customer_id"] = customer_id.strip().upper()

        start = time.perf_counter()
        error = None
        try:
            state = self.graph.invoke(inputs, config)
        except Exception as exc:  # keep serving; the error is logged and returned
            error = f"{exc.__class__.__name__}: {exc}"
            state = self.graph.get_state(config).values
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        new_messages = state.get("messages", [])[seen:]
        tool_messages = [m for m in new_messages if isinstance(m, ToolMessage)]
        tools_called = [call["name"] for m in new_messages if isinstance(m, AIMessage) for call in m.tool_calls]
        tool_errors = sum(1 for m in tool_messages if m.status == "error" or "error" in tool_result(m))

        sources = list(state.get("sources") or [])
        for m in tool_messages:
            sources += [s for s in tool_result(m).get("sources", []) if s not in sources]

        final = next((m for m in reversed(new_messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
        if error or final is None:
            answer = "Sorry, something went wrong while handling this request. Please try again or escalate."
        else:
            answer = message_text(final)

        reply = AgentReply(
            answer=answer, route=state.get("route", "unknown"), route_reason=state.get("route_reason", ""),
            session_id=session_id, run_id=run_id, latency_ms=latency_ms, sources=sources,
            tools_called=tools_called, tool_errors=tool_errors, error=error,
        )
        self.logger.log({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "llm": describe_model(self.settings),
            "router_mode": self.router.mode,
            "embedding_backend": self.settings.embedding_backend,
            "question": message[:200],
            **{k: v for k, v in asdict(reply).items() if k != "answer"},
        })
        return reply

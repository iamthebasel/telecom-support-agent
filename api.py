"""FastAPI service.

    uvicorn telecom_agent.api:app --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .agent import SupportAgent
from .churn_client import ChurnServiceError


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    customer_id: str | None = Field(default=None, examples=["NT-1001"])
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list[str]
    tools_called: list[str]
    session_id: str
    run_id: str
    latency_ms: float


def create_app(agent: SupportAgent | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # build the agent once at startup (loads the embedding model and opens the vector store)
        app.state.agent = agent or SupportAgent()
        yield

    app = FastAPI(title="NileTel Support Agent", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        support: SupportAgent = request.app.state.agent
        try:
            churn_status = support.churn.health().get("status", "unknown")
        except ChurnServiceError:
            churn_status = "unreachable"
        return {
            "status": "ok",
            "llm_provider": support.settings.llm_provider,
            "router_mode": support.router.mode,
            "embedding_backend": support.settings.embedding_backend,
            "kb_chunks": support.kb.count(),
            "churn_api": churn_status,
        }

    # a plain `def` endpoint runs in FastAPI's threadpool, so the blocking LLM call is fine
    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        support: SupportAgent = request.app.state.agent
        reply = support.ask(body.message, customer_id=body.customer_id, session_id=body.session_id)
        if reply.error:
            raise HTTPException(status_code=502, detail={"error": reply.error, "run_id": reply.run_id})
        return ChatResponse(
            answer=reply.answer, route=reply.route, sources=reply.sources, tools_called=reply.tools_called,
            session_id=reply.session_id, run_id=reply.run_id, latency_ms=reply.latency_ms,
        )

    return app


app = create_app()

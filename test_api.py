import pytest
import responses
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from telecom_agent.agent import SupportAgent
from telecom_agent.api import create_app
from telecom_agent.monitoring import summarize

from .conftest import CHURN_URL
from .fake_llm import ScriptedChatModel


@pytest.fixture
def client(settings, embeddings, indexed):
    llm = ScriptedChatModel(responses=[AIMessage("eSIM costs 50 EGP [sim_and_esim.md].")])
    agent = SupportAgent(settings, llm=llm, embeddings=embeddings)
    with TestClient(create_app(agent)) as c:  # the context manager runs the startup lifespan
        yield c


@responses.activate
def test_health_reports_components(client):
    responses.get(f"{CHURN_URL}/health", json={"status": "ok", "model": "LogisticRegression", "threshold": 0.4})
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["churn_api"] == "ok"
    assert body["kb_chunks"] > 0 and body["router_mode"] == "rules"


@responses.activate
def test_health_when_churn_api_is_down(client):
    responses.get(f"{CHURN_URL}/health", status=503)
    assert client.get("/health").json()["churn_api"] == "unreachable"


def test_chat_endpoint(client):
    response = client.post("/chat", json={"message": "How much is an eSIM?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "knowledge"
    assert "sim_and_esim.md" in body["sources"]
    assert body["answer"].startswith("eSIM costs 50 EGP")


def test_chat_validates_input(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_log_summary():
    runs = [
        {"route": "knowledge", "latency_ms": 100, "tools_called": [], "tool_errors": 0, "sources": ["a.md"], "error": None},
        {"route": "account", "latency_ms": 300, "tools_called": ["get_customer_profile", "predict_churn"],
         "tool_errors": 1, "sources": [], "error": None},
    ]
    summary = summarize(runs)
    assert summary["runs_per_route"] == {"knowledge": 1, "account": 1}
    assert summary["avg_latency_ms"] == 200
    assert summary["tool_error_rate"] == 0.5

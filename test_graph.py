"""End-to-end graph tests with a scripted LLM, hashing embeddings and a mocked churn API."""

import json

import pytest
import requests
import responses
from langchain_core.messages import AIMessage, ToolMessage

from telecom_agent.agent import SupportAgent
from telecom_agent.monitoring import read_runs
from telecom_agent.prompts import MISSING_CUSTOMER_ID, OTHER_REPLY

from .conftest import CHURN_URL
from .fake_llm import ScriptedChatModel, tool_call


def make_agent(settings, embeddings, responses_):
    llm = ScriptedChatModel(responses=responses_)
    return SupportAgent(settings, llm=llm, embeddings=embeddings), llm


def test_knowledge_route_answers_with_retrieved_context(settings, embeddings, indexed):
    agent, llm = make_agent(settings, embeddings, [AIMessage("A Zone 1 Day Pass costs 99 EGP [roaming.md].")])
    reply = agent.ask("How much is a roaming day pass in Saudi Arabia?")

    assert reply.route == "knowledge"
    assert reply.answer.endswith("[roaming.md].")
    assert "roaming.md" in reply.sources and reply.tools_called == []
    system_prompt = llm.prompts[0][0].content
    assert "[roaming.md]" in system_prompt  # retrieved chunks were injected into the prompt


@responses.activate
def test_account_route_runs_the_multi_step_tool_loop(settings, embeddings, indexed):
    responses.post(f"{CHURN_URL}/predict", json={"churn_probability": 0.81, "will_churn": True, "risk": "high"})
    agent, llm = make_agent(settings, embeddings, [
        tool_call("get_customer_profile", {"customer_id": "NT-1001"}, "c1"),
        tool_call("predict_churn", {"customer_id": "NT-1001"}, "c2"),
        tool_call("suggest_retention_offer", {"customer_id": "NT-1001", "churn_risk": "high"}, "c3"),
        AIMessage("High risk. Offer RET-01 and RET-03 [retention_offers.md]."),
    ])
    reply = agent.ask("Customer NT-1001 wants to cancel. What can I offer?")

    assert reply.route == "account"
    assert reply.tools_called == ["get_customer_profile", "predict_churn", "suggest_retention_offer"]
    assert reply.tool_errors == 0
    assert reply.sources == ["retention_offers.md"]
    assert "RET-01" in reply.answer

    # the last LLM call saw all three tool results
    tool_results = [m for m in llm.prompts[-1] if isinstance(m, ToolMessage)]
    assert [json.loads(m.content).get("risk") for m in tool_results][1] == "high"
    offers = json.loads(tool_results[2].content)["offers"]
    assert [o["code"] for o in offers] == ["RET-01", "RET-03"]


@responses.activate
def test_account_route_survives_churn_api_outage(settings, embeddings, indexed):
    responses.post(f"{CHURN_URL}/predict", body=requests.ConnectionError("refused"))
    agent, _ = make_agent(settings, embeddings, [
        tool_call("predict_churn", {"customer_id": "NT-1003"}, "c1"),
        tool_call("suggest_retention_offer", {"customer_id": "NT-1003", "churn_risk": "unknown"}, "c2"),
        AIMessage("The churn service is down; based on the profile I suggest RET-04 and RET-03."),
    ])
    reply = agent.ask("Is NT-1003 likely to churn?")
    assert reply.error is None
    assert reply.tool_errors == 1
    assert "churn service is down" in reply.answer


def test_account_route_asks_for_customer_id(settings, embeddings, indexed):
    agent, llm = make_agent(settings, embeddings, [])
    reply = agent.ask("The customer says they're thinking about switching to another operator.")
    assert reply.route == "account"
    assert reply.answer == MISSING_CUSTOMER_ID
    assert llm.prompts == []  # no LLM call needed


def test_other_route_returns_canned_reply(settings, embeddings, indexed):
    agent, llm = make_agent(settings, embeddings, [])
    reply = agent.ask("Hi there!")
    assert reply.route == "other" and reply.answer == OTHER_REPLY and llm.prompts == []


@responses.activate
def test_customer_id_persists_within_a_session(settings, embeddings, indexed):
    agent, llm = make_agent(settings, embeddings, [
        AIMessage("Our hotline is 1250 [contact_channels.md]."),
        tool_call("get_customer_profile", {"customer_id": "NT-1002"}, "c1"),
        AIMessage("Hassan has been with us for 62 months on a Two year contract."),
    ])
    first = agent.ask("What's the customer care hotline number?", customer_id="NT-1002")
    second = agent.ask("Pull up this customer's profile.", session_id=first.session_id)
    assert second.route == "account"
    assert second.tools_called == ["get_customer_profile"]
    assert "NT-1002" in llm.prompts[1][0].content  # system prompt names the remembered customer


def test_runs_are_logged(settings, embeddings, indexed):
    before = len(read_runs(settings.log_path))
    agent, _ = make_agent(settings, embeddings, [AIMessage("APN is internet.niletel [troubleshooting_mobile_data.md].")])
    agent.ask("What APN should I use?")
    runs = read_runs(settings.log_path)
    assert len(runs) == before + 1
    record = runs[-1]
    for key in ("timestamp", "route", "tools_called", "sources", "latency_ms", "error", "router_mode"):
        assert key in record
    assert record["route"] == "knowledge" and record["error"] is None


def test_llm_failure_is_reported_not_raised(settings, embeddings, indexed):
    agent, _ = make_agent(settings, embeddings, [])  # the scripted model raises when called
    reply = agent.ask("How do I activate an eSIM?")
    assert reply.error and "ran out of responses" in reply.error
    assert reply.answer.startswith("Sorry")


@pytest.fixture(autouse=True)
def _no_real_http():
    # any HTTP call that is not explicitly mocked fails loudly
    with responses.RequestsMock(assert_all_requests_are_fired=False):
        yield

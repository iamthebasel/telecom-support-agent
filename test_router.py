import pytest
from langchain_core.messages import AIMessage

from telecom_agent.router import LLMRouter, RuleRouter, SemanticRouter

from .fake_llm import ScriptedChatModel, tool_call


@pytest.mark.parametrize("text, route", [
    ("How much does Max 750 cost per month?", "knowledge"),
    ("What's the early termination fee policy?", "knowledge"),
    ("Look up NT-1010 please", "account"),
    ("The caller is threatening to switch to another operator", "account"),
    ("What's the churn risk for this customer?", "account"),
    ("Good morning!", "other"),
])
def test_rule_router(text, route):
    assert RuleRouter().route(text).route == route


def test_semantic_router_matches_its_own_examples(embeddings):
    router = SemanticRouter(embeddings, k=1)
    for route, examples in SemanticRouter.EXAMPLES.items():
        assert router.route(examples[0]).route == route


def test_llm_router_uses_structured_output():
    llm = ScriptedChatModel(responses=[
        tool_call("RouteDecision", {"route": "account", "reason": "mentions a customer"}, "call_1"),
    ])
    decision = LLMRouter(llm).route("customer wants to leave")
    assert decision.route == "account" and decision.reason == "mentions a customer"


def test_llm_router_falls_back_to_rules_on_bad_output():
    llm = ScriptedChatModel(responses=[AIMessage("I think it's account?")])  # no structured output
    decision = LLMRouter(llm).route("How do I activate eSIM?")
    assert decision.route == "knowledge"
    assert decision.reason.startswith("LLM router failed")

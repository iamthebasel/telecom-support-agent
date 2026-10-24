"""Supervisor routing: decide which specialist handles a message.

Three interchangeable routers:
- RuleRouter: regex/keyword rules. Free, instant and deterministic; also the fallback.
- SemanticRouter: nearest example utterances in embedding space.
- LLMRouter: asks the chat model for a structured RouteDecision; falls back to rules on failure.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .prompts import ROUTER_SYSTEM

Route = Literal["knowledge", "account", "other"]
CUSTOMER_ID = re.compile(r"\bNT-\d{4}\b", re.IGNORECASE)


class RouteDecision(BaseModel):
    route: Route = Field(description="knowledge, account or other")
    reason: str = Field(description="one short sentence explaining the choice")


class RuleRouter:
    mode = "rules"

    # (pattern, weight) - signals that the request is about a specific customer's account
    ACCOUNT_SIGNALS = [
        (CUSTOMER_ID, 3),
        (re.compile(r"\bchurn", re.I), 2),
        (re.compile(r"\b(want|wants|going|planning|thinking|threatening)\w*\s+(to|about)\s+"
                    r"(cancel|leave|leaving|switch|switching|port|porting)\b", re.I), 2),
        (re.compile(r"\b(switch|switching|move|moving|port|porting)\s+(out\s+)?to\s+(a\s+|another\s+)?"
                    r"(competitor|operator|network|provider)\b", re.I), 2),
        (re.compile(r"\b(keep|retain)\s+(them|him|her|the customer|this customer)\b", re.I), 2),
        (re.compile(r"\b(this|the|our)\s+(customer|caller|subscriber)('s)?\b", re.I), 1),
        (re.compile(r"\b(profile|account details|customer record)\b", re.I), 1),
        (re.compile(r"\b(pull up|look up|check)\b", re.I), 1),
    ]
    # general-policy phrasing pulls a borderline message back to the knowledge route
    POLICY_SIGNALS = re.compile(r"\b(policy|according to|in general|rules?)\b", re.I)

    DOMAIN_TERMS = re.compile(
        r"\b(plan|plans|bundle|flex|max|data|gb|mb|add-?ons?|roaming|travel\w*|abroad|umrah|hajj|"
        r"contract|cancel\w*|bill\w*|pay\w*|deposit|sim|esim|5g|4g|coverage|signal|network|apn|"
        r"internet|router|wi-?fi|fiber|fibre|dsl|quota|speed|points|loyalty|iot|m2m|meters?|"
        r"complaint|escalat\w*|hotline|store|recharge|technician|minutes|calls?|sms|price|cost|"
        r"offer|discount|upgrade|downgrade|line|phone|number|niletel|refund|fee|tier)\b",
        re.I,
    )

    def route(self, text: str) -> RouteDecision:
        score = sum(weight for pattern, weight in self.ACCOUNT_SIGNALS if pattern.search(text))
        if self.POLICY_SIGNALS.search(text) and not CUSTOMER_ID.search(text):
            score -= 2
        if score >= 2:
            return RouteDecision(route="account", reason=f"account signals (score {score})")
        if self.DOMAIN_TERMS.search(text):
            return RouteDecision(route="knowledge", reason="telecom topic without customer-specific signals")
        return RouteDecision(route="other", reason="no telecom keywords found")


class SemanticRouter:
    mode = "semantic"

    EXAMPLES: dict[str, list[str]] = {
        "knowledge": [
            "What does the Max 450 plan include?",
            "How much is the weekly data add-on?",
            "How do I set up roaming for my trip to Turkey?",
            "What is the fee for ending a two year contract early?",
            "Which payment methods avoid the deposit?",
            "How do I replace a damaged SIM card?",
            "Where is 5G coverage available?",
            "My router's internet light is off, what should I do?",
            "How do I redeem loyalty points?",
            "What IoT plans do you have for businesses?",
            "How do I file a complaint?",
            "What are the retention policy rules for low-risk customers?",
            "What are the store opening hours?",
            "How much is Home Fiber 100?",
        ],
        "account": [
            "Look up customer NT-1005",
            "What is the churn risk for NT-1018?",
            "This customer wants to leave, what can we offer?",
            "Recommend a retention offer for the customer I'm talking to",
            "Show me the account details for NT-1027",
            "The caller is unhappy and threatening to port out to a competitor",
            "Is this subscriber at risk of churning?",
            "What contract and tenure does customer NT-1040 have?",
            "The customer says they will cancel unless they get a discount",
        ],
        "other": [
            "Hello",
            "Good morning, how are you?",
            "Thanks, bye!",
            "Tell me a joke",
            "What's the capital of France?",
            "Can you help me write an essay?",
            "What time is the football match tonight?",
            "Recommend a good movie",
        ],
    }

    def __init__(self, embeddings: Embeddings, k: int = 3):
        self.embeddings = embeddings
        self.k = k
        self.labels = [route for route, texts in self.EXAMPLES.items() for _ in texts]
        texts = [t for ts in self.EXAMPLES.values() for t in ts]
        self.matrix = np.array(embeddings.embed_documents(texts))

    def route(self, text: str) -> RouteDecision:
        query = np.array(self.embeddings.embed_query(text))
        sims = self.matrix @ query / (np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(query) + 1e-9)
        top = np.argsort(-sims)[: self.k]
        votes: dict[str, float] = {}
        for i in top:
            votes[self.labels[i]] = votes.get(self.labels[i], 0.0) + float(sims[i])
        best = max(votes, key=votes.get)
        return RouteDecision(route=best, reason=f"nearest examples vote {votes[best]:.2f}")


class LLMRouter:
    mode = "llm"

    def __init__(self, llm: BaseChatModel):
        self.classifier = llm.with_structured_output(RouteDecision)
        self.fallback = RuleRouter()

    def route(self, text: str) -> RouteDecision:
        try:
            decision = self.classifier.invoke([SystemMessage(ROUTER_SYSTEM), HumanMessage(text)])
            if isinstance(decision, RouteDecision):
                return decision
            raise ValueError(f"unexpected router output: {decision!r}")
        except Exception as exc:  # network error, bad JSON, refusal...
            decision = self.fallback.route(text)
            decision.reason = f"LLM router failed ({exc.__class__.__name__}), rules: {decision.reason}"
            return decision


def make_router(mode: str, llm: BaseChatModel | None = None, embeddings: Embeddings | None = None):
    if mode == "rules":
        return RuleRouter()
    if mode == "semantic":
        if embeddings is None:
            raise ValueError("semantic router needs an embedding model")
        return SemanticRouter(embeddings)
    if mode == "llm":
        if llm is None:
            raise ValueError("llm router needs a chat model")
        return LLMRouter(llm)
    raise ValueError(f"Unknown ROUTER_MODE: {mode!r} (use rules, semantic or llm)")

"""Tools available to the account & retention agent (exposed to the LLM via function calling)."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from .churn_client import ChurnClient, ChurnServiceError
from .database import get_customer
from .retriever import KnowledgeBase

RETENTION_POLICY_DOC = "retention_offers.md"

OFFERS = {
    "RET-01": "Contract Upgrade Discount",
    "RET-02": "Loyalty Data Boost",
    "RET-03": "Home Care Bundle",
    "RET-04": "AutoPay Saver",
    "RET-05": "Thank-you Reward",
}


def estimate_risk(customer: dict) -> str:
    """Rough fallback used only when the churn service is unavailable."""
    if customer["contract"] == "Month-to-month":
        return "high" if customer["tenure_months"] < 12 else "medium"
    return "low"


def select_retention_offers(customer: dict, risk: str) -> list[dict]:
    """Apply the retention policy rules (see data/knowledge_base/retention_offers.md)."""
    if risk == "low":
        return [{"code": "RET-05", "reason": "low churn risk: thank-you reward, no discount"}]

    offers = []
    # at most one price discount
    if risk == "high" and customer["contract"] == "Month-to-month":
        offers.append({"code": "RET-01", "reason": "high risk on a month-to-month contract"})
    elif customer["payment_method"] in ("Electronic check", "Mailed check"):
        offers.append({"code": "RET-04", "reason": f"pays by {customer['payment_method']}, not automatic payment"})

    # at most one non-price offer
    if customer["internet_service"] in ("DSL", "Fiber optic") and customer["tech_support"] == "No":
        offers.append({"code": "RET-03", "reason": f"{customer['internet_service']} customer without Tech Assist"})
    elif customer["tenure_months"] >= 12:
        offers.append({"code": "RET-02", "reason": f"tenure of {customer['tenure_months']} months"})
    return offers


def build_tools(kb: KnowledgeBase, db_path: Path, churn: ChurnClient) -> list[BaseTool]:
    """Create the tools bound to concrete dependencies (makes them easy to test)."""

    @tool
    def get_customer_profile(customer_id: str) -> str:
        """Look up a NileTel customer's profile by customer ID (e.g. NT-1001): plans, home internet
        type, add-on services, contract type, tenure in months and payment method."""
        customer = get_customer(db_path, customer_id)
        if customer is None:
            return json.dumps({"error": f"no customer found with ID {customer_id}"})
        return json.dumps(customer)

    @tool
    def predict_churn(customer_id: str) -> str:
        """Predict how likely a customer is to leave NileTel, using the churn model service.
        Returns churn_probability (0-1), will_churn and risk (low, medium or high)."""
        customer = get_customer(db_path, customer_id)
        if customer is None:
            return json.dumps({"error": f"no customer found with ID {customer_id}"})
        try:
            return json.dumps({"customer_id": customer["customer_id"], **churn.predict(customer)})
        except ChurnServiceError as exc:
            return json.dumps({"error": str(exc), "risk": "unknown"})

    @tool
    def suggest_retention_offer(customer_id: str, churn_risk: str) -> str:
        """Choose the retention offers the NileTel policy allows for this customer.
        churn_risk is the risk returned by predict_churn: low, medium, high, or unknown if the
        churn service failed. Returns offer codes, the reason for each, and the policy text."""
        customer = get_customer(db_path, customer_id)
        if customer is None:
            return json.dumps({"error": f"no customer found with ID {customer_id}"})

        risk = churn_risk.strip().lower()
        estimated = risk not in ("low", "medium", "high")
        if estimated:
            risk = estimate_risk(customer)

        offers = select_retention_offers(customer, risk)
        policy = []
        for offer in offers:
            offer["name"] = OFFERS[offer["code"]]
            hits = kb.search(f"{offer['code']} {offer['name']}", k=1, source=RETENTION_POLICY_DOC)
            if hits:
                policy.append({"code": offer["code"], "section": hits[0].section, "text": hits[0].text})

        return json.dumps({
            "customer_id": customer["customer_id"],
            "risk_used": risk,
            "risk_estimated_from_profile": estimated,
            "offers": offers,
            "policy": policy,
            "sources": [RETENTION_POLICY_DOC] if policy else [],
        })

    return [get_customer_profile, predict_churn, suggest_retention_offer]

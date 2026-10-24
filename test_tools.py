import json

import pytest
import requests
import responses

from telecom_agent.churn_client import ChurnClient, ChurnServiceError, to_payload
from telecom_agent.database import generate_customers, get_customer
from telecom_agent.tools import build_tools, select_retention_offers

from .conftest import CHURN_URL

EXPECTED_FIELDS = {
    "Tenure Months", "Monthly Charges", "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", "Contract",
    "Paperless Billing", "Payment Method",
}


@pytest.fixture
def tools(kb, settings):
    return {t.name: t for t in build_tools(kb, settings.db_path, ChurnClient(CHURN_URL, timeout=1))}


def test_customer_db_has_synthetic_customers(settings, indexed):
    customer = get_customer(settings.db_path, "nt-1001")  # lookup is case-insensitive
    assert customer["contract"] == "Month-to-month"
    assert get_customer(settings.db_path, "NT-9999") is None


def test_payload_matches_churn_api_schema(settings, indexed):
    payload = to_payload(get_customer(settings.db_path, "NT-1002"))
    assert set(payload) == EXPECTED_FIELDS
    assert isinstance(payload["Tenure Months"], float)
    assert payload["Contract"] == "Two year"


def test_get_customer_profile_tool(tools):
    profile = json.loads(tools["get_customer_profile"].invoke({"customer_id": "NT-1003"}))
    assert profile["payment_method"] == "Mailed check"
    missing = json.loads(tools["get_customer_profile"].invoke({"customer_id": "NT-0000"}))
    assert "error" in missing


@responses.activate
def test_predict_churn_calls_the_api(tools):
    responses.post(f"{CHURN_URL}/predict", json={"churn_probability": 0.73, "will_churn": True, "risk": "high"})
    result = json.loads(tools["predict_churn"].invoke({"customer_id": "NT-1001"}))
    assert result == {"customer_id": "NT-1001", "churn_probability": 0.73, "will_churn": True, "risk": "high"}
    sent = json.loads(responses.calls[0].request.body)
    assert set(sent) == EXPECTED_FIELDS and sent["Internet Service"] == "Fiber optic"


@responses.activate
@pytest.mark.parametrize("failure", [
    requests.ConnectionError("refused"),
    requests.Timeout("slow"),
])
def test_predict_churn_handles_network_errors(tools, failure):
    responses.post(f"{CHURN_URL}/predict", body=failure)
    result = json.loads(tools["predict_churn"].invoke({"customer_id": "NT-1001"}))
    assert result["risk"] == "unknown" and "error" in result


@responses.activate
def test_churn_client_rejects_bad_responses():
    client = ChurnClient(CHURN_URL)
    customer = generate_customers(5)[0]
    responses.post(f"{CHURN_URL}/predict", status=500, body="boom")
    with pytest.raises(ChurnServiceError, match="HTTP 500"):
        client.predict(customer)
    responses.replace(responses.POST, f"{CHURN_URL}/predict", json={"unexpected": 1})
    with pytest.raises(ChurnServiceError, match="unexpected"):
        client.predict(customer)


@pytest.mark.parametrize("customer_id, risk, expected", [
    ("NT-1001", "high", ["RET-01", "RET-03"]),    # month-to-month fiber, no tech support
    ("NT-1003", "medium", ["RET-04", "RET-03"]),  # mailed check, DSL without tech support
    ("NT-1002", "medium", ["RET-02"]),            # automatic payment, has tech support, 62 months
    ("NT-1002", "low", ["RET-05"]),
])
def test_retention_rules(settings, indexed, customer_id, risk, expected):
    customer = get_customer(settings.db_path, customer_id)
    assert [o["code"] for o in select_retention_offers(customer, risk)] == expected


def test_suggest_retention_offer_cites_policy(tools):
    result = json.loads(tools["suggest_retention_offer"].invoke({"customer_id": "NT-1001", "churn_risk": "high"}))
    assert [o["code"] for o in result["offers"]] == ["RET-01", "RET-03"]
    assert result["sources"] == ["retention_offers.md"]
    assert "RET-01" in result["policy"][0]["section"]


def test_suggest_retention_offer_estimates_unknown_risk(tools):
    result = json.loads(tools["suggest_retention_offer"].invoke({"customer_id": "NT-1001", "churn_risk": "unknown"}))
    assert result["risk_estimated_from_profile"] is True
    assert result["risk_used"] == "high"  # month-to-month with 3 months tenure

"""HTTP client for the separate churn prediction service (churn-api, Flask).

API contract:
    GET  /health  -> {"status": "ok", "model": "LogisticRegression", "threshold": 0.4}
    POST /predict -> {"churn_probability": 0.73, "will_churn": true, "risk": "high"}
"""

from __future__ import annotations

import requests

# customer table column -> field name expected by the churn API
FIELD_MAP = {
    "tenure_months": "Tenure Months",
    "monthly_charges": "Monthly Charges",
    "gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
}


class ChurnServiceError(RuntimeError):
    pass


def to_payload(customer: dict) -> dict:
    payload = {api_name: customer[column] for column, api_name in FIELD_MAP.items()}
    payload["Tenure Months"] = float(payload["Tenure Months"])
    payload["Monthly Charges"] = float(payload["Monthly Charges"])
    return payload


class ChurnClient:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ChurnServiceError(f"churn API health check failed: {exc}") from exc

    def predict(self, customer: dict) -> dict:
        try:
            response = requests.post(f"{self.base_url}/predict", json=to_payload(customer), timeout=self.timeout)
        except requests.Timeout as exc:
            raise ChurnServiceError(f"churn API timed out after {self.timeout}s") from exc
        except requests.RequestException as exc:
            raise ChurnServiceError(f"churn API unreachable at {self.base_url}: {exc.__class__.__name__}") from exc

        if response.status_code != 200:
            raise ChurnServiceError(f"churn API returned HTTP {response.status_code}: {response.text[:200]}")
        try:
            data = response.json()
            return {
                "churn_probability": float(data["churn_probability"]),
                "will_churn": bool(data["will_churn"]),
                "risk": str(data["risk"]),
            }
        except (ValueError, KeyError, TypeError) as exc:
            raise ChurnServiceError(f"unexpected churn API response: {response.text[:200]}") from exc

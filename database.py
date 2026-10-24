"""Synthetic customer database (SQLite).

The feature columns mirror the IBM Telco customer churn dataset so a profile can be sent
straight to the churn model. Note that `monthly_charges` is kept on that dataset's scale
(roughly 18-120) because that is what the churn model was trained on; the NileTel bill in
EGP is stored separately in `monthly_bill_egp`. All customers are fake.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path

COLUMNS = [
    "customer_id", "name", "mobile_plan", "home_plan", "monthly_bill_egp",
    "gender", "senior_citizen", "partner", "dependents", "tenure_months",
    "phone_service", "multiple_lines", "internet_service", "online_security",
    "online_backup", "device_protection", "tech_support", "streaming_tv",
    "streaming_movies", "contract", "paperless_billing", "payment_method",
    "monthly_charges",
]

SQL_TYPES = {"tenure_months": "INTEGER", "monthly_bill_egp": "INTEGER", "monthly_charges": "REAL"}

ADDONS = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
ADDON_PRICE_EGP = {"online_security": 35, "online_backup": 40, "device_protection": 60,
                   "tech_support": 50, "streaming_tv": 120, "streaming_movies": 100}
MOBILE_PLANS = {"Max 250": 250, "Max 450": 450, "Max 750": 750}
HOME_PLANS = {"DSL": {"Home DSL 30": 300, "Home DSL 70": 450},
              "Fiber optic": {"Home Fiber 100": 650, "Home Fiber 300": 1100}}
PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]

FIRST_NAMES = ["Ahmed", "Mona", "Omar", "Salma", "Karim", "Nour", "Youssef", "Hana", "Mostafa", "Laila",
               "Tarek", "Dina", "Hassan", "Farida", "Amr", "Yasmin", "Khaled", "Rana", "Sherif", "Mariam"]
LAST_INITIALS = list("ABEFGHKMNRSTZ")

# A few hand-written customers so demos and tests have predictable cases.
DEMO_CUSTOMERS = [
    dict(customer_id="NT-1001", name="Salma R.", gender="Female", senior_citizen="No", partner="No",
         dependents="No", tenure_months=3, phone_service="Yes", multiple_lines="No",
         internet_service="Fiber optic", online_security="No", online_backup="No", device_protection="No",
         tech_support="No", streaming_tv="Yes", streaming_movies="Yes", contract="Month-to-month",
         paperless_billing="Yes", payment_method="Electronic check", mobile_plan="Max 450",
         home_plan="Home Fiber 100"),
    dict(customer_id="NT-1002", name="Hassan M.", gender="Male", senior_citizen="No", partner="Yes",
         dependents="Yes", tenure_months=62, phone_service="Yes", multiple_lines="Yes",
         internet_service="DSL", online_security="Yes", online_backup="Yes", device_protection="Yes",
         tech_support="Yes", streaming_tv="No", streaming_movies="No", contract="Two year",
         paperless_billing="No", payment_method="Bank transfer (automatic)", mobile_plan="Max 250",
         home_plan="Home DSL 70"),
    dict(customer_id="NT-1003", name="Omar K.", gender="Male", senior_citizen="Yes", partner="Yes",
         dependents="No", tenure_months=18, phone_service="Yes", multiple_lines="No",
         internet_service="DSL", online_security="No", online_backup="Yes", device_protection="No",
         tech_support="No", streaming_tv="No", streaming_movies="No", contract="Month-to-month",
         paperless_billing="Yes", payment_method="Mailed check", mobile_plan="Max 250",
         home_plan="Home DSL 30"),
]


def _model_scale_charges(c: dict, rng: random.Random) -> float:
    """Approximate the Telco dataset's monthly charge from the services a customer has."""
    total = 0.0
    if c["phone_service"] == "Yes":
        total += 20 + (5 if c["multiple_lines"] == "Yes" else 0)
    if c["internet_service"] == "DSL":
        total += 25
    elif c["internet_service"] == "Fiber optic":
        total += 50
    total += sum(10 if a.startswith("streaming") else 5 for a in ADDONS if c[a] == "Yes")
    return round(max(18.25, total + rng.uniform(-2, 2)), 2)


def _bill_egp(c: dict) -> int:
    bill = MOBILE_PLANS.get(c["mobile_plan"] or "", 0)
    for plans in HOME_PLANS.values():
        bill += plans.get(c["home_plan"] or "", 0)
    bill += sum(ADDON_PRICE_EGP[a] for a in ADDONS if c[a] == "Yes")
    return bill


def _random_customer(n: int, rng: random.Random) -> dict:
    internet = rng.choices(["DSL", "Fiber optic", "No"], weights=[35, 45, 20])[0]
    phone = "Yes" if internet == "No" else rng.choices(["Yes", "No"], weights=[90, 10])[0]
    c = dict(
        customer_id=f"NT-{1000 + n}",
        name=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_INITIALS)}.",
        gender=rng.choice(["Male", "Female"]),
        senior_citizen=rng.choices(["Yes", "No"], weights=[16, 84])[0],
        partner=rng.choice(["Yes", "No"]),
        dependents=rng.choices(["Yes", "No"], weights=[30, 70])[0],
        tenure_months=rng.randint(1, 72),
        phone_service=phone,
        multiple_lines=rng.choice(["Yes", "No"]) if phone == "Yes" else "No phone service",
        internet_service=internet,
        contract=rng.choices(["Month-to-month", "One year", "Two year"], weights=[55, 21, 24])[0],
        paperless_billing=rng.choices(["Yes", "No"], weights=[60, 40])[0],
        payment_method=rng.choice(PAYMENT_METHODS),
        mobile_plan=rng.choice(list(MOBILE_PLANS)) if phone == "Yes" else None,
        home_plan=rng.choice(list(HOME_PLANS[internet])) if internet != "No" else None,
    )
    for addon in ADDONS:
        c[addon] = rng.choice(["Yes", "No"]) if internet != "No" else "No internet service"
    return c


def generate_customers(n: int = 40, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    customers = [dict(c) for c in DEMO_CUSTOMERS]
    for i in range(len(customers) + 1, n + 1):
        customers.append(_random_customer(i, rng))
    for c in customers:
        c["monthly_charges"] = _model_scale_charges(c, rng)
        c["monthly_bill_egp"] = _bill_egp(c)
    return customers


def init_db(path: Path, n: int = 40, seed: int = 42, reset: bool = False) -> int:
    """Create and populate the customer table. Returns the number of customers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS customers")
        cols = ", ".join(f"{c} {SQL_TYPES.get(c, 'TEXT')}" for c in COLUMNS)
        conn.execute(f"CREATE TABLE IF NOT EXISTS customers ({cols}, PRIMARY KEY (customer_id))")
        if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
            rows = [tuple(c[col] for col in COLUMNS) for c in generate_customers(n, seed)]
            conn.executemany(f"INSERT INTO customers VALUES ({', '.join('?' * len(COLUMNS))})", rows)
        return conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]


def get_customer(path: Path, customer_id: str) -> dict | None:
    if not path.exists():
        init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id.strip().upper(),)).fetchone()
    return dict(row) if row else None

"""Banking relational simulator: customers, accounts, transactions, loans, defaults and fraud.

All rows are synthetic. The schema is deliberately awkward: heavy tails, seasonality, rare classes,
missing values, correlated attributes and enforced foreign keys.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from synthforge.simulator.common import add_missing, random_dates, rng, synthetic_name


def simulate_banking(
    seed: int = 7,
    n_customers: int = 300,
    n_accounts: int = 520,
    n_transactions: int = 6000,
    n_loans: int = 320,
) -> dict[str, pd.DataFrame]:
    r = rng(seed)
    customer_ids = [f"C{i:06d}" for i in range(1, n_customers + 1)]
    age = np.clip(r.normal(47, 16, n_customers), 18, 92).round().astype(int)
    income = np.exp(r.normal(10.7, 0.55, n_customers)).round(2)
    segment = np.where(income > 90_000, "premium", np.where(income > 45_000, "standard", "basic"))
    segment = np.where(r.random(n_customers) < 0.08, "high_risk", segment)
    risk_score = np.clip(0.15 + 0.5 * (income < 40_000) + 0.25 * (segment == "high_risk") + r.normal(0, 0.12, n_customers), 0, 1)
    region = r.choice(["North", "South", "East", "West", "Central"], n_customers, p=[0.22, 0.2, 0.21, 0.19, 0.18])
    city = np.where(region == "North", "Northford", np.where(region == "South", "Southport", "Metro City"))
    signup = random_dates(r, n_customers, "2016-01-01", "2024-06-30")
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "full_name": [synthetic_name(r) for _ in range(n_customers)],
            "age": add_missing(pd.Series(age), r, 0.03, "MAR", pd.Series(income)),
            "gender": add_missing(pd.Series(r.choice(["F", "M", "X"], n_customers, p=[0.49, 0.49, 0.02])), r, 0.02),
            "region": region,
            "city": city,
            "segment": segment,
            "annual_income": income,
            "risk_score": risk_score.round(4),
            "signup_date": signup,
            "kyc_status": r.choice(["verified", "pending", "review"], n_customers, p=[0.88, 0.07, 0.05]),
            "is_pep": r.random(n_customers) < 0.015,
        }
    )

    # accounts with a long-tail balance and FK integrity
    owner = r.choice(customer_ids, n_accounts, replace=True)
    acct_ids = [f"A{i:07d}" for i in range(1, n_accounts + 1)]
    acct_type = r.choice(["checking", "savings", "credit", "business"], n_accounts, p=[0.45, 0.3, 0.18, 0.07])
    balances = np.where(
        acct_type == "credit",
        -np.abs(r.lognormal(8.0, 1.2, n_accounts)),
        np.abs(r.lognormal(8.6, 1.35, n_accounts)),
    )
    opened = random_dates(r, n_accounts, "2016-01-01", "2024-12-01")
    accounts = pd.DataFrame(
        {
            "account_id": acct_ids,
            "customer_id": owner,
            "account_type": acct_type,
            "balance": add_missing(pd.Series(balances.round(2)), r, 0.008),
            "currency": r.choice(["USD", "EUR", "GBP"], n_accounts, p=[0.85, 0.1, 0.05]),
            "opened_date": opened,
            "status": r.choice(["active", "dormant", "closed"], n_accounts, p=[0.86, 0.09, 0.05]),
            "credit_limit": np.where(acct_type == "credit", np.round(r.lognormal(8.2, 0.8, n_accounts), 2), np.nan),
            "monthly_fee": np.where(acct_type == "basic", 0.0, np.round(r.choice([0, 4.99, 9.99, 24.99], n_accounts), 2)),
        }
    )

    # transactions: seasonal time-of-day and weekday, heavy-tailed amounts, fraud rare
    tx_acct = r.choice(acct_ids, n_transactions, replace=True)
    # simulate timestamps on a 2-year window with weekday/hour seasonality
    days = r.integers(0, 730, n_transactions)
    hour_weights = np.array([1, 1, 1, 1, 1, 2, 4, 8, 14, 16, 15, 14, 16, 18, 17, 16, 18, 22, 24, 20, 14, 8, 4, 2], dtype=float)
    hour = r.choice(np.arange(24), n_transactions, p=hour_weights / hour_weights.sum())
    minute = r.integers(0, 60, n_transactions)
    base = pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D")
    weekday_boost = np.where(base.dayofweek >= 5, 1.25, 1.0)
    ts = base + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    amount = np.round(np.exp(r.normal(3.7, 1.35, n_transactions)) * weekday_boost, 2)
    amount = np.where(r.random(n_transactions) < 0.002, amount * 25, amount)  # rare huge outliers
    channel = r.choice(["card", "transfer", "atm", "online"], n_transactions, p=[0.42, 0.23, 0.17, 0.18])
    merchant = r.choice(
        ["groceries", "restaurant", "travel", "utilities", "electronics", "health", "entertainment", "cash"],
        n_transactions,
        p=[0.25, 0.19, 0.08, 0.12, 0.09, 0.1, 0.12, 0.05],
    )
    # fraud probability increases with online + unusual hour + amount
    fraud_score = (
        -6.0
        + 1.8 * (channel == "online")
        + 1.1 * ((hour < 5) | (hour > 22))
        + 0.00035 * np.minimum(amount, 4000)
        + 1.2 * (merchant == "electronics")
        + r.normal(0, 0.5, n_transactions)
    )
    fraud = (fraud_score > np.quantile(fraud_score, 0.995)).astype(int)
    txn = pd.DataFrame(
        {
            "transaction_id": [f"T{i:09d}" for i in range(1, n_transactions + 1)],
            "account_id": tx_acct,
            "timestamp": ts,
            "amount": amount,
            "merchant_category": merchant,
            "channel": channel,
            "is_flagged": (r.random(n_transactions) < 0.012).astype(int),
            "is_fraud": fraud,
        }
    )
    # make flags correlated with fraud but noisy
    txn.loc[txn["is_fraud"] == 1, "is_flagged"] = (r.random((txn["is_fraud"] == 1).sum()) < 0.72).astype(int)

    loan_owner = r.choice(customer_ids, n_loans, replace=True)
    income_map = customers.set_index("customer_id")["annual_income"]
    risk_map = customers.set_index("customer_id")["risk_score"]
    principal = np.round(np.exp(r.normal(9.2, 0.8, n_loans)), 2)
    rate = np.clip(3.5 + 8 * risk_map.loc[loan_owner].to_numpy() + r.normal(0, 0.7, n_loans), 2.9, 24.0).round(2)
    term = r.choice([12, 24, 36, 48, 60, 84], n_loans, p=[0.1, 0.22, 0.3, 0.16, 0.17, 0.05])
    start = random_dates(r, n_loans, "2019-01-01", "2025-01-01")
    end = start + pd.to_timedelta(term * 30, unit="D")
    default_score = (
        -4.4
        + 0.045 * (principal / 1000)
        + 0.03 * term
        + 6.0 * risk_map.loc[loan_owner].to_numpy()
        - 0.000012 * income_map.loc[loan_owner].to_numpy()
        + r.normal(0, 0.6, n_loans)
    )
    # force a realistic 3% base rate while preserving ranking
    threshold = np.quantile(default_score, 0.97)
    default = (default_score > threshold).astype(int)
    loans = pd.DataFrame(
        {
            "loan_id": [f"L{i:06d}" for i in range(1, n_loans + 1)],
            "customer_id": loan_owner,
            "principal": principal,
            "interest_rate": rate,
            "term_months": term,
            "start_date": start,
            "end_date": end,
            "purpose": r.choice(["home", "auto", "personal", "business", "education"], n_loans, p=[0.2, 0.25, 0.3, 0.15, 0.1]),
            "status": np.where(default == 1, "default", r.choice(["current", "paid"], n_loans, p=[0.7, 0.3])),
            "defaulted": default,
        }
    )
    return {
        "customers": customers,
        "accounts": accounts,
        "transactions": txn,
        "loans": loans,
    }


RELATIONSHIPS = {
    "tables": {
        "customers": {"primary_key": ["customer_id"]},
        "accounts": {
            "primary_key": ["account_id"],
            "foreign_keys": [{"columns": ["customer_id"], "ref_table": "customers", "ref_columns": ["customer_id"]}],
        },
        "transactions": {
            "primary_key": ["transaction_id"],
            "foreign_keys": [{"columns": ["account_id"], "ref_table": "accounts", "ref_columns": ["account_id"]}],
        },
        "loans": {
            "primary_key": ["loan_id"],
            "foreign_keys": [{"columns": ["customer_id"], "ref_table": "customers", "ref_columns": ["customer_id"]}],
        },
    }
}


def write_banking(root: str | Path, seed: int = 7, **sizes: int) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tables = simulate_banking(seed=seed, **sizes)
    db_path = root / "banking.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "CREATE TABLE customers (customer_id TEXT PRIMARY KEY, full_name TEXT, age INTEGER, gender TEXT, region TEXT, city TEXT, segment TEXT, annual_income REAL, risk_score REAL, signup_date TEXT, kyc_status TEXT, is_pep INTEGER)"
        )
        conn.execute(
            "CREATE TABLE accounts (account_id TEXT PRIMARY KEY, customer_id TEXT REFERENCES customers(customer_id), account_type TEXT, balance REAL, currency TEXT, opened_date TEXT, status TEXT, credit_limit REAL, monthly_fee REAL)"
        )
        conn.execute(
            "CREATE TABLE transactions (transaction_id TEXT PRIMARY KEY, account_id TEXT REFERENCES accounts(account_id), timestamp TEXT, amount REAL, merchant_category TEXT, channel TEXT, is_flagged INTEGER, is_fraud INTEGER)"
        )
        conn.execute(
            "CREATE TABLE loans (loan_id TEXT PRIMARY KEY, customer_id TEXT REFERENCES customers(customer_id), principal REAL, interest_rate REAL, term_months INTEGER, start_date TEXT, end_date TEXT, purpose TEXT, status TEXT, defaulted INTEGER)"
        )
        for name, df in tables.items():
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
                elif out[c].dtype == bool:
                    out[c] = out[c].astype(int)
            out.where(pd.notna(out), None).to_sql(name, conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()
    (root / "relationships.json").write_text(json.dumps(RELATIONSHIPS, indent=2), encoding="utf-8")
    # also leave CSV copies for the folder-of-CSVs ingestion path
    csv_dir = root / "banking_csv"
    csv_dir.mkdir(exist_ok=True)
    for name, df in tables.items():
        df.to_csv(csv_dir / f"{name}.csv", index=False)
    (csv_dir / "relationships.json").write_text(json.dumps(RELATIONSHIPS, indent=2), encoding="utf-8")
    return db_path


def write_unseen_schemas(root: str | Path, seed: int = 11) -> Path:
    """Three differently shaped CSVs used to prove BYO-schema operation."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    r = rng(seed)
    n = 280
    # 1: logistics events, mostly numeric + dates, strong skew
    distance = np.abs(r.lognormal(4.2, 1.1, n))
    late = (distance / (r.uniform(40, 100, n)) > 1.05).astype(int)
    pd.DataFrame(
        {
            "shipment_ref": [f"SHP-{i:05d}" for i in range(n)],
            "carrier": r.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1]),
            "distance_km": distance.round(2),
            "weight_kg": np.abs(r.normal(12, 9, n)).round(2),
            "declared_value": np.exp(r.normal(5.0, 1.0, n)).round(2),
            "ship_date": random_dates(r, n, "2024-01-01", "2024-12-31").strftime("%Y-%m-%d"),
            "is_late": late,
        }
    ).to_csv(root / "logistics_unseen.csv", index=False)
    # 2: survey with many categoricals, missingness, sentinel -999
    score = r.normal(70, 14, n)
    df = pd.DataFrame(
        {
            "respondent": [f"R{i:04d}" for i in range(n)],
            "age_band": r.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n, p=[0.15, 0.28, 0.25, 0.2, 0.12]),
            "region_code": r.choice(list("ABCDE"), n),
            "satisfaction": r.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
            "nps": np.clip(r.normal(20, 45, n), -100, 100).round(0),
            "monthly_spend": np.where(r.random(n) < 0.07, -999, np.abs(r.lognormal(5, 0.6, n)).round(2)),
            "score_continuous": score.round(3),
            "churned": (score < np.quantile(score, 0.18)).astype(int),
            "comment": r.choice(["", "good service", "slow support", "too expensive", "will renew"], n),
        }
    )
    df.to_csv(root / "survey_unseen.csv", index=False)
    # 3: sensor readings with datetime + bool + rare fault
    ts = pd.date_range("2024-01-01", periods=n * 3, freq="h")[:n]
    temp = 21 + 4 * np.sin(np.arange(n) / 18) + r.normal(0, 0.7, n)
    fault = (r.random(n) < 0.035).astype(int)
    pd.DataFrame(
        {
            "reading_id": [f"RD{i:05d}" for i in range(n)],
            "device": r.choice(["pump-1", "pump-2", "valve-7"], n),
            "reading_ts": ts,
            "temperature_c": np.round(temp, 3),
            "pressure_kpa": np.round(101 + r.normal(0, 3, n), 2),
            "vibration_mm_s": np.round(np.abs(r.lognormal(-1.0, 0.8, n)), 3),
            "maintenance_due": (r.random(n) < 0.12),
            "fault": fault,
        }
    ).to_csv(root / "sensors_unseen.csv", index=False)
    return root

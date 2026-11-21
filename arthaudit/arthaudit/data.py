"""UCI Statlog German Credit data, with strict schema checks.

Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Target 1 means historical good risk, 2 means historical bad risk.
These are historical labels, not objective creditworthiness or approval records.
"""
from __future__ import annotations
import io
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/144/data.csv"
COLUMNS = ["checking_status", "duration", "credit_history", "purpose", "credit_amount",
           "savings_status", "employment", "installment_rate", "personal_status_sex",
           "other_debtors", "residence_since", "property", "age", "other_installment_plans",
           "housing", "existing_credits", "job", "num_dependents", "telephone",
           "foreign_worker", "class"]
NUMERIC = ["duration", "credit_amount", "installment_rate", "residence_since", "age",
           "existing_credits", "num_dependents", "class"]


def parse_credit_csv(raw: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(raw))
    expected = [f"Attribute{i}" for i in range(1, 21)] + ["class"]
    if list(df.columns) != expected:
        raise ValueError("Unexpected UCI CSV schema; refusing to guess column positions.")
    df.columns = COLUMNS
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df.isna().any().any() or not set(df["class"].unique()) <= {1, 2}:
        raise ValueError("Missing values or unknown target labels in credit data.")
    return df


def load_german_credit(url: str = UCI_CSV_URL) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return parse_credit_csv(resp.read().decode("utf-8"))


def derive_protected(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not set(out["class"]) <= {1, 2}:
        raise ValueError("Expected original class labels 1 or 2.")
    out["target"] = (out["class"] == 1).astype(int)
    mapping = {"A91": "male", "A92": "female", "A93": "male",
               "A94": "male", "A95": "female"}
    out["sex"] = out["personal_status_sex"].map(mapping)
    out["is_foreign_worker"] = out["foreign_worker"].map({"A201": "yes", "A202": "no"})
    if out[["sex", "is_foreign_worker"]].isna().any().any():
        raise ValueError("Unknown protected-attribute codes; refusing silent assignment.")
    # An illustrative audit grouping, not a legal or biological threshold.
    out["age_group"] = np.where(out["age"] >= 25, "25+", "under_25")
    return out


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw features. Exclude all audited attributes, including age itself.

    Categorical encoding and scaling are fit on the training split only.
    Exclusion alone does not guarantee fair predictions.
    """
    drop = {"class", "target", "sex", "age_group", "is_foreign_worker",
            "personal_status_sex", "foreign_worker", "age"}
    return df[[c for c in COLUMNS if c not in drop]].copy()

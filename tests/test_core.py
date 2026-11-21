import numpy as np
import pandas as pd
import pytest

from arthaudit.audit import train_and_audit
from arthaudit.data import COLUMNS, derive_protected, feature_frame
from arthaudit.fairness import group_fairness
from arthaudit.proxies import find_proxies


def test_fairness_parity_case():
    y_true = [1, 0, 1, 0] * 10
    y_pred = [1, 0, 1, 0] * 10
    groups = ["a", "a", "b", "b"] * 10
    r = group_fairness(y_true, y_pred, groups, "sex")
    assert r.demographic_parity_difference == pytest.approx(0.0)
    assert r.disparate_impact_ratio == pytest.approx(1.0)
    assert r.equal_opportunity_difference == pytest.approx(0.0)


def test_fairness_skewed_case():
    y_true = [1] * 40
    y_pred = [1] * 20 + [0] * 20
    groups = ["adv"] * 20 + ["dis"] * 20
    r = group_fairness(y_true, y_pred, groups, "sex")
    assert r.demographic_parity_difference == pytest.approx(1.0)
    assert r.disparate_impact_ratio == pytest.approx(0.0)
    assert r.equal_opportunity_difference == pytest.approx(1.0)


def test_tpr_none_when_no_positives():
    r = group_fairness([0, 0], [0, 0], ["x", "y"], "sex")
    assert all(g.tpr is None for g in r.groups)
    assert r.equal_opportunity_difference is None


def _synthetic_credit(n=200, seed=0):
    rng = np.random.default_rng(seed)
    pss = rng.choice(["A91", "A92", "A93", "A95"], n)
    age = rng.integers(19, 70, n)
    duration = rng.integers(6, 60, n)
    amount = rng.integers(500, 15000, n)
    # good risk more likely for smaller amounts and shorter durations
    good = (amount < 8000).astype(int) ^ (duration > 48).astype(int)
    rows = {
        "checking_status": rng.choice(["A11", "A12", "A14"], n),
        "duration": duration,
        "credit_history": rng.choice(["A30", "A32"], n),
        "purpose": rng.choice(["A40", "A43"], n),
        "credit_amount": amount,
        "savings_status": rng.choice(["A61", "A65"], n),
        "employment": rng.choice(["A73", "A75"], n),
        "installment_rate": rng.integers(1, 4, n),
        "personal_status_sex": pss,
        "other_debtors": rng.choice(["A101", "A103"], n),
        "residence_since": rng.integers(1, 4, n),
        "property": rng.choice(["A121", "A124"], n),
        "age": age,
        "other_installment_plans": rng.choice(["A141", "A143"], n),
        "housing": rng.choice(["A152", "A153"], n),
        "existing_credits": rng.integers(1, 3, n),
        "job": rng.choice(["A172", "A173"], n),
        "num_dependents": rng.integers(1, 3, n),
        "telephone": rng.choice(["A191", "A192"], n),
        "foreign_worker": rng.choice(["A201", "A202"], n),
        "class": np.where(good == 1, 1, 2),
    }
    df = pd.DataFrame(rows)[COLUMNS]
    return df


def test_derive_protected():
    df = derive_protected(_synthetic_credit(50))
    assert set(df["sex"]) <= {"male", "female"}
    assert set(df["age_group"]) <= {"under_25", "25+"}
    assert set(df["is_foreign_worker"]) <= {"yes", "no"}
    assert set(df["target"].unique()) <= {0, 1}
    # A92/A95 are female, others male
    df2 = pd.DataFrame({"personal_status_sex": ["A91", "A92", "A93", "A95"],
                        "age": [30, 30, 20, 30], "foreign_worker": ["A201"] * 4,
                        "class": [1, 2, 1, 2]})
    d2 = derive_protected(df2)
    assert list(d2["sex"]) == ["male", "female", "male", "female"]
    assert list(d2["age_group"]) == ["25+", "25+", "under_25", "25+"]


def test_feature_frame_excludes_protected():
    df = derive_protected(_synthetic_credit(30))
    X = feature_frame(df)
    for col in ["sex", "age_group", "is_foreign_worker",
                "personal_status_sex", "foreign_worker", "class", "target", "age"]:
        assert col not in X.columns
    assert X["checking_status"].dtype == object


def test_find_proxies_detects_strong_link():
    n = 100
    X = pd.DataFrame({"tracking_feature": [1] * 50 + [0] * 50,
                      "noise": np.random.default_rng(1).integers(0, 100, n)})
    protected = pd.DataFrame({"sex": ["female"] * 50 + ["male"] * 50})
    hits = find_proxies(X, protected, threshold=0.30)
    assert any(h.feature == "tracking_feature" for h in hits)
    assert not any(h.feature == "noise" for h in hits)


def test_train_and_audit_runs_on_synthetic():
    df = derive_protected(_synthetic_credit(300))
    result = train_and_audit(df)
    assert 0.0 <= result.accuracy <= 1.0
    assert len(result.fairness) == 3
    assert {f.attribute for f in result.fairness} == {
        "sex", "age_group", "is_foreign_worker"}
    assert result.n_test == 90
    assert all(isinstance(f.disparate_impact_ratio, float)
               for f in result.fairness)

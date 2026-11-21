"""Regression tests for undefined metrics, encoding and real-schema parsing."""
import numpy as np
import pandas as pd
import pytest
from arthaudit.data import parse_credit_csv, derive_protected, feature_frame
from arthaudit.fairness import group_fairness, wilson
from arthaudit.explain import explain_applicant
from arthaudit.audit import train_and_audit, threshold_study
from arthaudit.proxies import find_proxies
from test_core import _synthetic_credit


def test_real_schema_header_is_not_a_data_row():
    raw = _synthetic_credit(40)
    raw.columns = [f"Attribute{i}" for i in range(1,21)] + ["class"]
    parsed = parse_credit_csv(raw.to_csv(index=False))
    assert len(parsed) == 40
    assert parsed.age.dtype.kind in "iu"
    assert set(derive_protected(parsed).target) <= {0,1}


def test_unknown_schema_refused():
    with pytest.raises(ValueError):
        parse_credit_csv("unexpected,class\n1,2")


def test_unknown_protected_codes_refused():
    raw = _synthetic_credit(20)
    raw.loc[0,"personal_status_sex"] = "UNKNOWN"
    with pytest.raises(ValueError):
        derive_protected(raw)


def test_all_zero_selection_is_not_perfect_fairness():
    result=group_fairness([1,0,1,0],[0,0,0,0],["a","a","b","b"],"group")
    assert result.disparate_impact_ratio is None
    assert result.demographic_parity_difference == 0


def test_single_group_has_no_between_group_verdict():
    r=group_fairness([1,0],[1,0],["a","a"],"group")
    assert r.demographic_parity_difference is None
    assert r.disparate_impact_ratio is None
    assert r.equal_opportunity_difference is None


def test_one_missing_positive_denominator_invalidates_full_tpr_gap():
    r=group_fairness([1,0,0],[1,0,0],["a","b","c"],"group")
    assert r.equal_opportunity_difference is None


@pytest.mark.parametrize("yt,yp,gs",[([],[],[]),([1],[1,0],["a"]),([2],[1],["a"]),([1],[1],[None]),([1],[np.nan],["a"])])
def test_invalid_metric_inputs_raise(yt,yp,gs):
    with pytest.raises(ValueError):
        group_fairness(yt,yp,gs,"group")


def test_wilson_intervals_include_estimate_and_endpoints():
    for k,n in [(0,10),(5,10),(10,10),(1,100)]:
        low,high=wilson(k,n)
        assert 0 <= low <= k/n <= high <= 1
    assert wilson(0,0)==(None,None)


def test_nominal_proxy_screen_does_not_depend_on_code_names():
    groups=pd.DataFrame({"g":["a"]*40+["b"]*40})
    x=pd.DataFrame({"job":["A"]*40+["Z"]*40})
    first=find_proxies(x,groups)[0]
    second=find_proxies(x.replace({"A":"zebra","Z":"apple"}),groups)[0]
    assert first.correlation == pytest.approx(1)
    assert second.correlation == pytest.approx(first.correlation)
    assert first.method.startswith("Cramer's")


def test_train_onehot_and_attribution_reconstruction():
    result=train_and_audit(derive_protected(_synthetic_credit(300)))
    assert "age" not in result.feature_names
    assert any("checking_status_A" in f for f in result.feature_names)
    for i in [0,8,20]:
        vals,base,logit=explain_applicant(result,i)
        assert base+vals.sum()==pytest.approx(logit)
        assert 1/(1+np.exp(-logit))==pytest.approx(result.probabilities[i])


def test_reproducible_split_model_and_metrics():
    df=derive_protected(_synthetic_credit(300))
    a,b=train_and_audit(df),train_and_audit(df)
    assert np.array_equal(a.y_pred,b.y_pred)
    assert a.fairness==b.fairness
    assert threshold_study(a).shape==(5,5)


def test_unseen_category_is_supported_by_train_fitted_encoder():
    r=train_and_audit(derive_protected(_synthetic_credit(300)))
    row=r.X_test.iloc[[0]].copy()
    row["checking_status"]="not_seen_in_training"
    assert np.isfinite(r.model.predict_proba(r.scaler.transform(row))).all()


def test_all_audited_attributes_excluded():
    x=feature_frame(derive_protected(_synthetic_credit(40)))
    assert set(x).isdisjoint({"age","sex","age_group","personal_status_sex","foreign_worker","is_foreign_worker","target","class"})

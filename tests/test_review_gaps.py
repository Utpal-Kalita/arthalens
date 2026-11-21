"""Tests closing review gaps: threshold bounds, PDF build, explanation
bounds, and a Fairlearn cross-check (skipped when fairlearn is absent)."""
import numpy as np
import pytest
from arthaudit.audit import train_and_audit
from arthaudit.data import derive_protected
from arthaudit.explain import explain_applicant
from arthaudit.fairness import group_fairness
from arthaudit.report import build_scorecard
from test_core import _synthetic_credit


def test_threshold_bounds_accepted_and_outside_refused():
    df = derive_protected(_synthetic_credit(300))
    low = train_and_audit(df, threshold=0.0)
    high = train_and_audit(df, threshold=1.0)
    assert set(low.y_pred) == {1}
    assert set(high.y_pred) == {0}
    with pytest.raises(ValueError):
        train_and_audit(df, threshold=-0.01)
    with pytest.raises(ValueError):
        train_and_audit(df, threshold=1.01)


def test_scorecard_pdf_builds(tmp_path):
    result = train_and_audit(derive_protected(_synthetic_credit(300)))
    out = build_scorecard(result, str(tmp_path / "scorecard.pdf"))
    assert (tmp_path / "scorecard.pdf").stat().st_size > 1000
    assert out.endswith("scorecard.pdf")


def test_explain_applicant_rejects_out_of_range_row():
    result = train_and_audit(derive_protected(_synthetic_credit(300)))
    with pytest.raises(IndexError):
        explain_applicant(result, result.n_test)


def test_group_fairness_matches_fairlearn():
    fm = pytest.importorskip("fairlearn.metrics")
    rng = np.random.default_rng(7)
    n = 400
    y_true = rng.integers(0, 2, n)
    y_pred = (rng.random(n) < 0.6).astype(int)
    groups = np.where(rng.random(n) < 0.4, "b", "a")
    ours = group_fairness(y_true, y_pred, groups, "sex")
    mf = fm.MetricFrame(
        metrics={"sr": fm.selection_rate, "tpr": fm.true_positive_rate},
        y_true=y_true, y_pred=y_pred, sensitive_features=groups)
    for g in ours.groups:
        row = mf.by_group.loc[g.group]
        assert g.selection_rate == pytest.approx(row["sr"])
        assert g.tpr == pytest.approx(row["tpr"])
    assert ours.demographic_parity_difference == pytest.approx(
        mf.by_group["sr"].max() - mf.by_group["sr"].min())
    assert ours.equal_opportunity_difference == pytest.approx(
        mf.by_group["tpr"].max() - mf.by_group["tpr"].min())

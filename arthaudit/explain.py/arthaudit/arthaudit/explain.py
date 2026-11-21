"""Exact linear SHAP in log-odds space, using an independent training-mean background.

For this additive logistic model: phi_j = beta_j * (x_j - mean_train_j).
The sum plus base value must reconstruct the decision logit. This is
interventional linear attribution, not causal explanation. Correlated
features and the chosen background affect interpretation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def global_importance(model, feature_names: list[str]) -> pd.Series:
    return pd.Series(np.abs(model.coef_[0]), index=feature_names).sort_values(ascending=False)


def explain_applicant(result, row_index: int) -> tuple[pd.Series, float, float]:
    row = result.X_test.iloc[[row_index]]
    x = result.scaler.transform(row)[0]
    values = result.model.coef_[0] * (x - result.background)
    base = float(result.model.intercept_[0] + np.dot(result.model.coef_[0], result.background))
    logit = float(result.model.decision_function(result.scaler.transform(row))[0])
    if not np.isclose(base + values.sum(), logit):
        raise ArithmeticError("Explanation failed its reconstruction check.")
    return pd.Series(values, index=result.feature_names), base, logit

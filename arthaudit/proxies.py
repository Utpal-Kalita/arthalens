"""Exploratory feature/group associations, not proof of proxy discrimination."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 0.30

@dataclass
class ProxyHit:
    feature: str
    protected_attribute: str
    correlation: float
    method: str = "absolute Pearson (binary group)"


def find_proxies(X: pd.DataFrame, protected: pd.DataFrame,
                 threshold: float = DEFAULT_THRESHOLD) -> list[ProxyHit]:
    """Numeric: max absolute Pearson against one-vs-rest groups.

    Nominal: Cramer's V from a contingency table, without ordinal coding.
    Scores screen associations only; they are not causal evidence or tests
    of whether the fitted model actually uses a feature as a proxy.
    """
    if len(X) != len(protected) or len(X) < 2:
        raise ValueError("Need aligned feature and group rows (at least two).")
    if X.isna().any().any() or protected.isna().any().any():
        raise ValueError("Missing values must be handled before association screening.")
    hits = []
    for attr in protected:
        group = protected[attr].astype(str).to_numpy()
        if len(np.unique(group)) < 2:
            continue
        for col in X:
            x = X[col].to_numpy()
            if len(pd.unique(x)) < 2:
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                scores = [abs(float(np.corrcoef(x.astype(float), (group == g).astype(float))[0, 1]))
                          for g in np.unique(group)]
                score, method = max(scores), "absolute Pearson (one-vs-rest)"
            else:
                table = pd.crosstab(pd.Series(x), pd.Series(group)).to_numpy(dtype=float)
                expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
                chi2 = float(((table - expected) ** 2 / expected).sum())
                score = float(np.sqrt(chi2 / (table.sum() * min(table.shape[0] - 1, table.shape[1] - 1))))
                method = "Cramer's V (uncorrected)"
            if score >= threshold:
                hits.append(ProxyHit(col, attr, score, method))
    return sorted(hits, key=lambda h: -h.correlation)

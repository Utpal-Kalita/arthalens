"""Binary outcome disparity measures with denominators and Wilson intervals.

Metric definitions: https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html
No metric here certifies fairness, legality or RBI compliance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class GroupMetric:
    group: str
    count: int
    selection_rate: float
    tpr: float | None
    positive_count: int = 0
    selection_low: float = 0.0
    selection_high: float = 1.0
    tpr_low: float | None = None
    tpr_high: float | None = None

@dataclass
class FairnessResult:
    attribute: str
    groups: list[GroupMetric] = field(default_factory=list)
    demographic_parity_difference: float | None = None
    disparate_impact_ratio: float | None = None
    equal_opportunity_difference: float | None = None


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p, z = k / n, 1.959963984540054
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0., float(middle - half)), (1. if k == n else min(1., float(middle + half)))


def group_fairness(y_true, y_pred, groups, attribute: str) -> FairnessResult:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    gs = pd.Series(groups)
    if any(a.ndim != 1 for a in [yt, yp]) or not (len(yt) == len(yp) == len(gs)) or not len(yt):
        raise ValueError("Provide aligned, nonempty one-dimensional arrays.")
    if gs.isna().any() or not set(np.unique(yt)) <= {0, 1} or not set(np.unique(yp)) <= {0, 1}:
        raise ValueError("Labels must be binary; group values cannot be missing.")
    g = gs.astype(str).to_numpy()
    result = FairnessResult(attribute)
    for value in sorted(np.unique(g)):
        mask = g == value
        n, k = int(mask.sum()), int(yp[mask].sum())
        positives = mask & (yt == 1)
        npos, tp = int(positives.sum()), int(yp[positives].sum())
        sl, sh = wilson(k, n)
        tl, th = wilson(tp, npos)
        result.groups.append(GroupMetric(value, n, k / n, tp / npos if npos else None,
                                         npos, sl, sh, tl, th))
    if len(result.groups) >= 2:
        rates = [x.selection_rate for x in result.groups]
        result.demographic_parity_difference = max(rates) - min(rates)
        result.disparate_impact_ratio = min(rates) / max(rates) if max(rates) > 0 else None
        tprs = [x.tpr for x in result.groups]
        # A missing group TPR means the full between-group gap is undefined.
        if all(t is not None for t in tprs):
            result.equal_opportunity_difference = max(tprs) - min(tprs)
    return result

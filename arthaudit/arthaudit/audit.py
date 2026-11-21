"""Reproducible train-only preprocessing, held-out model audit and threshold study."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import derive_protected, feature_frame, load_german_credit
from .fairness import FairnessResult, group_fairness
from .proxies import ProxyHit, find_proxies

PROTECTED_ATTRIBUTES = ["sex", "age_group", "is_foreign_worker"]

@dataclass
class AuditResult:
    model_name: str
    n_rows: int
    n_test: int
    accuracy: float
    fairness: list[FairnessResult] = field(default_factory=list)
    proxies: list[ProxyHit] = field(default_factory=list)
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    groups_test: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    model: object | None = None
    scaler: object | None = None
    background: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    probabilities: np.ndarray | None = None
    threshold: float = 0.5
    balanced_accuracy: float = 0.0
    roc_auc: float = 0.0
    cost_per_case: float = 0.0
    seed: int = 42


def train_and_audit(df: pd.DataFrame | None = None, random_state: int = 42,
                    threshold: float = 0.5) -> AuditResult:
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one.")
    if df is None:
        df = derive_protected(load_german_credit())
    X, y, groups = feature_frame(df), df["target"].astype(int).to_numpy(), df[PROTECTED_ATTRIBUTES].copy()
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        X, y, groups, test_size=0.3, random_state=random_state, stratify=y)
    cats = list(X.select_dtypes(include=["object", "string"]).columns)
    nums = [c for c in X if c not in cats]
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), nums),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)
    ], verbose_feature_names_out=False)
    train = prep.fit_transform(Xtr)
    test = prep.transform(Xte)
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state).fit(train, ytr)
    prob = model.predict_proba(test)[:, 1]
    pred = (prob >= threshold).astype(int)
    # UCI cost matrix: bad predicted good costs 5; good predicted bad costs 1.
    cost = (5 * int(((yte == 0) & (pred == 1)).sum()) + int(((yte == 1) & (pred == 0)).sum())) / len(yte)
    result = AuditResult("Logistic regression", len(df), len(Xte), float(accuracy_score(yte, pred)),
                         y_test=yte, y_pred=pred, groups_test=gte.reset_index(drop=True),
                         X_test=Xte.reset_index(drop=True), model=model, scaler=prep,
                         background=train.mean(axis=0), feature_names=list(prep.get_feature_names_out()),
                         probabilities=prob, threshold=threshold,
                         balanced_accuracy=float(balanced_accuracy_score(yte, pred)),
                         roc_auc=float(roc_auc_score(yte, prob)), cost_per_case=cost, seed=random_state)
    result.fairness = [group_fairness(yte, pred, gte[attr], attr) for attr in PROTECTED_ATTRIBUTES]
    # Descriptive associations use TRAINING ROWS ONLY, so holdout never informs preprocessing or flags.
    result.proxies = find_proxies(Xtr.reset_index(drop=True), gtr.reset_index(drop=True))
    return result


def threshold_study(result: AuditResult, attribute: str = "sex") -> pd.DataFrame:
    """Exploratory on the fixed holdout, never presented as tuned validation."""
    rows = []
    for threshold in [0.2, 0.35, 0.5, 0.65, 0.8]:
        pred = (result.probabilities >= threshold).astype(int)
        fr = group_fairness(result.y_test, pred, result.groups_test[attribute], attribute)
        rows.append({"threshold": threshold,
                     "accuracy": float(accuracy_score(result.y_test, pred)),
                     "selection_rate": float(pred.mean()),
                     "parity_gap": fr.demographic_parity_difference,
                     "opportunity_gap": fr.equal_opportunity_difference})
    return pd.DataFrame(rows)

"""ArthaLens Credit Audit: audit a credit-scoring model before a bank trusts it."""

from .audit import AuditResult, train_and_audit
from .data import derive_protected, feature_frame, load_german_credit
from .fairness import FairnessResult, GroupMetric, group_fairness
from .proxies import ProxyHit, find_proxies
from .report import build_scorecard

__all__ = [
    "AuditResult", "train_and_audit",
    "derive_protected", "feature_frame", "load_german_credit",
    "FairnessResult", "GroupMetric", "group_fairness",
    "ProxyHit", "find_proxies",
    "build_scorecard",
]

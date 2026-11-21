"""Public-data provenance and a machine-readable audit summary.
Run: python -m examples.export_evidence
"""
import hashlib
import json
import urllib.request
from dataclasses import asdict
from pathlib import Path
from arthaudit.data import UCI_CSV_URL, parse_credit_csv, derive_protected
from arthaudit.audit import train_and_audit

with urllib.request.urlopen(UCI_CSV_URL, timeout=60) as response:
    raw = response.read()
result = train_and_audit(derive_protected(parse_credit_csv(raw.decode())))
evidence = {
    "dataset_url": UCI_CSV_URL,
    "dataset_sha256": hashlib.sha256(raw).hexdigest(),
    "dataset_rows": result.n_rows,
    "test_rows": result.n_test,
    "seed": result.seed,
    "threshold": result.threshold,
    "accuracy": result.accuracy,
    "balanced_accuracy": result.balanced_accuracy,
    "roc_auc": result.roc_auc,
    "uci_error_cost_per_case": result.cost_per_case,
    "fairness": [asdict(fr) for fr in result.fairness],
    "training_association_flags": [asdict(h) for h in result.proxies],
    "limitations": ["Historical German benchmark, not an Indian lending validation.",
                    "Historical labels can encode bias; metrics are not fairness certificates.",
                    "Per-group Wilson intervals do not quantify disparity-gap uncertainty."]
}
out = Path("out"); out.mkdir(exist_ok=True)
(out / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))
print(json.dumps({k:evidence[k] for k in ["dataset_sha256","dataset_rows","test_rows","accuracy","balanced_accuracy","roc_auc"]},indent=2))

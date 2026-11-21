"""Descriptive split-stability sweep.

Runs the documented train_and_audit pipeline across 30 split seeds to show
how much the headline figures move. Deterministic split seeds.

Run:  python -m examples.seed_sweep
Writes out/stability.json and rebuilds out/scorecard.pdf with the
stability block included.
"""
import json
from pathlib import Path

import numpy as np

from arthaudit import build_scorecard, train_and_audit
from arthaudit.data import derive_protected, load_german_credit

OUT = Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)

SEEDS = list(range(1, 31))  # 30 seeds; the documented benchmark is seed 42


def headline(result):
    by_attr = {fr.attribute: fr for fr in result.fairness}
    ratios = [fr.disparate_impact_ratio for fr in result.fairness
              if fr.disparate_impact_ratio is not None]
    return {
        "accuracy": result.accuracy,
        "sex_selection_gap": by_attr["sex"].demographic_parity_difference,
        "age_tpr_gap": by_attr["age_group"].equal_opportunity_difference,
        "foreign_selection_gap": by_attr["is_foreign_worker"].demographic_parity_difference,
        "min_disparate_impact": min(ratios) if ratios else None,
    }


def summarize(values):
    values = [v for v in values if v is not None]
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
            "min": float(arr.min()), "max": float(arr.max()), "n": int(arr.size)}


def main():
    df = derive_protected(load_german_credit())
    sweep = {k: [] for k in
             ["accuracy", "sex_selection_gap", "age_tpr_gap",
              "foreign_selection_gap", "min_disparate_impact"]}
    for seed in SEEDS:
        h = headline(train_and_audit(df, random_state=seed))
        for k, v in h.items():
            sweep[k].append(v)

    base = train_and_audit(df, random_state=42)
    stability = {
        "sweep_seeds": [SEEDS[0], SEEDS[-1]],
        "sweep": {k: summarize(v) for k, v in sweep.items()},
    }
    path = OUT / "stability.json"
    path.write_text(json.dumps(stability, indent=2))

    print(f"Seeds {SEEDS[0]}..{SEEDS[-1]} ({len(SEEDS)} runs):")
    for k, s in stability["sweep"].items():
        print(f"  {k:>24}: mean {s['mean']:.3f} sd {s['sd']:.3f} "
              f"range {s['min']:.3f} to {s['max']:.3f}")
    pdf = build_scorecard(base, str(OUT / "scorecard.pdf"), stability=stability)
    print(f"\nStability written to {path}")
    print(f"Scorecard rebuilt at {pdf}")


if __name__ == "__main__":
    main()

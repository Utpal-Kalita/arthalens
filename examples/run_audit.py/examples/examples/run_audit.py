"""End-to-end example: download German Credit, train, audit, scorecard.

Run:  python examples/run_audit.py
Writes the PDF scorecard to out/scorecard.pdf.
"""

from pathlib import Path

from arthaudit import build_scorecard, train_and_audit

OUT = Path(__file__).resolve().parent.parent / "out"
OUT.mkdir(exist_ok=True)

result = train_and_audit()

print(f"Model: {result.model_name}")
print(f"Rows: {result.n_rows} (test: {result.n_test}), accuracy: {result.accuracy:.3f}")
print()
for fr in result.fairness:
    print(f"[{fr.attribute}]")
    for gm in fr.groups:
        tpr = "-" if gm.tpr is None else f"{gm.tpr:.3f}"
        print(f"  {gm.group:>10}: n={gm.count:<4} selected={gm.selection_rate:.3f} tpr={tpr}")
    eo = ("-" if fr.equal_opportunity_difference is None
          else f"{fr.equal_opportunity_difference:.3f}")
    print(f"  parity diff: {fr.demographic_parity_difference:.3f} | "
          f"disparate impact: {fr.disparate_impact_ratio:.3f} | "
          f"equal-opportunity diff: {eo}")
print()
print("Proxy flags (|r| >= 0.30):")
for h in result.proxies[:8]:
    print(f"  {h.feature} ~ {h.protected_attribute} (r={h.correlation:+.2f})")
if not result.proxies:
    print("  none")

pdf = build_scorecard(result, str(OUT / "scorecard.pdf"))
print(f"\nScorecard written to {pdf}")

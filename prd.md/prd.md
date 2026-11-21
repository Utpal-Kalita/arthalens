# ArthaLens Credit Audit, product requirements

## Problem

A credit model can post a strong accuracy number while selecting and missing applicants at very different rates across groups. Small lenders rarely have the tooling or the statistical help to see that before a model reaches borrowers. Fairness tooling that does exist tends to print metrics without denominators, sample-size context or the limits that make a number reviewable.

## Users

- A risk or compliance reviewer at a small lender or NBFC checking a credit-scoring model before sign-off.
- A data scientist who needs to hand a reviewer evidence, not just a score.

## What the product does

1. Trains a reproducible baseline credit model on a public historical benchmark (UCI Statlog German Credit, id 144) and audits it on a held-out split.
2. Audits an existing model's binary predictions from a strict three-column CSV without retraining.
3. Reports selection-rate gaps, min/max selection ratios and true-positive-rate gaps across sex (source-coded), age group (illustrative under-25 split) and foreign-worker status, always with group sizes, positive-label counts and 95% Wilson intervals.
4. Screens training features for association with protected attributes (absolute Pearson for numeric, Cramer's V for nominal), labelled exploratory, never "proxy-free" certification.
5. Explains one held-out prediction with exact additive linear contributions in log-odds, with a numeric reconstruction check.
6. Exports a one-page PDF audit summary that carries its own limitations.

## Hard requirements

- Free, open-source dependencies only. No paid API, no hosted model, no bank connection.
- No real customer data in the demo path; uploads are anonymous numeric labels and are not intentionally persisted.
- No compliance certification claims, no 80% pass/fail badge, no causal discrimination claims.
- Every disparity number ships with its denominator and a stated uncertainty or instability caveat.

## Non-goals

- Lending decisions, credit scoring for real applicants, fairness mitigation, regulator filings.
- General-purpose explainability across model families (the explanation is exact for the included linear baseline only).

## Success for the hackathon

- A judge can run the app locally in minutes and reproduce the published numbers from seed 42.
- The demo video shows the app working, including the small-group warning and an undefined metric, without slides-only narration.
- README states models, APIs and data sources precisely.

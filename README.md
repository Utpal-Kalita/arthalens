# ArthaLens Credit Audit

A small, local-first workbench for reviewing binary credit-model predictions. It puts group disparities, denominators, uncertainty and one-record explanations in the same review flow, then exports a one-page evidence summary.

It is built for a risk or compliance reviewer at a small lender or NBFC checking a model before sign-off. "Selection" here means a model good-risk prediction at the displayed threshold, not an observed approval. ArthaLens brings per-group denominators and rate intervals, checked one-record explanations, a one-page evidence summary and split-sensitivity reporting into one local review workflow. It builds on standard fairness definitions rather than claiming new metrics.

**Prototype, not a lending service or compliance certificate.** No bank or Indian borrower portfolio has been tested. The included example uses the historical UCI Statlog German Credit benchmark.

## What works

- Reproducible logistic-regression baseline with a 70/30 stratified split, seed 42, train-only one-hot encoding and numeric scaling.
- Selection-rate difference, minimum/maximum selection-rate ratio and true-positive-rate difference across source-coded sex, illustrative age groups and foreign-worker status.
- Per-group sample counts, positive-label counts and 95% Wilson intervals. Undefined comparisons stay undefined; zero selections do not become a perfect ratio.
- Exploratory training-split feature association screening using absolute Pearson correlation for numeric features and Cramer's V for nominal features.
- One-record linear, independent-background SHAP-style log-odds contributions, with an exact reconstruction check.
- Threshold exploration on the fixed holdout. It shows tradeoffs; it is not a deployment-policy optimizer.
- Split-stability reporting: a 30-seed sweep for the headline gaps (`examples/seed_sweep.py`), because a gap measured on one split is not an established finding.
- Existing-model review through a strict three-column prediction CSV.
- One-page downloadable PDF summary with limitations attached to the numbers.

## Run

Tested with Python 3.10. Create a fresh environment rather than installing into a system environment.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The first benchmark run downloads the UCI CSV over HTTPS. An internet connection is needed for that download; the CSV-audit path can run without it. The display font has an optional Google Fonts stylesheet; system fonts are the fallback. No paid service, API key or bank connection is required.

### CLI and tests

```bash
python -m pytest tests -q
python -m examples.run_audit
```

The CLI downloads the real benchmark and writes `out/scorecard.pdf`. Tests generate explicitly synthetic fixtures and do not require internet access. They check schema parsing, protected-code mappings, invalid inputs, undefined metric cases, category handling, train/test consistency, reproducibility, nominal association invariance, explanation reconstruction, threshold bounds, PDF generation and explanation row bounds. If `fairlearn` is installed, one extra test cross-checks the group metrics against its reference implementation.

## Audit your model's predictions

Choose **Audit prediction CSV**. The file must be under 2 MB and contain exactly:

```csv
y_true,y_pred,group
1,1,A
1,0,A
0,1,B
0,0,B
```

Both labels must be 0 or 1, with **1 meaning the favourable outcome**. `y_true` is the chosen historical outcome label, `y_pred` is the model's binary prediction and `group` is the audit group. Missing values and nonbinary labels are refused. The on-screen download is a labelled toy dataset, not benchmark evidence.

Do not upload names, account numbers or other identifiers. The application does not intentionally persist prediction uploads, but its server processes them in memory. Use locally for sensitive information. This is not a security-reviewed production system.

## Dataset and model

Source: [UCI Statlog German Credit, dataset 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), Hofmann (1994), [DOI 10.24432/C5NC77](https://doi.org/10.24432/C5NC77).

- 1,000 records, 20 input attributes. The source target is 1 = good risk and 2 = bad risk. Here it is recoded to 1 = historical good-risk label, 0 = historical bad-risk label.
- Source labels are not ground-truth moral judgments, and the data is not an observed approval log. "Selection" in this app means a model good-risk prediction at the displayed threshold.
- Source personal-status/sex codes A92 and A95 map to female; A91, A93 and A94 map to male. This limited historical field does not represent all gender identities.
- Age groups use **under 25 / 25+** as an illustrative audit split, not a statutory rule.
- Foreign-worker codes A201 = yes and A202 = no. This status is not presented as nationality or ethnicity.
- Age, combined personal-status/sex, foreign-worker status and all derived audit attributes are excluded from model inputs. This exclusion alone does not establish fairness.
- Numeric features are standardized and nominal categories one-hot encoded using training rows only. Unseen categories are ignored by the fitted encoder.
- Model: scikit-learn logistic regression, C=1, max_iter=2000, probability threshold 0.50 by default. This is a demonstration baseline, not an optimized credit model.
- UCI's asymmetric error matrix is reported: a bad-risk label predicted good costs 5; a good-risk label predicted bad costs 1. The displayed per-case error cost uses that matrix, not currency. The model is trained on plain log loss, not cost-sensitively, so the cost figure is not decision-calibrated.

## Metric definitions

Definitions follow the [Fairlearn common fairness metrics guide](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html).

| Measure | Calculation | What it does not establish |
| --- | --- | --- |
| Selection-rate gap | Maximum minus minimum positive-prediction rate across observed groups | Whether differences are justified, causal or unlawful |
| Min/max selection ratio | Minimum divided by maximum positive-prediction rate | An 80% compliance threshold or a fairness certificate |
| True-positive-rate gap | Maximum minus minimum selection rate among positive-label records | Error parity for negative-label records, or unbiased historical labels |

With fewer than two observed groups, all between-group summaries are undefined. If every group has zero selections, the ratio is undefined. If any group has no positive labels, the full TPR gap is undefined. Single-group TPRs without a positive denominator are undefined, not zero.

95% Wilson intervals are shown for each group rate. They are not intervals for disparity gaps, are not simultaneous intervals and are not corrected for multiple comparisons. A warning appears when any group has fewer than 30 records or fewer than 10 positive labels. These cutoffs are display heuristics, not statistical guarantees.

No pass/fail fairness badge is assigned. Fairlearn explicitly warns against treating the US employment four-fifths rule as a portable fairness or legality test. An Indian credit audit needs its own legal, domain, data-quality and human review.

### Association screen

The default screen marker is 0.30, an illustrative descriptive cutoff. It is not a significance test. Numeric features use the largest absolute Pearson correlation with one-vs-rest group indicators. Nominal features use uncorrected Cramer's V from a contingency table, so arbitrary category names do not create false numeric ordering. These scores have different interpretations and do not prove causation or model reliance. Uncorrected Cramer's V can be inflated by sparse tables. Absence of a flag is not proof of absence of proxy effects.

### Explanation

For a logistic linear model with an independent training-mean background:

`contribution_j = coefficient_j * (encoded_x_j - training_mean_j)`

`base = intercept + dot(coefficients, training_mean)`

The base plus all contributions must reconstruct the model's decision logit. The app tests this numerically before displaying the result. The implementation does not call the SHAP package; it uses the exact additive linear formula for this explicitly chosen background. These are log-odds contributions, not percentage-point contributions, causal effects or actionable advice. One-hot categories are shown separately. Correlation between inputs can make independent-background explanations misleading.

## Example run, not an external validation result

On the retrieved 1,000-row UCI dataset, seed 42, threshold 0.50 and the recorded dependencies:

- Holdout: 300 rows. Accuracy 0.737, balanced accuracy 0.656, ROC AUC 0.769.
- Sex selection gap 0.066; age-group TPR gap 0.144.
- The non-foreign-worker holdout subgroup contains only 13 records, of which 12 carry a historical good-risk label. Its rate estimate is unstable and must not be treated as a strong finding.
- No feature reached the illustrative 0.30 association marker on the training split. This is not a "proxy-free" result.

Changing the split or threshold changes these figures. Exploring several thresholds on the holdout does not turn it into fresh validation data.

### Stability across splits

The figures above come from one split. Re-running the same pipeline across 30 split seeds (`python -m examples.seed_sweep`, seeds 1 to 30, deterministic):

- Accuracy: mean 0.747 (sd 0.019), range 0.717 to 0.780.
- Sex selection gap: mean 0.056 (sd 0.034), range 0.008 to 0.105.
- Age-group TPR gap: mean 0.139 (sd 0.077), range 0.008 to 0.288.
- Lowest min/max selection ratio across attributes: mean 0.820, range 0.669 to 0.925.

The sweep is descriptive sensitivity analysis, not a confidence interval, hypothesis test or independent validation. Seeds reuse the same 1,000 historical records. It shows why one split must not stand in for evidence from a relevant lending population.

## Models, APIs and privacy

- Local model: scikit-learn logistic regression. No LLM or hosted inference API is used.
- Dataset download: public UCI HTTPS CSV. No private credentials are sent.
- Interface: Streamlit. PDF: ReportLab. Chart rendering is provided by Streamlit's chart components.
- No account connection, spending, public publishing or competition submission is built into the app.
- The app intentionally does not persist uploads or transmit them to an inference service. This is not a guarantee about every deployment host's logs, memory or security.

## Limits and next steps

The demo does not cover intersectional fairness, calibration by group, causal discrimination, longitudinal stability or validated mitigation. Confidence intervals for disparity gaps are not implemented; the rate intervals are per-group Wilson intervals. The historical German dataset cannot establish performance or fairness for a present-day Indian lender. Before any real lending use: use lawfully collected relevant data, validate labels, evaluate negative-class error disparities and business costs, use separate validation, involve affected users/domain/legal reviewers, and conduct security/privacy review.

## Project layout

```text
app.py                    Streamlit review flow
arthaudit/data.py         source parsing, protected attributes and feature selection
arthaudit/audit.py        train-only preprocessing, baseline, holdout audit
arthaudit/fairness.py     group metrics, denominators and Wilson intervals
arthaudit/proxies.py      exploratory association screen
arthaudit/explain.py      checked additive linear explanations
arthaudit/report.py       one-page evidence summary
examples/run_audit.py     reproducible CLI example
examples/seed_sweep.py    descriptive split-stability sweep
tests/                    synthetic fixtures and regression tests
```

Built for Arthaton as a scoped prototype. No bank validation, regulator endorsement or compliance certification is claimed.

## Demo video

https://youtu.be/AsIRX8kecBk

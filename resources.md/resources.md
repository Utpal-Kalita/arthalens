# Resources

Links used while building ArthaLens, with what each one grounded.

## Event

- Arthaton 2026 project submission form: https://forms.gle/mdXFpDtq5R4inTbq8
  Deadline 22 November 2026, 11:59 PM IST; public GitHub repo; YouTube demo under 5 minutes, public or unlisted; README must cover what the app does, how to run it and which AI models/APIs are used; only commits before the deadline are judged.

## Dataset

- UCI Machine Learning Repository, Statlog (German Credit Data), dataset 144:
  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
  1,000 rows, 20 attributes, target 1 = good risk / 2 = bad risk, donated 1994.
- Direct CSV used by the loader: https://archive.ics.uci.edu/static/public/144/data.csv
- Citation DOI: https://doi.org/10.24432/C5NC77

## Fairness methodology

- Fairlearn, Common fairness metrics:
  https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html
  Definitions for demographic parity difference/ratio and true-positive-rate difference, and the explicit warning against treating the US employment four-fifths rule as a portable fairness or legality test. ArthaLens follows that warning: it shows ratios with denominators and assigns no pass/fail badge.
- Watkins, McKenna and Chen, "The four-fifths rule is not disparate impact": https://arxiv.org/abs/2202.09519
- Hardt, Price and Srebro, "Equality of Opportunity in Supervised Learning": https://papers.neurips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html

## Regulatory context (framing only, no compliance claim)

- Reserve Bank of India, FREE-AI committee report (Framework for Responsible and Ethical Enablement of AI), 2025: https://www.rbi.org.in/Scripts/PublicationReportDetails.aspx?UrlPage=&ID=1247
  Motivates audit tooling for lenders using AI. ArthaLens does not implement, certify or claim alignment with any RBI framework.

## Libraries

- scikit-learn (baseline model, preprocessing): https://scikit-learn.org
- Streamlit (interface): https://streamlit.io
- ReportLab (PDF export): https://www.reportlab.com
- Altair (chart): https://altair-viz.github.io

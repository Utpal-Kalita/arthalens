"""ArthaLens: a local-first credit-model review workbench.
Run with: streamlit run app.py
"""
from __future__ import annotations
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import altair as alt
import streamlit as st
from arthaudit import train_and_audit, build_scorecard
from arthaudit.audit import threshold_study
from arthaudit.data import derive_protected, load_german_credit
from arthaudit.explain import explain_applicant
from arthaudit.fairness import group_fairness

st.set_page_config(page_title="ArthaLens | Credit Audit", page_icon="◈", layout="wide")
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Libre+Caslon+Display&display=swap');
:root {--ink:#143c43;--mint:#dcebea;--paper:#f6f5ef;}
.stApp {background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif;}
h1,h2,h3 {color:var(--ink)!important;letter-spacing:-.035em;}
h1 {font-family:'Libre Caslon Display',Georgia,serif!important;font-size:4rem!important;line-height:1.02!important;font-weight:400!important;}
[data-testid="stMetric"] {background:#fff;border:1px solid #d5e0db;border-radius:7px;padding:17px 20px;}
[data-testid="stMetricLabel"] {text-transform:uppercase;letter-spacing:.07em;font-size:11px;}
[data-testid="stMetricValue"] {color:var(--ink);}
[data-testid="stTabs"] button {font-size:15px!important;}
.kicker {font-size:12px;letter-spacing:.18em;font-weight:700;color:#5d7878;margin-bottom:26px;}
.hero-note {font-size:18px;color:#5b7070;line-height:1.6;max-width:690px;}
.audit-rule {border-top:1px solid #b8cac5;margin:28px 0 22px;}
.stButton button,.stDownloadButton button {background:#143c43;color:white;border:0;border-radius:4px;}
footer {display:none;}
[data-testid="stToolbar"] {display:none;}
@media(max-width:600px) {h1 {font-size:2.8rem!important;} [data-testid="stTabs"] [role="tablist"] {flex-wrap:wrap;gap:3px;} [data-testid="stTabs"] [role="tab"] {min-width:45%;}}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def dataset():
    return derive_protected(load_german_credit())

@st.cache_resource(show_spinner=False)
def baseline(threshold):
    return train_and_audit(dataset(), threshold=threshold)

st.markdown('<div class="kicker">ARTHALENS &nbsp; / &nbsp; MODEL REVIEW WORKBENCH</div>', unsafe_allow_html=True)
left,right=st.columns([3,1])
with left:
    st.markdown("<h1>A good score can<br/>hide a wide gap.</h1>", unsafe_allow_html=True)
    st.markdown('<p class="hero-note">Inspect who a credit model selects, where its errors fall, and why one prediction changed. Keep the evidence, not a fairness stamp.</p>', unsafe_allow_html=True)
with right:
    st.caption("DEMONSTRATION DATA")
    st.write("UCI German Credit")
    st.caption("1,000 historical records · source donated 1994\n\nNo banking connection. No paid API.")
st.markdown('<div class="audit-rule"></div>', unsafe_allow_html=True)

mode = st.radio("Workspace", ["Explore the benchmark", "Audit prediction CSV"], horizontal=True, label_visibility="collapsed")


def render_groups(fr):
    rows=[]
    for g in fr.groups:
        rows.append({"Group":g.group,"Records":g.count,"Historical good-risk labels":g.positive_count,
                     "Selection rate":g.selection_rate,"95% lower":g.selection_low,"95% upper":g.selection_high,
                     "True-positive rate":g.tpr,"TPR 95% lower":g.tpr_low,"TPR 95% upper":g.tpr_high})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if any(g.count < 30 or g.positive_count < 10 for g in fr.groups):
        st.warning("Small-group evidence: at least one group has fewer than 30 records or 10 positive labels. Treat its estimate as unstable, not a reliable verdict.")


def metrics(fr):
    a,b,c=st.columns(3)
    a.metric("Selection-rate gap", "n/a" if fr.demographic_parity_difference is None else f"{fr.demographic_parity_difference:.1%}")
    b.metric("Min / max selection ratio", "n/a" if fr.disparate_impact_ratio is None else f"{fr.disparate_impact_ratio:.3f}")
    c.metric("True-positive-rate gap", "n/a" if fr.equal_opportunity_difference is None else f"{fr.equal_opportunity_difference:.1%}")

if mode == "Audit prediction CSV":
    st.subheader("Bring decisions. Not personal identifiers.")
    st.write("Audit an existing binary model without retraining it. Include y_true, y_pred and group columns; both labels must be 0 or 1. Here 1 is the favourable outcome.")
    st.caption("Avoid names, account numbers and other identifiers. This app does not intentionally persist uploads, but the server processes uploaded data in memory. Local use is recommended for sensitive data.")
    example=pd.DataFrame({"y_true":[1,1,0,0,1,1,0,0],"y_pred":[1,0,1,0,1,1,0,0],"group":["A","A","A","A","B","B","B","B"]})
    st.download_button("Download labelled toy CSV",example.to_csv(index=False),"toy_predictions.csv","text/csv")
    upload=st.file_uploader("Prediction CSV",type=["csv"])
    if upload:
        try:
            if upload.size>2_000_000:
                raise ValueError("Use a CSV smaller than 2 MB for this demo.")
            df=pd.read_csv(upload)
            if set(df.columns)!={"y_true","y_pred","group"}:
                raise ValueError("CSV must have exactly y_true, y_pred, group columns.")
            fr=group_fairness(df.y_true,df.y_pred,df.group,"uploaded_group")
            metrics(fr);render_groups(fr)
            st.caption("Gaps compare observed groups only. Missing groups cannot be assessed. An undefined ratio means no positive predictions or no between-group comparison; an undefined TPR means a missing positive-label denominator.")
        except Exception as exc:
            st.error(str(exc))
else:
    try:
        threshold=st.slider("Good-risk probability threshold",0.05,0.95,0.5,0.05,
                            help="A higher threshold selects fewer applicants. This is an exploratory holdout view, not a tuned or deployable lending policy.")
        with st.spinner("Loading the public benchmark and fitting the baseline..."):
            result=baseline(threshold)
    except Exception as exc:
        st.error(f"The benchmark could not be loaded: {exc}")
        st.stop()
    a,b,c,d=st.columns(4)
    a.metric("Held-out records", result.n_test)
    b.metric("Accuracy", f"{result.accuracy:.1%}")
    c.metric("Balanced accuracy", f"{result.balanced_accuracy:.1%}")
    d.metric("UCI error cost / case", f"{result.cost_per_case:.3f}")
    st.caption(f"Logistic regression · 70/30 stratified split · seed {result.seed} · train-only one-hot encoding and scaling · sex/status, age and foreign-worker fields excluded from model inputs. Selection = predicted good risk, not an observed approval.")
    tab1,tab2,tab3,tab4=st.tabs(["01 / Group gaps","02 / Decision evidence","03 / Associations","04 / Review record"])
    with tab1:
        st.subheader("Who gets selected, and who gets missed?")
        attribute=st.selectbox("Compare groups",["sex","age_group","is_foreign_worker"],format_func=lambda x:{"sex":"Sex (source-coded)","age_group":"Age: under 25 / 25+","is_foreign_worker":"Foreign-worker status"}[x])
        fr=next(f for f in result.fairness if f.attribute==attribute)
        metrics(fr)
        chart=pd.DataFrame({"Group":[g.group for g in fr.groups],"Selection rate":[g.selection_rate for g in fr.groups],"True-positive rate":[g.tpr for g in fr.groups]}).set_index("Group")
        rates = chart.reset_index().melt("Group", var_name="Measure", value_name="Rate")
        comparison = alt.Chart(rates).mark_bar().encode(
            x=alt.X("Group:N", title="Audit group"),
            xOffset="Measure:N",
            y=alt.Y("Rate:Q", scale=alt.Scale(domain=[0, 1]), title="Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color("Measure:N", scale=alt.Scale(domain=["Selection rate", "True-positive rate"],
                                                       range=["#1c6e75", "#d8ad68"])),
            tooltip=["Group:N", "Measure:N", alt.Tooltip("Rate:Q", format=".1%")]
        ).properties(height=265)
        st.altair_chart(comparison, width="stretch")
        render_groups(fr)
        with st.expander("How to read these measures"):
            st.write("Selection gap = largest minus smallest selection rate. Min/max ratio = smallest divided by largest selection rate. TPR gap compares selection among records with a historical good-risk label. Gaps need context; neither a small gap nor a high ratio proves fairness.")
            st.write("95% Wilson intervals describe individual group rates, not gaps. They do not correct for multiple comparisons. Historical labels may already encode bias. The age cutoff is illustrative, not a lending rule.")
    with tab2:
        st.subheader("Open one prediction.")
        st.caption("Exact linear SHAP-style attribution under an independent training-mean background. Contributions are log-odds toward the historical good-risk class, not probability points or causes.")
        idx=st.number_input("Held-out row (not a customer identifier)",0,result.n_test-1,0)
        vals,base,logit=explain_applicant(result,int(idx))
        aa,bb,cc=st.columns(3)
        aa.metric("Good-risk probability",f"{result.probabilities[int(idx)]:.1%}")
        bb.metric("Model prediction","Good risk" if result.y_pred[int(idx)] else "Bad risk")
        cc.metric("Historical label","Good risk" if result.y_test[int(idx)] else "Bad risk")
        top=vals.reindex(vals.abs().sort_values(ascending=False).head(10).index)
        contrib=top.rename_axis("Encoded feature").reset_index(name="Log-odds contribution")
        chart=alt.Chart(contrib).mark_bar(color="#1c6e75").encode(
            x=alt.X("Log-odds contribution:Q",title="Log-odds toward good risk"),
            y=alt.Y("Encoded feature:N",sort=None,axis=alt.Axis(labelLimit=230,labelFontSize=11)),
            tooltip=["Encoded feature","Log-odds contribution"]).properties(height=355)
        st.altair_chart(chart,width="stretch")
        st.dataframe(contrib,hide_index=True,width="stretch")
        st.caption(f"Reconstruction check: background {base:.4f} + all feature contributions {vals.sum():.4f} = decision logit {logit:.4f}. Correlations and the chosen background affect these attributions.")
        with st.expander("View encoded source values"):
            st.dataframe(result.X_test.iloc[[int(idx)]],hide_index=True)
    with tab3:
        st.subheader("Excluding a group label is not enough.")
        st.write("Other inputs can still track a group. This screen measures training-data associations; it does not prove the model uses a feature as a proxy, and it does not find every possible proxy.")
        if result.proxies:
            st.dataframe(pd.DataFrame([vars(h) for h in result.proxies]),hide_index=True,width="stretch")
        else:
            st.info("No feature reached the illustrative association marker of 0.30 on the training split. This is not evidence that the model is proxy-free.")
        st.caption("Numeric features: absolute Pearson against one-vs-rest group indicators. Nominal features: uncorrected Cramer's V. These are different descriptive measures, not causal effects or significance tests.")
        st.subheader("A stricter threshold has a cost.")
        st.dataframe(threshold_study(result,attribute),hide_index=True,width="stretch")
        st.caption("Exploratory threshold sweep on the same holdout. Do not select a deployment policy from this table; use separate validation, error-cost analysis and domain review.")
    with tab4:
        st.subheader("Hand over the evidence, with its limits.")
        with TemporaryDirectory() as tmp:
            path=build_scorecard(result,str(Path(tmp)/"review.pdf"))
            pdf=Path(path).read_bytes()
        st.download_button("Download one-page audit summary",pdf,"arthalens-audit-summary.pdf","application/pdf")
        st.write("This prototype is an evidence organizer, not a compliance certification or a loan-decision service. It does not claim RBI endorsement, bank validation or production readiness.")
        st.markdown("""**Limits to carry into review**
- The 1994 German benchmark is not an Indian lender portfolio. No Indian borrowers were tested.
- Binary sex labels come from a combined status/sex field and do not represent all identities.
- Very small groups make disparity estimates unstable; historical labels can reproduce bias.
- One seeded split is not external validation. No causal discrimination claim is made.
- No mitigation has been validated; a small measured gap does not make a model safe.
""")
        st.markdown("**Sources and definitions**\n\n[UCI dataset 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) · [Fairlearn metric definitions and cautions](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html)")

st.markdown('<div class="audit-rule"></div>',unsafe_allow_html=True)
st.caption("ARTHALENS / ARTHATON PROTOTYPE     •     Evidence for human review. No fairness stamp.")

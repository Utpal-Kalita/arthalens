"""Compact human-review PDF. No legal verdicts or unsupported certifications."""
from __future__ import annotations
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from .audit import AuditResult


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def build_scorecard(result: AuditResult, path: str, stability: dict | None = None) -> str:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("SmallNote", parent=styles["Normal"], fontSize=8, leading=11,
                              textColor=colors.HexColor("#475569")))
    title = ParagraphStyle("AuditTitle", fontName="Helvetica-Bold", fontSize=25, leading=29,
                           textColor=colors.HexColor("#102e38"), spaceAfter=9)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=38, rightMargin=38, topMargin=32, bottomMargin=30)
    story = [Paragraph("ARTHALENS / CREDIT AUDIT", styles["SmallNote"]), Spacer(1, 8),
             Paragraph("Know the gaps.<br/>Keep the evidence.", title),
             Paragraph("Historical benchmark demonstration. Human review required.", styles["Normal"]), Spacer(1, 12)]
    story.append(Paragraph(
        f"{escape(result.model_name)} | UCI German Credit | {result.n_rows} records | "
        f"{result.n_test} held out | seed {result.seed} | threshold {result.threshold:.2f}", styles["SmallNote"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Accuracy {result.accuracy:.3f} &nbsp; Balanced accuracy {result.balanced_accuracy:.3f} &nbsp; "
        f"ROC AUC {result.roc_auc:.3f}<br/>UCI cost / held-out case: {result.cost_per_case:.3f} "
        "(bad predicted good = 5; good predicted bad = 1; the model is not trained to this cost).",
        styles["Normal"]))
    story.append(Spacer(1, 12))
    rows = [["Group comparison", "Selection gap", "Min/max ratio", "TPR gap"]]
    for fr in result.fairness:
        rows.append([fr.attribute, fmt(fr.demographic_parity_difference),
                     fmt(fr.disparate_impact_ratio), fmt(fr.equal_opportunity_difference)])
    def table(rows, widths):
        t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123b46")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf4f4"), colors.white]),
            ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#cad5d8"))]))
        return t
    story.append(table(rows, [175, 115, 115, 114]))
    story.append(Spacer(1, 10))
    detail = [["Attribute / group", "N / positive N", "Selected", "95% Wilson interval", "TPR"]]
    for fr in result.fairness:
        for g in fr.groups:
            detail.append([f"{fr.attribute} / {g.group}", f"{g.count} / {g.positive_count}",
                           fmt(g.selection_rate), f"{g.selection_low:.3f} to {g.selection_high:.3f}", fmt(g.tpr)])
    story.append(table(detail, [166, 88, 65, 125, 75]))
    if stability:
        s = stability["sweep"]
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"<b>Split stability, seeds {stability['sweep_seeds'][0]} to {stability['sweep_seeds'][1]}</b> "
            f"(mean +/- sd; range): accuracy {s['accuracy']['mean']:.3f} +/- {s['accuracy']['sd']:.3f}; "
            f"sex selection gap {s['sex_selection_gap']['mean']:.3f} +/- {s['sex_selection_gap']['sd']:.3f} "
            f"({s['sex_selection_gap']['min']:.3f} to {s['sex_selection_gap']['max']:.3f}); "
            f"age TPR gap {s['age_tpr_gap']['mean']:.3f} +/- {s['age_tpr_gap']['sd']:.3f} "
            f"({s['age_tpr_gap']['min']:.3f} to {s['age_tpr_gap']['max']:.3f}); "
            f"lowest min/max selection ratio {s['min_disparate_impact']['mean']:.3f} "
            f"({s['min_disparate_impact']['min']:.3f} to {s['min_disparate_impact']['max']:.3f}).<br/>"
            "Descriptive split sensitivity only, not a confidence interval, significance test or independent validation.",
            styles["SmallNote"]))
    story.append(Spacer(1, 10))
    small = [f"{escape(h.feature)} / {escape(h.protected_attribute)}: {h.correlation:.2f} ({escape(h.method)})"
             for h in result.proxies[:3]]
    story.append(Paragraph("<b>Association screen, training split only</b><br/>" +
                           ("<br/>".join(small) if small else "No feature reached the illustrative 0.30 screen marker."),
                           styles["SmallNote"]))
    story.append(Spacer(1, 9))
    story.append(Paragraph(
        "<b>Read before acting.</b> Selection means a model good-risk prediction, not an observed loan approval. "
        "The age cutoff of 25 is illustrative. Small groups have unstable estimates. Per-group Wilson intervals "
        "are not confidence intervals for gaps, simultaneous intervals or corrected for multiple comparisons. Historical labels may encode bias. "
        "Sex is derived from the source's limited combined status/sex codes. Association is not causation. "
        "This 1994 German benchmark does not validate an Indian lender or certify fairness, legality or RBI compliance.",
        styles["SmallNote"]))
    story.append(Spacer(1, 7))
    story.append(Paragraph("Sources: UCI dataset 144, doi:10.24432/C5NC77; Fairlearn common fairness metrics. "
                           "Reproduce with seed 42 and the recorded dependency versions.", styles["SmallNote"]))
    doc.build(story)
    return path

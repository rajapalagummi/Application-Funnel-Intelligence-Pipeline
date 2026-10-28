import json
import pandas as pd
from pathlib import Path
from fpdf import FPDF

OUTPUTS = Path("data/outputs")
DOCS = Path("docs")
README = Path("README.md")
START = "<!-- CREDIT_RISK_START -->"
END = "<!-- CREDIT_RISK_END -->"


def load():
    j = lambda n: json.load(open(OUTPUTS / n))
    agent = j("agent_runs.json") if (OUTPUTS / "agent_runs.json").exists() else None
    return j("credit_data_summary.json"), j("credit_model_metrics.json"), j("validation_results.json"), pd.read_csv(OUTPUTS / "iv_table.csv"), agent


def pct(v):
    return f"{v * 100:.2f}%"


def model_card(d, m, v, iv):
    sc, ch = v["discrimination"]["scorecard"], v["discrimination"]["challenger"]
    top = iv[iv["in_model"]].head(10)
    lines = [
        "# Model Card - Home Credit PD Scorecard",
        "",
        "## Purpose",
        "Estimate the probability of default (PD) of a loan applicant at origination and express it as a points-based credit score for approve/decline and pricing decisions.",
        "",
        "## Data",
        f"- Source: Kaggle Home Credit Default Risk ({', '.join(d['sources'])}), data_source = {d['data_source']}",
        f"- Rows modeled: {d['rows_total']:,}; default rate {pct(d['default_rate_total'])}",
        f"- Split: {d['split']} - train {d['rows']['train']:,} / validation {d['rows']['val']:,} / test {d['rows']['test']:,}",
        f"- Default rate by split: train {pct(d['default_rate']['train'])}, validation {pct(d['default_rate']['val'])}, test {pct(d['default_rate']['test'])}",
        f"- Leakage checks: SK_ID_CURR overlap across splits = {sum(d['id_overlap'].values())}; features with univariate AUC > {d['leakage_threshold_auc']}: {len(d['leakage_flags'])}",
        "",
        "## Design",
        f"- Champion: logistic regression on Weight-of-Evidence features; score = {m['scorecard_config']['base_score']} at {m['scorecard_config']['odds']}:1 good:bad odds, {m['scorecard_config']['pdo']} points to double the odds",
        f"- Binning: up to 10 quantile bins per numeric feature merged until WOE is monotonic; categories under 1% grouped; missing values binned separately",
        f"- Feature selection: {m['features']['n_candidates']} candidates -> {m['features']['n_iv_selected']} with {m['features']['iv_range'][0]} < IV < {m['features']['iv_range'][1]} -> {m['features']['n_after_correlation_filter']} after |corr| < {m['features']['max_correlation']} filter -> {m['features']['n_final']} with negative coefficients",
        f"- Challenger: {m['challenger_config']['model']} on {m['challenger_config']['n_features']} raw features, early stopping at {m['challenger_config']['best_iteration']} trees",
        "",
        "## Features in model (top 10 by IV)",
        "| Feature | IV | Bins |",
        "|---|---|---|",
        *[f"| {r.feature} | {r.iv:.4f} | {r.n_bins} |" for r in top.itertuples()],
        "",
        "## Performance (test set, 95% bootstrap CI)",
        "| Model | AUC | Gini | KS |",
        "|---|---|---|---|",
        f"| Scorecard | {sc['auc']:.4f} [{sc['auc_ci'][0]:.4f}, {sc['auc_ci'][1]:.4f}] | {sc['gini']:.4f} [{sc['gini_ci'][0]:.4f}, {sc['gini_ci'][1]:.4f}] | {sc['ks']:.4f} [{sc['ks_ci'][0]:.4f}, {sc['ks_ci'][1]:.4f}] |",
        f"| LightGBM challenger | {ch['auc']:.4f} [{ch['auc_ci'][0]:.4f}, {ch['auc_ci'][1]:.4f}] | {ch['gini']:.4f} | {ch['ks']:.4f} |",
        "",
        f"Calibration: Hosmer-Lemeshow p = {v['calibration']['hosmer_lemeshow_p']:.4f}, Brier = {v['calibration']['brier']:.4f} (reference {v['calibration']['brier_reference']:.4f}). Stability: score PSI train vs test = {v['stability']['psi_train_test']:.4f}.",
        "",
        "## Limitations",
        "- Single static snapshot; no out-of-time sample, so temporal stability is approximated by a random holdout.",
        "- Application-time features only; no behavioural or repayment history beyond bureau and previous-application aggregates.",
        "- External scores (EXT_SOURCE_*) dominate the model and are vendor black boxes.",
        "- No fairness review across protected attributes has been performed.",
        "",
        "## Intended use",
        "Origination PD ranking and score-band cut-offs in a portfolio-analytics context. Not intended for automated decisions without human review or for portfolios outside the development population.",
        "",
        "## Monitoring plan",
        "- Monthly score PSI vs development (alert > 0.10, action > 0.25) and CSI per model feature (alert > 0.10)",
        "- Quarterly AUC/KS on matured accounts against the development benchmark",
        "- Decile calibration review; recalibrate the intercept if Hosmer-Lemeshow p < 0.01 for two consecutive quarters",
        "- Automated monthly monitoring summary drafted by the reporting agent with a numeric guardrail (credit_risk/report_agent.py)",
    ]
    return "\n".join(lines) + "\n"


def validation_report(v):
    s = v["summary"]
    lines = [
        "# Independent Model Validation Report - Home Credit PD Scorecard",
        "",
        "## Scope",
        f"Validation of the {v['model']} on a held-out test set of {v['test_rows']:,} applications (default rate {pct(v['test_default_rate'])}), covering discrimination, calibration, stability, benchmarking, sensitivity and model assumptions in the spirit of SR 11-7.",
        "",
        "## Results",
        "| Category | Test | Value | Threshold | Status |",
        "|---|---|---|---|---|",
        *[f"| {t['category']} | {t['test']} | {t['value']} | {t['threshold']} | {t['status']} |" for t in v["tests"]],
        "",
        f"## Overall rating: {s['overall_rating']}",
        f"{s['n_tests']} tests: {s['pass']} PASS, {s['conditional']} CONDITIONAL, {s['fail']} FAIL.",
        "",
        "## Calibration by decile",
        "| Decile | N | Predicted rate | Actual rate |",
        "|---|---|---|---|",
        *[f"| {c['decile']} | {c['n']:,} | {c['predicted_rate']:.4f} | {c['actual_rate']:.4f} |" for c in v["calibration"]["deciles"]],
        "",
        "## Sensitivity (mean PD change, %)",
        "| Feature | -10% | +10% |",
        "|---|---|---|",
        *[f"| {r['feature']} | {r['mean_pd_change_pct_minus10']} | {r['mean_pd_change_pct_plus10']} |" for r in v["sensitivity"]],
        "",
        "## Findings and recommendations",
    ]
    findings = [t for t in v["tests"] if t["status"] != "PASS"]
    if not findings:
        lines.append("- No findings; all tests passed.")
    for t in findings:
        lines.append(f"- {t['status']}: {t['test']} = {t['value']} (threshold {t['threshold']}).")
    lines += [
        "- Re-run this validation on an out-of-time sample once later vintages are available.",
        "- Review EXT_SOURCE dependence and run a fairness assessment before production use.",
    ]
    return "\n".join(lines) + "\n"


def to_pdf(md, path):
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    for line in md.splitlines():
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 8, line[2:], new_x="LMARGIN", new_y="NEXT")
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, line[3:], new_x="LMARGIN", new_y="NEXT")
        elif line.startswith("|"):
            if set(line) <= set("|-"):
                continue
            pdf.set_font("Courier", "", 7)
            pdf.multi_cell(0, 4, "  ".join(c.strip() for c in line.strip("|").split("|")), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def readme_section(d, m, v, agent):
    sc, ch = v["discrimination"]["scorecard"], v["discrimination"]["challenger"]
    s = v["summary"]
    rows = [
        ("Data", "Rows modeled / default rate", f"{d['rows_total']:,} / {pct(d['default_rate_total'])}"),
        ("Features", "Candidates -> IV-selected -> in model", f"{m['features']['n_candidates']} -> {m['features']['n_iv_selected']} -> {m['features']['n_final']}"),
        ("Scorecard", "Test AUC (95% CI)", f"{sc['auc']:.4f} [{sc['auc_ci'][0]:.4f}, {sc['auc_ci'][1]:.4f}]"),
        ("Scorecard", "Test Gini / KS", f"{sc['gini']:.4f} / {sc['ks']:.4f}"),
        ("Challenger", "LightGBM test AUC / Gini / KS", f"{ch['auc']:.4f} / {ch['gini']:.4f} / {ch['ks']:.4f}"),
        ("Benchmark", "Challenger - scorecard AUC (p-value)", f"{v['benchmark']['auc_difference']:+.4f} (p={v['benchmark']['p_value']:.4f})"),
        ("Calibration", "Hosmer-Lemeshow p / Brier", f"{v['calibration']['hosmer_lemeshow_p']:.4f} / {v['calibration']['brier']:.4f}"),
        ("Stability", "Score PSI train vs test / features with CSI > 0.1", f"{v['stability']['psi_train_test']:.4f} / {v['stability']['n_csi_above_0_1']}"),
        ("Validation", "Tests PASS / CONDITIONAL / FAIL", f"{s['pass']} / {s['conditional']} / {s['fail']} ({s['overall_rating']})"),
    ]
    if agent:
        a = agent["summary"]
        rows.append(("GenAI agent", "Runs / numbers checked / guardrail pass rate", f"{a['total_runs']} / {a['numbers_checked_total']} / {pct(a['guardrail_pass_rate'])}"))
    lines = [
        START,
        "## Credit Risk Track - PD Scorecard, Validation, Monitoring",
        "",
        "Built on the real Home Credit `TARGET` default label (separate from the funnel drop-off model). Every number below is generated from committed files in `data/outputs/`.",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[application_train + bureau + previous_application] --> B[data_prep: 60/20/20 split, leakage checks]",
        "    B --> C[scorecard: WOE/IV binning, logistic champion, 600/20/50 points]",
        "    B --> D[LightGBM challenger]",
        "    C --> E[validation: discrimination, calibration, PSI/CSI, benchmark, sensitivity, VIF]",
        "    D --> E",
        "    E --> F[docs: model card + validation report PDF]",
        "    E --> G[Streamlit Credit Risk page + credit_scores.csv]",
        "    E --> H[LangGraph report agent with numeric guardrail]",
        "```",
        "",
        "| Area | Metric | Value |",
        "|---|---|---|",
        *[f"| {a} | {b} | {c} |" for a, b, c in rows],
        "",
        "Docs: [model card](docs/model_card.pdf) | [validation report](docs/validation_report.pdf) | Plots: [ROC](images/roc.png), [KS](images/ks.png), [calibration](images/calibration.png)",
        "",
        "```bash",
        "python3 -m credit_risk.run",
        "python3 -m credit_risk.report_agent 5",
        "```",
        END,
    ]
    return "\n".join(lines)


def update_readme(section):
    text = README.read_text()
    if START in text:
        text = text.split(START)[0] + section + text.split(END)[1]
    else:
        text = text.rstrip() + "\n\n---\n\n" + section + "\n"
    README.write_text(text)


def run():
    DOCS.mkdir(exist_ok=True)
    d, m, v, iv, agent = load()
    card, report = model_card(d, m, v, iv), validation_report(v)
    (DOCS / "model_card.md").write_text(card)
    (DOCS / "validation_report.md").write_text(report)
    to_pdf(card, DOCS / "model_card.pdf")
    to_pdf(report, DOCS / "validation_report.pdf")
    update_readme(readme_section(d, m, v, agent))
    print("[build_docs] docs/model_card.md|pdf, docs/validation_report.md|pdf, README Credit Risk section updated")


if __name__ == "__main__":
    run()

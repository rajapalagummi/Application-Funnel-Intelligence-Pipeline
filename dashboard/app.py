import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="FunnelIQ", layout="wide")
st.title("FunnelIQ — Application Funnel Intelligence Platform")
st.caption("Multi-stage funnel diagnostics · Causal intervention analysis · Drop-off prediction · Champion/Challenger · Live observability")

RESULTS = Path("data/outputs/results.json")
DRIFT = Path("data/outputs/drift_monitoring.csv")
FUNNEL = Path("data/outputs/funnel_metrics.csv")

if not RESULTS.exists():
    st.warning("Run `python main.py` first to generate results.")
    st.stop()

with open(RESULTS) as f:
    results = json.load(f)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Funnel Diagnostics",
    "Causal Analysis",
    "Drop-Off Model",
    "Champion/Challenger",
    "Observability"
])

with tab1:
    st.subheader("Multi-Stage Funnel Lineage & Bottleneck Diagnostics")
    summary = results.get("funnel_summary", {})
    col1, col2 = st.columns(2)
    col1.metric("Total Applications", f"{summary.get('total', 0):,}")
    col2.metric("End-to-End Conversion", f"{summary.get('conversion_rate', 0):.1f}%")

    if FUNNEL.exists():
        df = pd.read_csv(FUNNEL)
        fig = px.bar(df, x="drop_stage", y="n_applicants",
                     color="pct_total", title="Applications by Drop-Off Stage",
                     labels={"drop_stage": "Stage", "n_applicants": "Applicants"})
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(df, x="drop_stage", y="avg_days_to_decision",
                          size="n_applicants", title="Median Time-to-Decision by Stage")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Causal Intervention Analysis — DiD Framework")
    causal = results.get("causal", {})

    for outcome in ["converted", "submitted"]:
        r = causal.get(outcome, {})
        did = r.get("did", {})
        placebo = r.get("placebo", {})
        parallel = r.get("parallel_trends", {})

        if not did:
            continue

        st.markdown(f"**Outcome: {outcome.upper()}**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("DiD Estimate", did.get("did_estimate", "N/A"))
        col2.metric("p-value", did.get("p_value", "N/A"))
        col3.metric("Significant", "✅ Yes" if did.get("significant") else "❌ No")
        col4.metric("Placebo p-value", placebo.get("p_value", "N/A"))

        rates = {
            "Group": ["Treatment Pre", "Treatment Post", "Control Pre", "Control Post"],
            "Rate": [did.get("treat_pre", 0), did.get("treat_post", 0),
                     did.get("ctrl_pre", 0), did.get("ctrl_post", 0)],
            "Period": ["Pre", "Post", "Pre", "Post"],
            "Cohort": ["Treatment", "Treatment", "Control", "Control"],
        }
        fig = px.line(pd.DataFrame(rates), x="Period", y="Rate", color="Cohort",
                      title=f"DiD Parallel Trends — {outcome}", markers=True)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Predictive Drop-Off Model — XGBoost / LightGBM")
    model_r = results.get("model", {})
    cv = model_r.get("cv_results", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("XGBoost AUC", f"{cv.get('xgb_mean_auc', 0)} ± {cv.get('xgb_std_auc', 0)}")
    col2.metric("LightGBM AUC", f"{cv.get('lgb_mean_auc', 0)} ± {cv.get('lgb_std_auc', 0)}")
    col3.metric("Champion", model_r.get("champion", "N/A"))

    shap = model_r.get("shap_importance", {})
    if shap:
        shap_df = pd.DataFrame(list(shap.items()), columns=["Feature", "SHAP Value"])
        fig = px.bar(shap_df.head(10), x="SHAP Value", y="Feature", orientation="h",
                     title="Top SHAP Drivers of Application Drop-Off")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Champion vs Challenger Experimentation")
    exp = results.get("experimentation", {})
    cc = exp.get("champion_challenger", {})
    power = exp.get("power_analysis", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Champion AUC", cc.get("champion_mean_auc", "N/A"))
    col2.metric("Challenger AUC", cc.get("challenger_mean_auc", "N/A"))
    col3.metric("AUC Lift", cc.get("auc_lift", "N/A"))
    col4.metric("Statistical Power", f"{power.get('statistical_power', 0):.1%}")

    curve = exp.get("power_curve", [])
    if curve:
        curve_df = pd.DataFrame(curve)
        fig = px.line(curve_df, x="effect_size", y="statistical_power",
                      title="Power Curve — Effect Size vs Statistical Power",
                      markers=True)
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                      annotation_text="80% threshold")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Live Observability — PSI Drift & SPC Monitoring")
    obs = results.get("observability", {})
    spc = obs.get("spc", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Max PSI", obs.get("max_psi", "N/A"),
                delta="Alert" if obs.get("max_psi", 0) > 0.25 else "Stable")
    col2.metric("PSI Alerts Fired", obs.get("psi_alerts", 0))
    col3.metric("KS Alerts Fired", obs.get("ks_alerts", 0))

    if DRIFT.exists():
        drift_df = pd.read_csv(DRIFT)
        fig = px.line(drift_df, x="window", y="conversion_rate",
                      title="Conversion Rate Over Time (SPC Chart)")
        fig.add_hline(y=spc.get("ucl", 1), line_dash="dash", line_color="red",
                      annotation_text="UCL")
        fig.add_hline(y=spc.get("lcl", 0), line_dash="dash", line_color="red",
                      annotation_text="LCL")
        fig.add_hline(y=spc.get("mean", 0.5), line_dash="dot", line_color="green",
                      annotation_text="Mean")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(drift_df, x="window", y="psi",
                      color=drift_df["psi_alert"].map({True: "Alert", False: "OK"}),
                      title="PSI Drift by Time Window",
                      color_discrete_map={"Alert": "#e74c3c", "OK": "#2ecc71"})
        fig2.add_hline(y=0.25, line_dash="dash", annotation_text="PSI Alert Threshold")
        st.plotly_chart(fig2, use_container_width=True)

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

OUT = Path("data/outputs")
st.title("Credit Risk - PD Scorecard")

if not (OUT / "validation_results.json").exists():
    st.warning("Run `python3 -m credit_risk.run` first.")
    st.stop()

v = json.load(open(OUT / "validation_results.json"))
m = json.load(open(OUT / "credit_model_metrics.json"))
scores = pd.read_csv(OUT / "credit_scores.csv")
iv = pd.read_csv(OUT / "iv_table.csv")
sc, ch = v["discrimination"]["scorecard"], v["discrimination"]["challenger"]

c = st.columns(5)
c[0].metric("Scorecard AUC", f"{sc['auc']:.4f}", f"CI {sc['auc_ci'][0]:.3f}-{sc['auc_ci'][1]:.3f}")
c[1].metric("Gini", f"{sc['gini']:.4f}")
c[2].metric("KS", f"{sc['ks']:.4f}")
c[3].metric("Challenger AUC", f"{ch['auc']:.4f}", f"{v['benchmark']['auc_difference']:+.4f} vs scorecard")
c[4].metric("Validation", v["summary"]["overall_rating"], f"{v['summary']['pass']} pass / {v['summary']['conditional']} cond / {v['summary']['fail']} fail")

left, right = st.columns(2)
scores["outcome"] = np.where(scores["TARGET"] == 1, "Default", "Non-default")
left.plotly_chart(px.histogram(scores, x="score", color="outcome", nbins=50, barmode="overlay", histnorm="probability density",
                               title="Score distribution by outcome (test set)"), use_container_width=True)
dec = scores.groupby("score_decile").agg(default_rate=("TARGET", "mean"), min_score=("score", "min")).reset_index()
right.plotly_chart(px.bar(dec, x="score_decile", y="default_rate", hover_data=["min_score"],
                          title="Default rate by score decile (1 = lowest score)"), use_container_width=True)

left, right = st.columns(2)
psi = pd.DataFrame({"comparison": ["train vs validation", "train vs test"],
                    "psi": [v["stability"]["psi_train_val"], v["stability"]["psi_train_test"]]})
left.plotly_chart(px.bar(psi, x="comparison", y="psi", title="Score PSI across splits (alert at 0.10)"), use_container_width=True)
top = iv[iv["in_model"]].head(15).sort_values("iv")
right.plotly_chart(px.bar(top, x="iv", y="feature", orientation="h", title=f"Top IV features in model ({m['features']['n_final']} total)"),
                   use_container_width=True)

cal = pd.DataFrame(v["calibration"]["deciles"])
st.plotly_chart(px.line(cal, x="predicted_rate", y="actual_rate", markers=True,
                        title=f"Calibration by decile (Hosmer-Lemeshow p = {v['calibration']['hosmer_lemeshow_p']:.4f})"), use_container_width=True)
st.subheader("Validation tests")
st.dataframe(pd.DataFrame(v["tests"]), use_container_width=True, hide_index=True)

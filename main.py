import json
import mlflow
from pathlib import Path

from pipeline.funnel_pipeline import run_module1
from analysis.causal_analysis import run_module2
from models.dropout_model import run_module3
from analysis.challenger_framework import run_module4
from analysis.monitoring import run_module5

Path("data/outputs").mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlruns.db")

print("=== FunnelIQ — Application Funnel Intelligence Platform ===")

print("\n[1/5] Funnel Lineage & Bottleneck Diagnostics...")
funnel_metrics, funnel_summary = run_module1()

print("\n[2/5] Causal Intervention Analysis...")
causal_results = run_module2()

print("\n[3/5] Drop-Off Classifier...")
model_results = run_module3()

print("\n[4/5] Champion/Challenger Framework...")
exp_results = run_module4(model_results)

print("\n[5/5] Observability & Audit Governance...")
obs_results = run_module5(model_results, causal_results, exp_results["champion_challenger"])

results = {
    "funnel_summary": funnel_summary,
    "causal": causal_results,
    "model": {
        "cv_results": model_results["cv_results"],
        "champion": model_results["champion"],
        "shap_importance": model_results["shap_importance"],
    },
    "experimentation": {
        "power_analysis": exp_results["power_analysis"],
        "power_curve": exp_results["power_curve"],
        "champion_challenger": exp_results["champion_challenger"],
    },
    "observability": {
        "max_psi": obs_results["max_psi"],
        "psi_alerts": obs_results["psi_alerts"],
        "ks_alerts": obs_results["ks_alerts"],
        "spc": obs_results["spc"],
    },
}

with open("data/outputs/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n=== Final Numbers ===")
print(json.dumps({
    "funnel_conversion_rate": funnel_summary["conversion_rate"],
    "did_converted": causal_results["converted"]["did"]["did_estimate"],
    "did_p_value": causal_results["converted"]["did"]["p_value"],
    "xgb_auc": model_results["cv_results"]["xgb_mean_auc"],
    "lgb_auc": model_results["cv_results"]["lgb_mean_auc"],
    "cc_auc_lift": exp_results["champion_challenger"]["auc_lift"],
    "cc_p_value": exp_results["champion_challenger"]["p_value"],
    "power_at_2pct_mde": exp_results["power_analysis"]["statistical_power"],
    "max_psi": obs_results["max_psi"],
    "psi_alerts": obs_results["psi_alerts"],
}, indent=2))

print("\nRun dashboard: streamlit run dashboard/app.py")
print("Run API: python api/server.py")

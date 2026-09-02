import numpy as np
import pandas as pd
import mlflow
from scipy import stats


def compute_psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_bins = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_bins = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_bins = np.where(expected_bins == 0, 1e-4, expected_bins)
    actual_bins = np.where(actual_bins == 0, 1e-4, actual_bins)

    psi = np.sum((actual_bins - expected_bins) * np.log(actual_bins / expected_bins))
    return round(float(psi), 4)


def compute_ks_stat(expected, actual):
    ks_stat, p_value = stats.ks_2samp(expected, actual)
    return round(float(ks_stat), 4), round(float(p_value), 4)


def spc_control_limits(series, n_sigma=3):
    mean = series.mean()
    std = series.std()
    ucl = mean + n_sigma * std
    lcl = max(0, mean - n_sigma * std)
    return round(mean, 4), round(ucl, 4), round(lcl, 4)


def simulate_drift_windows(X, y, model, n_windows=6):
    n = len(X)
    window_size = n // n_windows
    results = []

    baseline_probs = model.predict_proba(X[:window_size])[:, 1]
    baseline_conversion = 1 - baseline_probs.mean()

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, n)
        X_window = X[start:end]
        y_window = y[start:end]

        window_probs = model.predict_proba(X_window)[:, 1]
        psi = compute_psi(baseline_probs, window_probs)
        ks_stat, ks_p = compute_ks_stat(baseline_probs, window_probs)
        conversion_rate = 1 - window_probs.mean()
        drop_off_rate = y_window.mean()

        results.append({
            "window": i,
            "n_applicants": len(X_window),
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_p_value": ks_p,
            "conversion_rate": round(float(conversion_rate), 4),
            "drop_off_rate": round(float(drop_off_rate), 4),
            "mean_score": round(float(window_probs.mean()), 4),
            "psi_alert": psi > 0.25,
            "ks_alert": ks_p < 0.05,
        })

    return pd.DataFrame(results)


def log_governance_metrics(drift_df, cv_results, cc_result):
    mlflow.set_experiment("funneliq_governance")
    with mlflow.start_run(run_name="audit_governance_log"):
        mlflow.log_metric("champion_auc", cv_results.get("xgb_mean_auc", 0))
        mlflow.log_metric("challenger_auc", cv_results.get("lgb_mean_auc", 0))
        mlflow.log_metric("cc_auc_lift", cc_result.get("auc_lift", 0))
        mlflow.log_metric("cc_p_value", cc_result.get("p_value", 1))
        mlflow.log_metric("max_psi_drift", drift_df["psi"].max())
        mlflow.log_metric("psi_alerts_fired", int(drift_df["psi_alert"].sum()))
        mlflow.log_metric("ks_alerts_fired", int(drift_df["ks_alert"].sum()))
        mlflow.log_metric("mean_conversion_rate", drift_df["conversion_rate"].mean())
        mlflow.set_tag("audit_ready", "true")
        mlflow.set_tag("governance_framework", "PSI+KS+SPC")
        print("MLflow governance log recorded")


def run_module5(model_results, causal_results, cc_result):
    print("\n=== Module 5: Observability & Audit Governance ===")

    X = model_results["X"]
    y = model_results["y"]
    champion_model = model_results["models"]["xgb"]

    drift_df = simulate_drift_windows(X, y, champion_model)

    mean_conv, ucl, lcl = spc_control_limits(drift_df["conversion_rate"])

    drift_df["spc_breach"] = (drift_df["conversion_rate"] > ucl) | (drift_df["conversion_rate"] < lcl)

    print(f"\nDrift monitoring across {len(drift_df)} time windows:")
    print(drift_df[["window", "psi", "ks_stat", "conversion_rate", "psi_alert", "ks_alert", "spc_breach"]].to_string(index=False))
    print(f"\nSPC Control Limits (conversion rate):")
    print(f"  Mean: {mean_conv}, UCL: {ucl}, LCL: {lcl}")
    print(f"  SPC breaches: {drift_df['spc_breach'].sum()}")

    log_governance_metrics(drift_df, model_results["cv_results"], cc_result)

    drift_df.to_csv("data/outputs/drift_monitoring.csv", index=False)

    return {
        "drift_df": drift_df,
        "spc": {"mean": mean_conv, "ucl": ucl, "lcl": lcl},
        "max_psi": round(float(drift_df["psi"].max()), 4),
        "psi_alerts": int(drift_df["psi_alert"].sum()),
        "ks_alerts": int(drift_df["ks_alert"].sum()),
    }

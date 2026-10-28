import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, roc_curve
from credit_risk.scorecard import to_woe, bin_labels, to_score, discrimination

PROCESSED = Path("data/processed")
OUTPUTS = Path("data/outputs")
IMAGES = Path("images")
SEED = 42
N_BOOT = 1000
ID = "SK_ID_CURR"
TARGET = "TARGET"


def status(value, pass_if, conditional_if):
    if pass_if(value):
        return "PASS"
    if conditional_if(value):
        return "CONDITIONAL"
    return "FAIL"


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def bootstrap(y, p_sc, p_ch):
    rng = np.random.default_rng(SEED)
    y, p_sc, p_ch = np.asarray(y), np.asarray(p_sc), np.asarray(p_ch)
    auc_sc, auc_ch, ks_sc = [], [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        auc_sc.append(roc_auc_score(y[i], p_sc[i]))
        auc_ch.append(roc_auc_score(y[i], p_ch[i]))
        ks_sc.append(ks_stat(y[i], p_sc[i]))
    return np.array(auc_sc), np.array(auc_ch), np.array(ks_sc)


def ci(a):
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def psi(expected, actual, edges):
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def csi(train_bins, test_bins):
    e = train_bins.value_counts(normalize=True)
    a = test_bins.value_counts(normalize=True).reindex(e.index, fill_value=0)
    e, a = np.clip(e.values, 1e-6, None), np.clip(a.values, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration(y, p):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=False) + 1
    g = d.groupby("decile").agg(n=("y", "size"), actual=("y", "sum"), expected=("p", "sum"))
    hl = float((((g["actual"] - g["expected"]) ** 2) / (g["expected"] * (1 - g["expected"] / g["n"]))).sum())
    table = [{
        "decile": int(k),
        "n": int(r["n"]),
        "predicted_rate": round(float(r["expected"] / r["n"]), 4),
        "actual_rate": round(float(r["actual"] / r["n"]), 4),
    } for k, r in g.iterrows()]
    return table, hl, float(chi2.sf(hl, len(g) - 2))


def vif(woe_train):
    inv = np.linalg.inv(np.corrcoef(woe_train.values, rowvar=False))
    return {f: round(float(v), 2) for f, v in zip(woe_train.columns, np.diag(inv))}


def plots(y, p_sc, p_ch, cal):
    IMAGES.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 5))
    for name, p in (("Scorecard", p_sc), ("LightGBM challenger", p_ch)):
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y, p):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC - test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES / "roc.png", dpi=150)
    plt.close()

    s = to_score(np.asarray(p_sc))
    order = np.argsort(s)
    yy = np.asarray(y)[order]
    cum_bad = np.cumsum(yy) / yy.sum()
    cum_good = np.cumsum(1 - yy) / (1 - yy).sum()
    k = int(np.argmax(np.abs(cum_bad - cum_good)))
    plt.figure(figsize=(6, 5))
    plt.plot(s[order], cum_bad, label="Defaults")
    plt.plot(s[order], cum_good, label="Non-defaults")
    plt.vlines(s[order][k], cum_good[k], cum_bad[k], colors="red", label=f"KS={abs(cum_bad[k] - cum_good[k]):.3f}")
    plt.xlabel("Score")
    plt.ylabel("Cumulative share")
    plt.title("KS - scorecard, test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES / "ks.png", dpi=150)
    plt.close()

    c = pd.DataFrame(cal)
    plt.figure(figsize=(6, 5))
    plt.plot(c["predicted_rate"], c["actual_rate"], "o-", label="Scorecard deciles")
    top = max(c["predicted_rate"].max(), c["actual_rate"].max())
    plt.plot([0, top], [0, top], "--", color="grey", label="Perfect calibration")
    plt.xlabel("Predicted default rate")
    plt.ylabel("Actual default rate")
    plt.title("Calibration - test set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES / "calibration.png", dpi=150)
    plt.close()


def run():
    m = joblib.load(PROCESSED / "credit_models.joblib")
    binning, final, lr = m["binning"], m["final"], m["lr"]
    parts = {k: pd.read_parquet(PROCESSED / f"credit_{k}.parquet") for k in ("train", "val", "test")}
    preds = {k: pd.read_parquet(PROCESSED / f"preds_{k}.parquet") for k in ("train", "val", "test")}
    iv = pd.read_csv(OUTPUTS / "iv_table.csv")
    test, pt = parts["test"], preds["test"]
    y = pt[TARGET].values

    auc_sc, auc_ch, ks_sc = bootstrap(y, pt["pd_scorecard"], pt["pd_challenger"])
    point = discrimination(y, pt["pd_scorecard"])
    diff = auc_ch - auc_sc
    p_value = max(2 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0))), 1 / N_BOOT)

    cal, hl, hl_p = calibration(y, pt["pd_scorecard"])
    brier = float(np.mean((pt["pd_scorecard"].values - y) ** 2))
    brier_ref = float(y.mean() * (1 - y.mean()))
    brier_skill = 1 - brier / brier_ref

    edges = np.unique(np.quantile(preds["train"]["score"], np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    psi_val = psi(preds["train"]["score"], preds["val"]["score"], edges)
    psi_test = psi(preds["train"]["score"], pt["score"], edges)
    csis = {f: round(csi(bin_labels(parts["train"][f], binning[f]), bin_labels(test[f], binning[f])), 4) for f in final}
    n_csi = sum(v > 0.1 for v in csis.values())

    numeric_final = iv[iv["feature"].isin(final) & (iv["type"] == "numeric")].sort_values("iv", ascending=False)["feature"].head(5).tolist()
    base_pd = float(pt["pd_scorecard"].mean())
    sens = []
    for f in numeric_final:
        row = {"feature": f}
        for shift in (-0.1, 0.1):
            shifted = test.copy()
            shifted[f] = shifted[f] * (1 + shift)
            p = lr.predict_proba(to_woe(shifted, binning, final))[:, 1]
            row[f"mean_pd_change_pct_{'minus' if shift < 0 else 'plus'}10"] = round(float((p.mean() - base_pd) / base_pd * 100), 2)
        sens.append(row)
    max_sens = max(abs(v) for r in sens for k, v in r.items() if k != "feature")

    vifs = vif(to_woe(parts["train"], binning, final))
    mono = {f: bool(iv.loc[iv["feature"] == f, "monotonic"].iloc[0]) for f in final}

    tests = [
        ("Discrimination", "Test AUC", point["auc"], ">=0.70 PASS, >=0.65 CONDITIONAL", status(point["auc"], lambda v: v >= 0.70, lambda v: v >= 0.65)),
        ("Discrimination", "Test Gini", point["gini"], ">=0.40 PASS, >=0.30 CONDITIONAL", status(point["gini"], lambda v: v >= 0.40, lambda v: v >= 0.30)),
        ("Discrimination", "Test KS", point["ks"], ">=0.30 PASS, >=0.20 CONDITIONAL", status(point["ks"], lambda v: v >= 0.30, lambda v: v >= 0.20)),
        ("Calibration", "Hosmer-Lemeshow p-value", round(hl_p, 4), ">0.05 PASS, >0.01 CONDITIONAL", status(hl_p, lambda v: v > 0.05, lambda v: v > 0.01)),
        ("Calibration", "Brier skill score", round(brier_skill, 4), ">0.05 PASS, >0 CONDITIONAL", status(brier_skill, lambda v: v > 0.05, lambda v: v > 0)),
        ("Stability", "Score PSI train vs test", round(psi_test, 4), "<0.10 PASS, <0.25 CONDITIONAL", status(psi_test, lambda v: v < 0.10, lambda v: v < 0.25)),
        ("Stability", "Features with CSI > 0.1", n_csi, "0 PASS, <=2 CONDITIONAL", status(n_csi, lambda v: v == 0, lambda v: v <= 2)),
        ("Benchmark", "Challenger minus scorecard AUC", round(float(diff.mean()), 4), "p>=0.05 PASS, else CONDITIONAL", "PASS" if p_value >= 0.05 else "CONDITIONAL"),
        ("Sensitivity", "Max mean PD change for +/-10% shift (%)", max_sens, "<=10 PASS, <=25 CONDITIONAL", status(max_sens, lambda v: v <= 10, lambda v: v <= 25)),
        ("Assumptions", "Max VIF", max(vifs.values()), "<=5 PASS, <=10 CONDITIONAL", status(max(vifs.values()), lambda v: v <= 5, lambda v: v <= 10)),
        ("Assumptions", "WOE monotonic features", f"{sum(mono.values())}/{len(mono)}", "all PASS, else CONDITIONAL", "PASS" if all(mono.values()) else "CONDITIONAL"),
    ]
    tests = [{"category": c, "test": t, "value": v, "threshold": th, "status": s} for c, t, v, th, s in tests]
    counts = {s: sum(t["status"] == s for t in tests) for s in ("PASS", "CONDITIONAL", "FAIL")}
    rating = "Satisfactory" if counts["FAIL"] == 0 and counts["CONDITIONAL"] == 0 else ("Satisfactory with findings" if counts["FAIL"] == 0 else "Requires remediation")

    results = {
        "model": "WOE logistic scorecard (champion) vs LightGBM (challenger)",
        "test_rows": int(len(y)),
        "test_default_rate": round(float(y.mean()), 4),
        "ci_level": 0.95,
        "n_bootstrap": N_BOOT,
        "discrimination": {
            "scorecard": {**point, "auc_ci": ci(auc_sc), "gini_ci": ci(2 * auc_sc - 1), "ks_ci": ci(ks_sc)},
            "challenger": {**discrimination(y, pt["pd_challenger"]), "auc_ci": ci(auc_ch), "gini_ci": ci(2 * auc_ch - 1)},
        },
        "benchmark": {"auc_difference": round(float(diff.mean()), 4), "auc_difference_ci": ci(diff), "p_value": round(p_value, 4)},
        "calibration": {"deciles": cal, "hosmer_lemeshow_stat": round(hl, 2), "hosmer_lemeshow_p": round(hl_p, 4),
                        "brier": round(brier, 4), "brier_reference": round(brier_ref, 4), "brier_skill": round(brier_skill, 4)},
        "stability": {"psi_train_val": round(psi_val, 4), "psi_train_test": round(psi_test, 4), "csi": csis, "n_csi_above_0_1": n_csi},
        "sensitivity": sens,
        "vif": vifs,
        "monotonicity": mono,
        "tests": tests,
        "summary": {"n_tests": len(tests), "pass": counts["PASS"], "conditional": counts["CONDITIONAL"], "fail": counts["FAIL"], "overall_rating": rating},
    }
    with open(OUTPUTS / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    scores = pt[[ID, TARGET, "score", "pd_scorecard", "pd_challenger"]].copy()
    scores["score_decile"] = pd.qcut(scores["score"].rank(method="first"), 10, labels=False) + 1
    scores.round(4).to_csv(OUTPUTS / "credit_scores.csv", index=False)

    plots(y, pt["pd_scorecard"].values, pt["pd_challenger"].values, cal)
    print(f"[validation] AUC {point['auc']} CI {ci(auc_sc)}, KS {point['ks']}, HL p {hl_p:.4f}, Brier {brier:.4f}, PSI {psi_test:.4f}, CSI>0.1 {n_csi}")
    print(f"[validation] {counts} -> {rating}")
    return results


if __name__ == "__main__":
    run()

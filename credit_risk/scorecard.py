import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

PROCESSED = Path("data/processed")
OUTPUTS = Path("data/outputs")
SEED = 42
ID = "SK_ID_CURR"
TARGET = "TARGET"
BASE_SCORE = 600
PDO = 20
ODDS = 50
N_BINS = 10
IV_MIN = 0.02
IV_MAX = 0.5
MAX_CORR = 0.7
RARE = 0.01


def woe_table(bins, y):
    t = pd.DataFrame({"bin": bins, "y": y}).groupby("bin", observed=True)["y"].agg(["count", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    dist_bad = (t["bad"] + 0.5) / (t["bad"].sum() + 0.5)
    dist_good = (t["good"] + 0.5) / (t["good"].sum() + 0.5)
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv"] = (dist_good - dist_bad) * t["woe"]
    return t


def numeric_bin_ids(x, edges):
    ids = pd.cut(x, edges, labels=False, right=True, include_lowest=True)
    return ids.astype("float").fillna(-1).astype(int)


def fit_numeric(x, y):
    inner = np.unique(np.nanquantile(x.dropna(), np.linspace(0, 1, N_BINS + 1)))[1:-1]
    if len(inner) == 0:
        u = np.sort(x.dropna().unique())
        if len(u) < 2:
            return None
        inner = (u[:-1] + u[1:]) / 2
    edges = [-np.inf] + list(inner) + [np.inf]
    while True:
        t = woe_table(numeric_bin_ids(x, edges), y)
        w = t.loc[t.index >= 0, "woe"].values
        if len(w) <= 2:
            break
        diffs = np.diff(w) * np.sign(w[-1] - w[0] or 1)
        if (diffs >= 0).all():
            break
        del edges[int(np.argmin(diffs)) + 1]
    t = woe_table(numeric_bin_ids(x, edges), y)
    return {"type": "numeric", "edges": edges, "woe": t["woe"].to_dict(), "table": t}


def category_labels(x, levels):
    s = x.astype("object").where(x.notna(), "MISSING")
    return s.where(s.isin(levels) | (s == "MISSING"), "OTHER")


def fit_categorical(x, y):
    share = x.value_counts(normalize=True)
    levels = list(share[share >= RARE].index)
    t = woe_table(category_labels(x, levels), y)
    return {"type": "categorical", "levels": levels, "woe": t["woe"].to_dict(), "table": t}


def bin_labels(x, spec):
    if spec["type"] == "numeric":
        return numeric_bin_ids(x, spec["edges"])
    return category_labels(x, spec["levels"])


def to_woe(df, binning, features):
    return pd.DataFrame({f: bin_labels(df[f], binning[f]).map(binning[f]["woe"]).astype(float).fillna(0.0).values for f in features}, index=df.index)


def is_monotonic(spec):
    if spec["type"] != "numeric":
        return True
    w = spec["table"].loc[spec["table"].index >= 0, "woe"].values
    d = np.diff(w)
    return bool((d >= 0).all() or (d <= 0).all())


def discrimination(y, p):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {"auc": round(float(auc), 4), "gini": round(float(2 * auc - 1), 4), "ks": round(float(np.max(tpr - fpr)), 4)}


def score_params():
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(ODDS)
    return factor, offset


def to_score(p):
    factor, offset = score_params()
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return offset - factor * np.log(p / (1 - p))


def select_features(iv, woe_train):
    candidates = iv[(iv["iv"] > IV_MIN) & (iv["iv"] < IV_MAX)].sort_values("iv", ascending=False)["feature"].tolist()
    kept = []
    corr = woe_train[candidates].corr().abs()
    for f in candidates:
        if all(corr.loc[f, k] < MAX_CORR for k in kept):
            kept.append(f)
    return candidates, kept


def fit_logistic(X, y, features):
    features = list(features)
    while True:
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(X[features], y)
        positive = [f for f, c in zip(features, lr.coef_[0]) if c > 0]
        if not positive:
            return lr, features
        features = [f for f in features if f not in positive]


def points_table(binning, features, lr):
    factor, offset = score_params()
    n = len(features)
    b0 = lr.intercept_[0]
    rows = []
    for f, b in zip(features, lr.coef_[0]):
        for bin_id, w in binning[f]["woe"].items():
            rows.append({
                "feature": f,
                "bin": str(bin_id),
                "woe": round(float(w), 4),
                "coefficient": round(float(b), 4),
                "points": round(float(-factor * (b * w + b0 / n) + offset / n), 2),
            })
    return pd.DataFrame(rows)


def lgb_frame(df, features, cat_levels):
    X = df[features].copy()
    for c, levels in cat_levels.items():
        X[c] = pd.Categorical(X[c].astype("object"), categories=levels)
    return X


def run():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    parts = {k: pd.read_parquet(PROCESSED / f"credit_{k}.parquet") for k in ("train", "val", "test")}
    train = parts["train"]
    y = train[TARGET]
    features = [c for c in train.columns if c not in (ID, TARGET)]

    binning = {}
    for f in features:
        spec = fit_numeric(train[f], y) if pd.api.types.is_numeric_dtype(train[f]) else fit_categorical(train[f], y)
        if spec is not None:
            binning[f] = spec

    iv = pd.DataFrame([{
        "feature": f,
        "type": s["type"],
        "iv": round(float(s["table"]["iv"].sum()), 4),
        "n_bins": len(s["table"]),
        "monotonic": is_monotonic(s),
    } for f, s in binning.items()]).sort_values("iv", ascending=False)
    iv["selected"] = (iv["iv"] > IV_MIN) & (iv["iv"] < IV_MAX)
    iv["suspicious"] = iv["iv"] >= IV_MAX

    woe = {k: to_woe(v, binning, list(binning)) for k, v in parts.items()}
    candidates, uncorrelated = select_features(iv, woe["train"])
    lr, final = fit_logistic(woe["train"], y, uncorrelated)
    iv["in_model"] = iv["feature"].isin(final)
    iv.to_csv(OUTPUTS / "iv_table.csv", index=False)
    points_table(binning, final, lr).to_csv(OUTPUTS / "scorecard_points.csv", index=False)

    cat_levels = {c: sorted(train[c].dropna().astype("object").unique().tolist()) for c in features if not pd.api.types.is_numeric_dtype(train[c])}
    X = {k: lgb_frame(v, features, cat_levels) for k, v in parts.items()}
    gbm = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
        random_state=SEED, verbose=-1,
    )
    gbm.fit(X["train"], y, eval_set=[(X["val"], parts["val"][TARGET])], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)])

    metrics = {"scorecard": {}, "challenger": {}}
    for k, part in parts.items():
        p_sc = lr.predict_proba(woe[k][final])[:, 1]
        p_ch = gbm.predict_proba(X[k])[:, 1]
        metrics["scorecard"][k] = discrimination(part[TARGET], p_sc)
        metrics["challenger"][k] = discrimination(part[TARGET], p_ch)
        pd.DataFrame({
            ID: part[ID].values,
            TARGET: part[TARGET].values,
            "pd_scorecard": p_sc,
            "score": to_score(p_sc),
            "pd_challenger": p_ch,
        }).to_parquet(PROCESSED / f"preds_{k}.parquet", index=False)

    metrics["scorecard_config"] = {"base_score": BASE_SCORE, "pdo": PDO, "odds": ODDS}
    metrics["features"] = {
        "n_candidates": len(binning),
        "n_iv_selected": len(candidates),
        "n_after_correlation_filter": len(uncorrelated),
        "n_final": len(final),
        "final": final,
        "iv_range": [IV_MIN, IV_MAX],
        "max_correlation": MAX_CORR,
        "suspicious_iv": iv.loc[iv["suspicious"], "feature"].tolist(),
    }
    metrics["challenger_config"] = {"model": "LightGBM", "best_iteration": int(gbm.best_iteration_), "n_features": len(features)}
    with open(OUTPUTS / "credit_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump({"binning": binning, "final": final, "lr": lr, "gbm": gbm, "features": features, "cat_levels": cat_levels},
                PROCESSED / "credit_models.joblib")

    print(f"[scorecard] {len(binning)} binned, {len(candidates)} IV-selected, {len(uncorrelated)} after correlation filter, {len(final)} in model")
    for m in ("scorecard", "challenger"):
        print(f"[scorecard] {m} test {metrics[m]['test']}")
    return metrics


if __name__ == "__main__":
    run()

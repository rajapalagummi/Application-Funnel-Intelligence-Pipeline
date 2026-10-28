import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
OUTPUTS = Path("data/outputs")
SEED = 42
ID = "SK_ID_CURR"
TARGET = "TARGET"
LEAKAGE_AUC = 0.90


def aggregate_previous(path):
    p = pd.read_csv(path, usecols=[ID, "NAME_CONTRACT_STATUS", "AMT_CREDIT", "DAYS_DECISION"])
    p["approved"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["refused"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    return p.groupby(ID).agg(
        PREV_COUNT=("approved", "size"),
        PREV_APPROVED_RATE=("approved", "mean"),
        PREV_REFUSED_RATE=("refused", "mean"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
    ).reset_index()


def aggregate_bureau(path):
    b = pd.read_csv(path, usecols=[ID, "CREDIT_ACTIVE", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT"])
    b["active"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    return b.groupby(ID).agg(
        BUREAU_COUNT=("active", "size"),
        BUREAU_ACTIVE_COUNT=("active", "sum"),
        BUREAU_MAX_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
    ).reset_index()


def load():
    path = RAW / "application_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: credit risk track requires real Home Credit data")
    df = pd.read_csv(path)
    sources = ["application_train"]
    if (RAW / "previous_application.csv").exists():
        df = df.merge(aggregate_previous(RAW / "previous_application.csv"), on=ID, how="left")
        sources.append("previous_application")
    if (RAW / "bureau.csv").exists():
        df = df.merge(aggregate_bureau(RAW / "bureau.csv"), on=ID, how="left")
        sources.append("bureau")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    return df, sources


def split(df):
    train, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=SEED)
    return train, val, test


def univariate_auc(train, features):
    out = {}
    y = train[TARGET]
    for f in features:
        x = train[f]
        if x.nunique() < 2:
            continue
        auc = roc_auc_score(y, x.fillna(x.median()))
        out[f] = round(max(auc, 1 - auc), 4)
    return out


def run():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df, sources = load()
    features = [c for c in df.columns if c not in (ID, TARGET)]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    train, val, test = split(df)

    ids = {k: set(v[ID]) for k, v in {"train": train, "val": val, "test": test}.items()}
    overlap = {
        "train_val": len(ids["train"] & ids["val"]),
        "train_test": len(ids["train"] & ids["test"]),
        "val_test": len(ids["val"] & ids["test"]),
    }
    aucs = univariate_auc(train, numeric)
    flags = sorted([f for f, a in aucs.items() if a > LEAKAGE_AUC])
    top = max(aucs, key=aucs.get)

    for name, part in {"train": train, "val": val, "test": test}.items():
        part.to_parquet(PROCESSED / f"credit_{name}.parquet", index=False)

    summary = {
        "data_source": "home_credit_real",
        "sources": sources,
        "rows_total": len(df),
        "default_rate_total": round(float(df[TARGET].mean()), 4),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "default_rate": {k: round(float(v[TARGET].mean()), 4) for k, v in {"train": train, "val": val, "test": test}.items()},
        "n_features": len(features),
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "split": "60/20/20 stratified, seed 42",
        "id_overlap": overlap,
        "leakage_threshold_auc": LEAKAGE_AUC,
        "leakage_flags": flags,
        "max_univariate_auc": {"feature": top, "auc": aucs[top]},
    }
    with open(OUTPUTS / "credit_data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[data_prep] {len(df):,} rows, default rate {summary['default_rate_total']:.2%}, {len(features)} features")
    print(f"[data_prep] splits {summary['rows']}, id overlap {overlap}, leakage flags {flags}")
    return summary


if __name__ == "__main__":
    run()

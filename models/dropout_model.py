import duckdb
import numpy as np
import pandas as pd
import shap
import mlflow
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

DB_PATH = "data/funneliq.duckdb"


def load_model_data(con):
    app = con.execute("SELECT * FROM application_train").df()
    funnel = con.execute("""
        SELECT SK_ID_CURR, final_stage, drop_stage, policy_group,
               days_since_decision, AMT_APPLICATION
        FROM funnel_stages
    """).df()

    if "SK_ID_CURR" in app.columns and "SK_ID_CURR" in funnel.columns:
        df = funnel.merge(app, on="SK_ID_CURR", how="left")
    else:
        df = funnel.copy()
        df["drop_off_target"] = (df["final_stage"] < 5).astype(int)

    return df


def engineer_features(df):
    df = df.copy()

    df["drop_off"] = (df["final_stage"] < 5).astype(int)

    cat_cols = ["policy_group", "NAME_CLIENT_TYPE"] if "NAME_CLIENT_TYPE" in df.columns else ["policy_group"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    num_cols = ["days_since_decision", "AMT_APPLICATION",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
            "AMT_INCOME_TOTAL", "AMT_CREDIT",
            "DAYS_BIRTH", "DAYS_EMPLOYED",
            "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_6"]

    feature_cols = [c for c in num_cols if c in df.columns]
    feature_cols += [c for c in df.columns if c.endswith("_enc")]

    df_clean = df[feature_cols + ["drop_off"]].dropna(subset=["drop_off"])
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    return df_clean, feature_cols


def train_xgboost(X, y):
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y[y == 0]) / len(y[y == 1]),
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    )
    model.fit(X, y)
    return model


def train_lightgbm(X, y):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def cross_validate_models(df_clean, feature_cols):
    X = df_clean[feature_cols].values
    y = df_clean["drop_off"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_aucs, lgb_aucs = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        xgb_model = train_xgboost(X_train, y_train)
        lgb_model = train_lightgbm(X_train, y_train)

        xgb_aucs.append(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))
        lgb_aucs.append(roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1]))

    return {
        "xgb_mean_auc": round(np.mean(xgb_aucs), 4),
        "xgb_std_auc": round(np.std(xgb_aucs), 4),
        "lgb_mean_auc": round(np.mean(lgb_aucs), 4),
        "lgb_std_auc": round(np.std(lgb_aucs), 4),
        "n_folds": 5,
    }


def compute_shap(model, X, feature_cols):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:500])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_cols, [round(float(v), 4) for v in mean_abs_shap]))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])


def run_module3():
    mlflow.set_experiment("funneliq_dropout_model")
    con = duckdb.connect(DB_PATH)
    df = load_model_data(con)
    con.close()

    df_clean, feature_cols = engineer_features(df)
    X = df_clean[feature_cols].values
    y = df_clean["drop_off"].values

    print(f"\n=== Module 3: Drop-Off Classifier ===")
    print(f"Training samples: {len(X):,}, Drop-off rate: {y.mean():.2%}")

    with mlflow.start_run(run_name="xgb_lgb_dropout"):
        cv_results = cross_validate_models(df_clean, feature_cols)

        xgb_model = train_xgboost(X, y)
        lgb_model = train_lightgbm(X, y)

        champion_model = xgb_model if cv_results["xgb_mean_auc"] >= cv_results["lgb_mean_auc"] else lgb_model
        champion_name = "XGBoost" if cv_results["xgb_mean_auc"] >= cv_results["lgb_mean_auc"] else "LightGBM"

        shap_importance = compute_shap(champion_model, X, feature_cols)

        mlflow.log_metric("xgb_mean_auc", cv_results["xgb_mean_auc"])
        mlflow.log_metric("lgb_mean_auc", cv_results["lgb_mean_auc"])
        mlflow.log_param("champion_model", champion_name)

        print(f"\nCV Results:")
        print(f"  XGBoost AUC: {cv_results['xgb_mean_auc']} ± {cv_results['xgb_std_auc']}")
        print(f"  LightGBM AUC: {cv_results['lgb_mean_auc']} ± {cv_results['lgb_std_auc']}")
        print(f"  Champion: {champion_name}")
        print(f"\nTop SHAP drivers of drop-off:")
        for feat, val in list(shap_importance.items())[:5]:
            print(f"  {feat}: {val}")

    return {
        "cv_results": cv_results,
        "champion": champion_name,
        "shap_importance": shap_importance,
        "models": {"xgb": xgb_model, "lgb": lgb_model},
        "features": feature_cols,
        "X": X, "y": y,
    }

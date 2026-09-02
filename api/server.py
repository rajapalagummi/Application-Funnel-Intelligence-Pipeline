import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="FunnelIQ Scoring API")

RESULTS_PATH = Path("data/outputs/results.json")
_model_cache = {}


def load_model():
    if "model" not in _model_cache:
        try:
            from models.dropout_model import load_model_data, engineer_features, train_xgboost
            import duckdb
            con = duckdb.connect("data/funneliq.duckdb")
            df = load_model_data(con)
            con.close()
            df_clean, feature_cols = engineer_features(df)
            X = df_clean[feature_cols].values
            y = df_clean["drop_off"].values
            model = train_xgboost(X, y)
            _model_cache["model"] = model
            _model_cache["features"] = feature_cols
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not ready: {e}")
    return _model_cache["model"], _model_cache["features"]


class ApplicantFeatures(BaseModel):
    days_since_decision: float = 300.0
    AMT_APPLICATION: float = 150000.0
    EXT_SOURCE_1: float = 0.5
    EXT_SOURCE_2: float = 0.5
    EXT_SOURCE_3: float = 0.5
    AMT_INCOME_TOTAL: float = 180000.0
    AMT_CREDIT: float = 200000.0
    DAYS_BIRTH: float = -12000.0
    DAYS_EMPLOYED: float = -2000.0
    FLAG_DOCUMENT_3: int = 1
    FLAG_DOCUMENT_6: int = 0
    final_stage: int = 3


class ScoreResponse(BaseModel):
    drop_off_probability: float
    conversion_probability: float
    risk_tier: str
    top_shap_drivers: dict
    routing_decision: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ScoreResponse)
def predict(features: ApplicantFeatures):
    model, feature_cols = load_model()

    feature_values = {
        "days_since_decision": features.days_since_decision,
        "AMT_APPLICATION": features.AMT_APPLICATION,
        "EXT_SOURCE_1": features.EXT_SOURCE_1,
        "EXT_SOURCE_2": features.EXT_SOURCE_2,
        "EXT_SOURCE_3": features.EXT_SOURCE_3,
        "AMT_INCOME_TOTAL": features.AMT_INCOME_TOTAL,
        "AMT_CREDIT": features.AMT_CREDIT,
        "DAYS_BIRTH": features.DAYS_BIRTH,
        "DAYS_EMPLOYED": features.DAYS_EMPLOYED,
        "FLAG_DOCUMENT_3": features.FLAG_DOCUMENT_3,
        "FLAG_DOCUMENT_6": features.FLAG_DOCUMENT_6,
        "final_stage": features.final_stage,
    }

    X = np.array([[feature_values.get(f, 0.0) for f in feature_cols]])
    drop_prob = float(model.predict_proba(X)[0][1])
    conv_prob = round(1 - drop_prob, 4)
    drop_prob = round(drop_prob, 4)

    if drop_prob < 0.3:
        risk_tier = "low"
        routing = "automated_pass_through"
    elif drop_prob < 0.6:
        risk_tier = "medium"
        routing = "standard_review"
    else:
        risk_tier = "high"
        routing = "manual_verification_queue"

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_dict = {f: round(float(v), 4) for f, v in zip(feature_cols, shap_vals[0])}
        top_shap = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    except Exception:
        top_shap = {}

    return ScoreResponse(
        drop_off_probability=drop_prob,
        conversion_probability=conv_prob,
        risk_tier=risk_tier,
        top_shap_drivers=top_shap,
        routing_decision=routing,
    )


@app.get("/results")
def get_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    raise HTTPException(status_code=404, detail="Run main.py first")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

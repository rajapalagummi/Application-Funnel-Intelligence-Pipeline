# FunnelIQ — Application Funnel Intelligence & Conversion Optimization Platform

> End-to-end production platform modeling a multi-stage loan application funnel — causal intervention analysis, explainable drop-off prediction, champion/challenger experimentation, and audit-ready observability built on Home Credit public data.

---

## Problem Statement

Multi-step application funnels lose applicants at every stage — document submission, identity verification, credit check, offer presentation, and offer acceptance. Most analytics teams can measure *where* applicants drop off but not *why*, *what intervention would help*, or *whether a model change actually improved things*. FunnelIQ addresses all three.

---

## Dataset

**Home Credit Default Risk** — 307K real loan applications across 8 relational tables including `previous_application.csv` (historical application outcomes with stage-level metadata) and `application_train.csv` (120+ applicant features).

Download:
```bash
kaggle datasets download -d home-credit/home-credit-credit-risk-model-stability
unzip home-credit-credit-risk-model-stability.zip -d data/raw/
```

The pipeline runs on synthetic data if Kaggle credentials are unavailable.

---

## Architecture — 5 Modules

```
Home Credit Data (previous_application.csv + application_train.csv)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Module 1: Funnel Lineage & Bottleneck Diagnostics  │
│  DuckDB SQL → stage conversion rates, time-to-      │
│  decision, cohort friction points                   │
└─────────────────┬───────────────────────────────────┘
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
┌───────────┐ ┌───────────┐ ┌───────────────────────┐
│ Module 2  │ │ Module 3  │ │ Module 4              │
│ DiD Causal│ │ XGBoost + │ │ Champion/Challenger   │
│ Analysis  │ │ LightGBM  │ │ Monte Carlo Power     │
│ + Placebo │ │ + SHAP    │ │ Simulation            │
└─────┬─────┘ └─────┬─────┘ └──────────┬────────────┘
      └─────────────┼───────────────────┘
                    ▼
     ┌──────────────────────────────┐
     │  Module 5: Observability     │
     │  PSI + KS-Test drift         │
     │  SPC control limits          │
     │  MLflow audit governance     │
     └──────────────────────────────┘
             │              │
             ▼              ▼
     FastAPI /predict   Streamlit
     scoring service    dashboard
```

---

## Key Results

| Module | Metric | Value |
|---|---|---|
| Funnel | End-to-end conversion rate | TBD after run |
| Causal | DiD estimate (submit rate) | TBD after run |
| Causal | DiD p-value | TBD after run |
| Model | XGBoost CV AUC | TBD after run |
| Model | LightGBM CV AUC | TBD after run |
| Challenger | AUC lift vs champion | TBD after run |
| Challenger | Statistical power at 2% MDE | TBD after run |
| Observability | Max PSI drift | TBD after run |

---

## Methodology

**Module 1 — Funnel Lineage:**
DuckDB SQL pipeline mapping `NAME_CONTRACT_STATUS` and `CODE_REJECT_REASON` to five funnel stages. Computes stage-level conversion rates, median time-to-decision per stage, and cohort-level friction points.

**Module 2 — Causal Analysis:**
Difference-in-Differences on `FLAG_LAST_APPL_PER_CONTRACT` as a policy change instrument. Parallel trends tested before estimation. Placebo test on pre-period data validates no spurious effect.

$$\hat{\tau}_{DiD} = (\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}) - (\bar{Y}_{ctrl,post} - \bar{Y}_{ctrl,pre})$$

**Module 3 — Drop-Off Classifier:**
XGBoost and LightGBM trained on 120+ applicant features with StratifiedKFold CV. SHAP TreeExplainer computes per-inference feature attribution. Champion selected by CV AUC.

**Module 4 — Champion/Challenger:**
80/20 traffic split across 100 simulation rounds. Monte Carlo power curves compute MDE and required sample sizes. Welch t-test with Cohen's d for effect sizing.

**Module 5 — Observability:**
Population Stability Index (PSI > 0.25 = alert), KS-Test (p < 0.05 = alert), Statistical Process Control (3-sigma limits on conversion rate). MLflow governance log tagged `audit_ready`.

---

## Stack

| Component | Technology |
|---|---|
| Data & SQL | DuckDB, Pandas |
| ML Models | XGBoost, LightGBM, Scikit-learn |
| Explainability | SHAP TreeExplainer |
| Causal Inference | SciPy, statsmodels |
| Experimentation | Monte Carlo simulation, SciPy t-test |
| Observability | PSI, KS-Test, SPC, MLflow |
| API | FastAPI |
| Dashboard | Streamlit, Plotly |
| CI/CD | GitHub Actions |

---

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: download real data
kaggle datasets download -d home-credit/home-credit-credit-risk-model-stability
unzip *.zip -d data/raw/

# Run pipeline
python main.py

# Dashboard
streamlit run dashboard/app.py

# API
python api/server.py

# Tests
pytest tests/ -v
```

---

## References

Home Credit Group (2018). Home Credit Default Risk. *Kaggle Competition Dataset*.

Angrist, J.D. & Pischke, J.S. (2008). *Mostly Harmless Econometrics*. Princeton University Press.

Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.

Lundberg, S. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.

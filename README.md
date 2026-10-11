# LoanPath — Application Funnel Intelligence & Conversion Optimization Platform

> End-to-end production platform modeling a multi-stage loan application funnel — causal intervention analysis, explainable drop-off prediction, champion/challenger experimentation, and audit-ready observability built on 307K real Home Credit loan applications.

---

## Problem Statement

Multi-step application funnels lose applicants at every stage — document submission, identity verification, credit check, offer presentation, and offer acceptance. Most analytics teams can measure *where* applicants drop off but not *why*, *what intervention would help*, or *whether a model change actually improved things*. LoanPath addresses all three using real loan application data and production-grade methodology.

---

## Dataset

**Home Credit Default Risk** — 307K real loan applications across 8 relational tables.

Key files used:
- `previous_application.csv` — historical application outcomes with stage-level metadata (`NAME_CONTRACT_STATUS`, `CODE_REJECT_REASON`, `DAYS_DECISION`)
- `application_train.csv` — 120+ applicant features including external credit scores, income, document flags

Download:
```bash
# Accept competition rules at kaggle.com/competitions/home-credit-default-risk first
kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/raw/
kaggle competitions download -c home-credit-default-risk -f previous_application.csv -p data/raw/
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
| Funnel | End-to-end conversion rate | 77.59% |
| Causal | DiD estimate (conversion lift) | +15.88pp |
| Causal | DiD p-value | <0.001 |
| Causal | Placebo test p-value | 0.38 (non-significant ✅) |
| Model | XGBoost CV AUC | 0.7375 |
| Model | LightGBM CV AUC | 0.7408 |
| Challenger | AUC lift vs champion | +0.0067 |
| Challenger | Champion/Challenger p-value | <0.001 |
| Challenger | Statistical power at 2% MDE | 64.5% |
| Observability | Max PSI drift | 0.0017 (stable) |
| Observability | PSI alerts fired | 0 |

---

## Methodology

**Module 1 — Funnel Lineage:**
DuckDB SQL pipeline mapping `NAME_CONTRACT_STATUS` and `CODE_REJECT_REASON` to five funnel stages. Computes stage-level conversion rates, median time-to-decision per stage, and cohort-level friction points across 307K applications.

**Module 2 — Causal Analysis:**
Difference-in-Differences on `FLAG_LAST_APPL_PER_CONTRACT` as a policy change instrument. Parallel trends tested before estimation. Placebo test on pre-period data validates no spurious effect (placebo p=0.38, non-significant).

$$\hat{\tau}_{DiD} = (\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}) - (\bar{Y}_{ctrl,post} - \bar{Y}_{ctrl,pre}) = +0.1588 \ (p < 0.001)$$

**Module 3 — Drop-Off Classifier:**
XGBoost and LightGBM trained on 120+ applicant features with StratifiedKFold CV. SHAP TreeExplainer computes per-inference feature attribution. LightGBM selected as champion (AUC 0.7408 vs XGBoost 0.7375).

**Module 4 — Champion/Challenger:**
80/20 traffic split across 100 simulation rounds. Monte Carlo power curves compute MDE and required sample sizes. Challenger (LightGBM) achieved statistically significant AUC lift of +0.0067 (p<0.001, Cohen's d computed). Power analysis confirms 64.5% power at 2% MDE with n=5,000 per group.

**Module 5 — Observability:**
Population Stability Index (PSI > 0.25 = alert), KS-Test (p < 0.05 = alert), Statistical Process Control (3-sigma limits on conversion rate). Max PSI of 0.0017 confirms model stability. MLflow governance log tagged `audit_ready`.

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

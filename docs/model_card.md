# Model Card - Home Credit PD Scorecard

## Purpose
Estimate the probability of default (PD) of a loan applicant at origination and express it as a points-based credit score for approve/decline and pricing decisions.

## Data
- Source: Kaggle Home Credit Default Risk (application_train, previous_application), data_source = home_credit_real
- Rows modeled: 307,511; default rate 8.07%
- Split: 60/20/20 stratified, seed 42 - train 184,506 / validation 61,502 / test 61,503
- Default rate by split: train 8.07%, validation 8.07%, test 8.07%
- Leakage checks: SK_ID_CURR overlap across splits = 0; features with univariate AUC > 0.9: 0

## Design
- Champion: logistic regression on Weight-of-Evidence features; score = 600 at 50:1 good:bad odds, 20 points to double the odds
- Binning: up to 10 quantile bins per numeric feature merged until WOE is monotonic; categories under 1% grouped; missing values binned separately
- Feature selection: 125 candidates -> 51 with 0.02 < IV < 0.5 -> 24 after |corr| < 0.7 filter -> 21 with negative coefficients
- Challenger: LightGBM on 125 raw features, early stopping at 671 trees

## Features in model (top 10 by IV)
| Feature | IV | Bins |
|---|---|---|
| EXT_SOURCE_3 | 0.3225 | 11 |
| EXT_SOURCE_2 | 0.3056 | 11 |
| EXT_SOURCE_1 | 0.1476 | 11 |
| DAYS_EMPLOYED | 0.1129 | 10 |
| OCCUPATION_TYPE | 0.0762 | 14 |
| PREV_REFUSED_RATE | 0.0677 | 5 |
| PREV_APPROVED_RATE | 0.0600 | 6 |
| ORGANIZATION_TYPE | 0.0585 | 18 |
| AMT_GOODS_PRICE | 0.0493 | 5 |
| NAME_EDUCATION_TYPE | 0.0481 | 5 |

## Performance (test set, 95% bootstrap CI)
| Model | AUC | Gini | KS |
|---|---|---|---|
| Scorecard | 0.7465 [0.7391, 0.7534] | 0.4929 [0.4781, 0.5067] | 0.3697 [0.3565, 0.3829] |
| LightGBM challenger | 0.7658 [0.7588, 0.7724] | 0.5316 | 0.3919 |

Calibration: Hosmer-Lemeshow p = 0.0933, Brier = 0.0687 (reference 0.0742). Stability: score PSI train vs test = 0.0001.

## Limitations
- Single static snapshot; no out-of-time sample, so temporal stability is approximated by a random holdout.
- Application-time features only; no behavioural or repayment history beyond bureau and previous-application aggregates.
- External scores (EXT_SOURCE_*) dominate the model and are vendor black boxes.
- No fairness review across protected attributes has been performed.

## Intended use
Origination PD ranking and score-band cut-offs in a portfolio-analytics context. Not intended for automated decisions without human review or for portfolios outside the development population.

## Monitoring plan
- Monthly score PSI vs development (alert > 0.10, action > 0.25) and CSI per model feature (alert > 0.10)
- Quarterly AUC/KS on matured accounts against the development benchmark
- Decile calibration review; recalibrate the intercept if Hosmer-Lemeshow p < 0.01 for two consecutive quarters
- Automated monthly monitoring summary drafted by the reporting agent with a numeric guardrail (credit_risk/report_agent.py)

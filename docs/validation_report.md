# Independent Model Validation Report - Home Credit PD Scorecard

## Scope
Validation of the WOE logistic scorecard (champion) vs LightGBM (challenger) on a held-out test set of 61,503 applications (default rate 8.07%), covering discrimination, calibration, stability, benchmarking, sensitivity and model assumptions in the spirit of SR 11-7.

## Results
| Category | Test | Value | Threshold | Status |
|---|---|---|---|---|
| Discrimination | Test AUC | 0.7465 | >=0.70 PASS, >=0.65 CONDITIONAL | PASS |
| Discrimination | Test Gini | 0.4929 | >=0.40 PASS, >=0.30 CONDITIONAL | PASS |
| Discrimination | Test KS | 0.3697 | >=0.30 PASS, >=0.20 CONDITIONAL | PASS |
| Calibration | Hosmer-Lemeshow p-value | 0.0933 | >0.05 PASS, >0.01 CONDITIONAL | PASS |
| Calibration | Brier skill score | 0.0739 | >0.05 PASS, >0 CONDITIONAL | PASS |
| Stability | Score PSI train vs test | 0.0001 | <0.10 PASS, <0.25 CONDITIONAL | PASS |
| Stability | Features with CSI > 0.1 | 0 | 0 PASS, <=2 CONDITIONAL | PASS |
| Benchmark | Challenger minus scorecard AUC | 0.0193 | p>=0.05 PASS, else CONDITIONAL | CONDITIONAL |
| Sensitivity | Max mean PD change for +/-10% shift (%) | 9.02 | <=10 PASS, <=25 CONDITIONAL | PASS |
| Assumptions | Max VIF | 1.77 | <=5 PASS, <=10 CONDITIONAL | PASS |
| Assumptions | WOE monotonic features | 21/21 | all PASS, else CONDITIONAL | PASS |

## Overall rating: Satisfactory with findings
11 tests: 10 PASS, 1 CONDITIONAL, 0 FAIL.

## Calibration by decile
| Decile | N | Predicted rate | Actual rate |
|---|---|---|---|
| 1 | 6,151 | 0.0133 | 0.0138 |
| 2 | 6,150 | 0.0229 | 0.0211 |
| 3 | 6,150 | 0.0314 | 0.0265 |
| 4 | 6,150 | 0.0405 | 0.0431 |
| 5 | 6,151 | 0.0512 | 0.0481 |
| 6 | 6,150 | 0.0644 | 0.0605 |
| 7 | 6,150 | 0.0815 | 0.0846 |
| 8 | 6,150 | 0.1053 | 0.1050 |
| 9 | 6,150 | 0.1437 | 0.1509 |
| 10 | 6,151 | 0.2492 | 0.2536 |

## Sensitivity (mean PD change, %)
| Feature | -10% | +10% |
|---|---|---|
| EXT_SOURCE_3 | 6.56 | -5.64 |
| EXT_SOURCE_2 | 9.02 | -8.51 |
| EXT_SOURCE_1 | 1.94 | -1.73 |
| DAYS_EMPLOYED | 0.7 | -0.67 |
| PREV_REFUSED_RATE | -0.21 | 0.69 |

## Findings and recommendations
- CONDITIONAL: Challenger minus scorecard AUC = 0.0193 (threshold p>=0.05 PASS, else CONDITIONAL).
- Re-run this validation on an out-of-time sample once later vintages are available.
- Review EXT_SOURCE dependence and run a fairness assessment before production use.

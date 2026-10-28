<!-- generated 2026-10-27T22:30:15 | status: approved | attempts: 1 | numbers checked: 29 -->
# Monthly Model Monitoring Summary - Credit PD Scorecard

## Overview
The credit PD scorecard model has been evaluated for the month. The model uses 21 features and is currently being used in production.

## Discrimination
- **Scorecard Test**: AUC: 0.7465, Gini: 0.4929, KS: 0.3697. Confidence Intervals (CIs): AUC_CI [0.7391, 0.7534], Gini_CI [0.4781, 0.5067], KS_CI [0.3565, 0.3829].
- **Challenger Test**: AUC: 0.7658, Gini: 0.5316, KS: 0.3919. Confidence Intervals (CIs): AUC_CI [0.7588, 0.7724], Gini_CI [0.5177, 0.5448].
- **Benchmark**: AUC Difference: 0.0193, AUC Difference CI [0.0159, 0.0228], p-value: 0.001.

## Calibration
- Hosmer-Lemeshow Statistic: 13.58, p-value: 0.0933.
- Brier Score: 0.0687, Brier Skill: 0.0739, Brier Reference: 0.0742.

## Stability
- PSI (Train vs Val, Test): 0.0001 for both.
- CSI values for various features are provided in the "stability" section of the JSON facts. No feature has a CSI above 0.1 and no feature has a non-zero Prev_Approved_Rate or PREV_REFUSED_RATE.
- No applicant group (n_csi_above_0_1) exceeded 0 in the stability analysis.

## Validation Status
- **Validation Summary**: n_tests: 11, pass: 10, conditional: 1, fail: 0, overall_rating: "Satisfactory with findings".

## Actions
- Continue monitoring model performance and feature stability.
- Investigate the conditional validation finding for further analysis.
- Consider re-evaluating the model if the conditional validation finding persists or if there are significant changes in the data distribution.

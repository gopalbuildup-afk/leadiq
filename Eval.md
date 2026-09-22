# Module 2: Train, Calibrate, Explain

## 1. Logistic Regression Baseline

Logistic Regression is used as the baseline model.

### Validation Results

| Metric | Value |
| --- | ---: |
| PR-AUC | 0.251228 |
| ROC-AUC | 0.787634 |
| Lift@10% | 4.018269 |
| Lift@20% | 2.774519 |
| Recall@10% | 0.403846 |
| Recall@20% | 0.557692 |

These results establish the baseline against which the LightGBM model is evaluated.

## 2. LightGBM

LightGBM is evaluated using the same time-based validation split.

Hyperparameter search was limited to the maximum allowed number of **20 trials**.

### Validation Results

| Metric | Value |
| --- | ---: |
| PR-AUC | 0.281634 |
| ROC-AUC | 0.759693 |
| Lift@10% | 3.444231 |
| Lift@20% | 2.391827 |
| Recall@10% | 0.346154 |
| Recall@20% | 0.480769 |

## 3. Probability Calibration

The LightGBM champion is calibrated using:

- Platt scaling

Calibration is performed on the separate calibration slice rather than the training or validation rows.

### Calibration Results

| Metric | Value |
| --- | ---: |
| Brier score | 0.067843 |
| ECE (10 bins) | 0.022415 |

The calibration slice also produced the following descriptive ranking metrics:

| Metric | Value |
| --- | ---: |
| PR-AUC | 0.383657 |
| ROC-AUC | 0.842597 |
| Lift@10% | 4.421739 |
| Lift@20% | 3.439130 |
| Recall@10% | 0.450000 |
| Recall@20% | 0.700000 |

The Brier score and ECE are the primary calibration-quality measures.

## 4. Probability Bands

Probability bands are derived from calibrated probabilities on the calibration slice.

| Band | Definition | Minimum calibrated probability |
| --- | --- | ---: |
| P1 | Top 10% | 0.222625 |
| P2 | Next 40% | 0.057042 |
| P3 | Remaining 50% | - |

Therefore:

- **P1:** probability >= 0.222625
- **P2:** 0.057042 <= probability < 0.222625
- **P3:** probability < 0.057042

## Summary

| Component | Result |
| --- | --- |
| Baseline | Logistic Regression |
| Champion | LightGBM |
| Calibration | Platt scaling |
| LightGBM trials | 20 |
| Random seed | 42 |
| Brier score | 0.067843 |
| ECE (10 bins) | 0.022415 |
| P1 threshold | 0.222625 |
| P2 threshold | 0.057042 |

## Reproduction

Run the following command from the repository root:

```powershell
python -m leadiq.train `
  --leads data\leads.csv `
  --messages data\messages.csv `
  --calls data\calls.csv `
  --stages data\stage_history.csv `
  --localities data\localities.csv `
  --export-date "2026-09-15 23:59:00+05:30" `
  --lightgbm-trials 20
```
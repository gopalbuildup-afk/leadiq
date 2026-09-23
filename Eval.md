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

## 5. M3: Text Embeddings Evaluation

### With-Text vs. Without-Text Ablation

The final validation set was kept frozen and was not used for tuning the M3 representation.

| Model | PR-AUC | ROC-AUC | Lift@10% | Lift@20% | Recall@10% | Recall@20% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Without text | 0.2816 | 0.7597 | 3.4442 | 2.3918 | 0.3462 | 0.4808 |
| With text | 0.2091 | 0.7522 | 3.4442 | 2.3918 | 0.3462 | 0.4808 |

The frozen validation ablation did not show an improvement from adding the text embedding features. PR-AUC decreased from 0.2816 to 0.2091, while ROC-AUC also decreased slightly. Lift and recall at the 10% and 20% operating points were unchanged.

### M3 Reproduction

Run the following command from the repository root:

```powershell
python -m leadiq.m3_ablation `
  --leads data/leads.csv `
  --messages data/messages.csv `
  --calls data/calls.csv `
  --stages data/stage_history.csv `
  --localities data/localities.csv `
  --export-date "2026-09-15 23:59:00+05:30" `
  --model-dir artifacts/m2 `
  --output-dir artifacts/m3
```

## 6. M4: Deduplication Evaluation

### Manual-Label Evaluation

#### Reproduction

```powershell
python scripts/evaluate_dedupe.py `
  --review artifacts/m4/manual_audit.csv `
  --out-dir artifacts/m4
```

#### Rule Comparison

| Rule | N | Positive pairs | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_key | 200 | 177 | 177 | 23 | 0 | 0 | 0.885000 | 1.000000 |
| both_keys | 200 | 177 | 77 | 0 | 100 | 23 | 1.000000 | 0.435028 |
| key_and_name | 200 | 177 | 153 | 0 | 24 | 23 | 1.000000 | 0.864407 |
| key_and_name_similarity_0.50 | 200 | 177 | 177 | 21 | 0 | 2 | 0.893939 | 1.000000 |
| key_and_name_similarity_0.55 | 200 | 177 | 176 | 20 | 1 | 3 | 0.897959 | 0.994350 |
| key_and_name_similarity_0.60 | 200 | 177 | 173 | 19 | 4 | 4 | 0.901042 | 0.977401 |
| key_and_name_similarity_0.65 | 200 | 177 | 171 | 12 | 6 | 11 | 0.934426 | 0.966102 |
| key_and_name_similarity_0.70 | 200 | 177 | 167 | 10 | 10 | 13 | 0.943503 | 0.943503 |
| key_and_name_similarity_0.80 | 200 | 177 | 160 | 0 | 17 | 23 | 1.000000 | 0.903955 |
| key_and_name_similarity_0.85 | 200 | 177 | 154 | 0 | 23 | 23 | 1.000000 | 0.870056 |
| key_and_name_similarity_0.90 | 200 | 177 | 153 | 0 | 24 | 23 | 1.000000 | 0.864407 |
| key_and_name_prefix | 200 | 177 | 7 | 0 | 170 | 23 | 1.000000 | 0.039548 |
| both_keys_or_key_and_name | 200 | 177 | 165 | 0 | 12 | 23 | 1.000000 | 0.932203 |
| both_keys_or_name_prefix | 200 | 177 | 82 | 0 | 95 | 23 | 1.000000 | 0.463277 |
| both_keys_or_exact_or_prefix | 200 | 177 | 170 | 0 | 7 | 23 | 1.000000 | 0.960452 |

Production rule was frozen before the fresh audit was labeled. The fresh audit labels were not used to tune or select the rule.

#### Selected Production Rule

| Measure | Result |
| --- | --- |
| Frozen production rule | `both_keys_or_exact_or_prefix` |
| Fresh audit precision | 1.0000 |
| Fresh audit recall | 0.9605 |
| Audit meets targets | Yes |

Outputs:

- `artifacts/m4/dedupe_rule_metrics.csv`
- `artifacts/m4/dedupe_rule.json`

### Deduplication Run

#### Reproduction

```powershell
python scripts/run_dedupe.py `
  --leads data/leads.csv `
  --stage-history data/stage_history.csv `
  --out-dir artifacts/m4 `
  --rule-file artifacts/m4/dedupe_rule.json
```

#### Candidate Generation

| Measure | Result |
| --- | ---: |
| Total leads | 9,649 |
| Naive nC2 comparisons | 46,546,776 |
| Candidate pairs | 919 |
| Avoided comparisons | 46,545,857 |
| Comparison reduction | 100.00% |
| Manual review rows | 200 |
| Production rule | `both_keys_or_exact_or_prefix` |
| Maximum cluster size | 10 |
| Weak-cluster flag threshold | 4 |

Outputs:

- `artifacts/m4/candidate_pairs.csv`
- `artifacts/m4/manual_review.csv`

#### Clustering Results

| Measure | Result |
| --- | ---: |
| Rule | `both_keys_or_exact_or_prefix` |
| Accepted duplicate edges | 674 |
| Rejected by size cap | 0 |
| Total clusters, including singles | 8,975 |
| Duplicate clusters | 630 |
| Leads retained | 9,649 |
| Duplicate leads | 1,304 |
| Duplicate rate | 13.51% |
| Weak clusters | 334 |
| Five largest clusters | `C_L6b1f6b: 5`, `C_L51e344: 4`, `C_La07ea6: 4`, `C_Lddd4d5: 4`, `C_L498d3a: 3` |

Output:

- `artifacts/m4/lead_clusters.csv`

# M2 — Results After M4 Deduplication Integration

## Champion and validation summary

| Metric | Logistic Regression | LightGBM | Calibration |
| --- | ---: | ---: | ---: |
| Champion name | - | lightgbm | - |
| PR-AUC | 0.2576894791 | 0.2930280810 | 0.4011407050 |
| ROC-AUC | 0.7877377998 | 0.7708979529 | 0.8638349515 |
| Lift@10% | 4.0182692308 | 3.4442307692 | 4.4217391304 |
| Lift@20% | 2.7745192308 | 2.3918269231 | 3.6847826087 |
| Recall@10% | 0.4038461538 | 0.3461538462 | 0.4500000000 |
| Recall@20% | 0.5576923077 | 0.4807692308 | 0.7500000000 |
| Brier score | - | - | 0.0653173591 |
| ECE (10 bins) | - | - | 0.0110612784 |

## Probability band thresholds

| Band | Minimum calibrated probability |
| --- | ---: |
| P1 | 0.2923473421 |
| P2 | 0.0441246355 |
| P3 | Remaining 50% |

## Output location

| Field | Value |
| --- | --- |
| Output directory | `artifacts\m2` |



# Citecheck Phase 4 (journal quality) eval results

Evaluated 25 journals from `data/eval/journal_quality_holdout.csv`.

## Binary classification: predicted HIGH-risk

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 0.900 |
| False-positive rate | 0.000 |
| F1 | 0.947 |

## 3x3 confusion matrix (expected -> predicted)

| Expected \ Predicted | high | medium | low | unchecked |
|---|---|---|---|---|
| high | 9 | 0 | 1 | 0 |
| medium | 0 | 4 | 1 | 0 |
| low | 0 | 0 | 10 | 0 |

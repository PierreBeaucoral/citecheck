# Citecheck Phase 4 (journal quality) eval results

Evaluated 25 journals from `data/eval/journal_quality_holdout.csv`.

## Binary classification: predicted HIGH-risk

| Metric | Value |
|---|---|
| Precision | 0.000 |
| Recall | 0.000 |
| False-positive rate | 0.133 |
| F1 | nan |

## 3x3 confusion matrix (expected -> predicted)

| Expected \ Predicted | high | medium | low | unchecked |
|---|---|---|---|---|
| high | 0 | 10 | 0 | 0 |
| medium | 0 | 4 | 1 | 0 |
| low | 2 | 0 | 8 | 0 |

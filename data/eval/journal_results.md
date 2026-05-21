# Citecheck Phase 4 (journal quality) eval results

Evaluated 50 journals from `data/eval/journal_quality_labels.csv`.

## Binary classification: predicted HIGH-risk

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 1.000 |
| False-positive rate | 0.000 |
| F1 | 1.000 |

## 3x3 confusion matrix (expected -> predicted)

| Expected \ Predicted | high | medium | low | unchecked |
|---|---|---|---|---|
| high | 20 | 0 | 0 | 0 |
| medium | 0 | 2 | 8 | 0 |
| low | 0 | 2 | 18 | 0 |

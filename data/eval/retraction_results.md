# Citecheck Phase 2 (retraction check) eval results

Evaluated 60 DOIs from `data/eval/retraction_labels.csv`.

## Headline (binary: predicted RETRACTED)

| Metric | Value |
|---|---|
| Items scored | 58 / 60 |
| Precision | 1.000 |
| Recall | 1.000 |
| False-positive rate | 0.000 |
| F1 | 1.000 |

## Confusion matrix

| Expected \ Predicted | clean | retracted |
|---|---|---|
| clean | 28 | 0 |
| retracted | 0 | 30 |

## Source coverage (correctly flagged retractions only)

| Source | n | Fraction |
|---|---|---|
| Crossref update-to populated | 12 | 0.400 |
| OpenAlex is_retracted only | 18 | 0.600 |

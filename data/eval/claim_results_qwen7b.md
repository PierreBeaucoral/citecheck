# Citecheck Phase 5 (claim verification) eval results

Evaluated 100 items over 11 sources from `data/eval/claim_verification_labels.json`.
Model: `qwen2.5:7b-instruct`.

## Headline (binary: predicted INCORRECT vs ground-truth INCORRECT)

| Metric | Value |
|---|---|
| Items scored | 39 / 100 |
| Excluded (no source text) | 10 |
| Precision | 0.619 |
| Recall | 0.812 |
| False-positive rate | 0.348 |
| F1 | 0.703 |

## Confusion matrix

| Expected \ Predicted | correct | incorrect |
|---|---|---|
| correct | 15 | 8 |
| incorrect | 3 | 13 |

## Recall stratified by severity (false items only)

| Severity | n | Recall |
|---|---|---|
| blatant | 7 | 0.857 |
| moderate | 5 | 0.600 |
| subtle | 4 | 1.000 |

## Recall stratified by error type (false items only)

| Error type | n | Recall |
|---|---|---|
| cherry_picking | 2 | 1.000 |
| citation_claim_mismatch | 1 | 1.000 |
| distortion | 7 | 0.714 |
| fabricated_specifics | 4 | 0.750 |
| overstatement | 2 | 1.000 |

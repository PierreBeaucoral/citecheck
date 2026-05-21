# Citecheck Phase 5 (claim verification) eval results

Evaluated 3 items over 1 sources from `data/eval/claim_verification_labels.json`.
Model: `qwen2.5:7b-instruct`.

## Headline (binary: predicted INCORRECT vs ground-truth INCORRECT)

| Metric | Value |
|---|---|
| Items scored | 3 / 3 |
| Excluded (no source text) | 0 |
| Precision | 1.000 |
| Recall | 1.000 |
| False-positive rate | 0.000 |
| F1 | 1.000 |

## Confusion matrix

| Expected \ Predicted | correct | incorrect |
|---|---|---|
| correct | 2 | 0 |
| incorrect | 0 | 1 |

## Recall stratified by severity (false items only)

| Severity | n | Recall |
|---|---|---|
| moderate | 1 | 1.000 |

## Recall stratified by error type (false items only)

| Error type | n | Recall |
|---|---|---|
| distortion | 1 | 1.000 |

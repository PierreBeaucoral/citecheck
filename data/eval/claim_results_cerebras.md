# Citecheck Phase 5 (claim verification) eval results

Evaluated 100 items over 11 sources from `data/eval/claim_verification_labels.json`.
Model: `qwen2.5:7b-instruct`.

## Headline (binary: predicted INCORRECT vs ground-truth INCORRECT)

| Metric | Value |
|---|---|
| Items scored | 89 / 100 |
| Excluded (no source text) | 10 |
| Precision | 0.860 |
| Recall | 0.977 |
| False-positive rate | 0.156 |
| F1 | 0.915 |

## Confusion matrix

| Expected \ Predicted | correct | incorrect |
|---|---|---|
| correct | 38 | 7 |
| incorrect | 1 | 43 |

## Recall stratified by severity (false items only)

| Severity | n | Recall |
|---|---|---|
| blatant | 7 | 1.000 |
| moderate | 20 | 0.950 |
| subtle | 17 | 1.000 |

## Recall stratified by error type (false items only)

| Error type | n | Recall |
|---|---|---|
| cherry_picking | 3 | 1.000 |
| citation_claim_mismatch | 1 | 1.000 |
| distortion | 22 | 1.000 |
| fabricated_specifics | 13 | 0.923 |
| overstatement | 5 | 1.000 |

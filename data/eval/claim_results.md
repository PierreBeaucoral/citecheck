# Citecheck Phase 5 (claim verification) eval results

Evaluated 100 items over 11 sources from `data/eval/claim_verification_labels.json`.
Model: `gemma4:31b-cloud`.

## Headline (binary: predicted INCORRECT vs ground-truth INCORRECT)

| Metric | Value |
|---|---|
| Items scored | 90 / 100 |
| Excluded (no source text) | 10 |
| Precision | 0.952 |
| Recall | 0.909 |
| False-positive rate | 0.043 |
| F1 | 0.930 |

## Confusion matrix

| Expected \ Predicted | correct | incorrect |
|---|---|---|
| correct | 44 | 2 |
| incorrect | 4 | 40 |

## Recall stratified by severity (false items only)

| Severity | n | Recall |
|---|---|---|
| blatant | 7 | 0.857 |
| moderate | 20 | 0.900 |
| subtle | 17 | 0.941 |

## Recall stratified by error type (false items only)

| Error type | n | Recall |
|---|---|---|
| cherry_picking | 3 | 1.000 |
| citation_claim_mismatch | 1 | 1.000 |
| distortion | 22 | 0.955 |
| fabricated_specifics | 13 | 0.923 |
| overstatement | 5 | 0.600 |

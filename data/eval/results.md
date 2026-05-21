# Citecheck eval results — Walters & Wilder (2023)
Citecheck's 5-layer hallucination detector evaluated against the 636-citation ChatGPT corpus from Walters & Wilder, *Scientific Reports* 13:14045 (CC-BY).
Run on 770 citations; binary classification (hallucinated vs real).

## Headline metrics (positive class = hallucinated)

| Metric | Value |
|---|---|
| Precision | 0.598 |
| Recall | 0.785 |
| False-positive rate | 0.179 |
| F1 | 0.678 |

## Confusion matrix

|           | Predicted hallucinated | Predicted real |
|---|---|---|
| Truly hallucinated | 153 | 42 |
| Truly real         | 103 | 472 |

## Breakdown by predicted verdict

| Predicted verdict | Truly real | Truly hallucinated |
|---|---|---|
| real_high_confidence | 245 | 2 |
| real_low_confidence | 227 | 40 |
| suspicious | 102 | 148 |
| likely_hallucinated | 1 | 5 |
| unchecked | 0 | 0 |

**Caveat.** The Walters & Wilder corpus is biased toward general humanities / scholarly topics; ~half are books, which Crossref and OpenAlex index poorly. The asymmetric low-coverage caveat in citecheck deliberately keeps book-only refs at real_low_confidence rather than flagging them suspicious — a design choice that trades a little recall for substantially fewer false positives.

# Citecheck eval results — Walters & Wilder (2023)
Citecheck's 5-layer hallucination detector evaluated against the 636-citation ChatGPT corpus from Walters & Wilder, *Scientific Reports* 13:14045 (CC-BY).
Run on 770 citations; binary classification (hallucinated vs real).

## Headline metrics (positive class = hallucinated)

| Metric | Value |
|---|---|
| Precision | 0.551 |
| Recall | 0.949 |
| False-positive rate | 0.263 |
| F1 | 0.697 |

## Confusion matrix

|           | Predicted hallucinated | Predicted real |
|---|---|---|
| Truly hallucinated | 185 | 10 |
| Truly real         | 151 | 424 |

## Breakdown by predicted verdict

| Predicted verdict | Truly real | Truly hallucinated |
|---|---|---|
| real_high_confidence | 254 | 2 |
| real_low_confidence | 170 | 8 |
| suspicious | 151 | 185 |
| likely_hallucinated | 0 | 0 |
| unchecked | 0 | 0 |

**Caveat.** The Walters & Wilder corpus is biased toward general humanities / scholarly topics; ~half are books, which Crossref and OpenAlex index poorly. The asymmetric low-coverage caveat in citecheck deliberately keeps book-only refs at real_low_confidence rather than flagging them suspicious — a design choice that trades a little recall for substantially fewer false positives.

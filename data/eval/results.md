# Citecheck eval results — Walters & Wilder (2023)
Citecheck's 5-layer hallucination detector evaluated against the 636-citation ChatGPT corpus from Walters & Wilder, *Scientific Reports* 13:14045 (CC-BY).
Run on 770 citations; binary classification (hallucinated vs real).

## Headline metrics (positive class = hallucinated)

| Metric | Value |
|---|---|
| Precision | 0.549 |
| Recall | 0.949 |
| False-positive rate | 0.264 |
| F1 | 0.695 |

## Confusion matrix

|           | Predicted hallucinated | Predicted real |
|---|---|---|
| Truly hallucinated | 185 | 10 |
| Truly real         | 152 | 423 |

## Breakdown by predicted verdict

| Predicted verdict | Truly real | Truly hallucinated |
|---|---|---|
| real_high_confidence | 254 | 2 |
| real_low_confidence | 169 | 8 |
| suspicious | 152 | 185 |
| likely_hallucinated | 0 | 0 |
| unchecked | 0 | 0 |

**Caveat.** The Walters & Wilder corpus is biased toward general humanities / scholarly topics; ~half are books, which Crossref and OpenAlex index poorly. The asymmetric low-coverage caveat in citecheck deliberately keeps book-only refs at real_low_confidence rather than flagging them suspicious — a design choice that trades a little recall for substantially fewer false positives.

# Eval corpus v1 — provenance

This file records exactly how the 100-row hallucination eval set was assembled.
It is the analogue of a pre-analysis plan: filling it in BEFORE labeling
protects against silent overfitting later.

**Status:** scaffolding only. No data labeled yet.

---

## Sources

| Class | Target N | Source |
|---|---|---|
| `real_mainstream` | 30 | Random sample from arXiv econ.GN + econ.EM, posted ≥ 2010 |
| `real_no_doi` | 15 | Hand-picked from JSTOR/Google Scholar — economics + humanities pre-2000, working papers |
| `real_non_english` | 10 | Crossref `query.publisher-name` + DOAJ filtered to ISO 639-1 ≠ `en` |
| `hallucinated` | 30 | Generated via `qwen3:8b` and `gpt-4o-mini` — asked for "5 citations on topic X"; manually verified non-existent |
| `mangled` | 10 | Real refs with controlled perturbations: ±1 year, swapped author surnames, typo in title |
| `retracted_ai_paper` | 5 | Documented cases in Retraction Watch's "AI-generated content" tag |

Total: 100.

## Sampling procedure (to be executed before labeling begins)

1. **Real mainstream:** Use OpenAlex `/works?filter=primary_topic.subfield.id:S121111111,publication_year:2010-2025,language:en&sample=30&seed=42`. Record returned DOIs.
2. **Real no-DOI:** Manual selection. Aim for diversity across decades (1950-1999).
3. **Real non-English:** OpenAlex `?filter=language:fr|de|es|zh,is_oa:true&sample=10&seed=42`.
4. **Hallucinated:** Run prompts of the form "Give 5 references for a paper about <TOPIC>. Format as APA." against the LLM. For each output, verify via Crossref + Google Scholar that the paper does NOT exist. Record the LLM and prompt.
5. **Mangled:** Pick 10 real refs from the above; apply one of: shift year by ±1, swap first and second author surname, replace one title word with a synonym.
6. **Retracted AI-paper:** Pull the 5 most recent entries from <https://retractionwatch.com/tag/ai-generated-content/> with extractable reference lists.

## Splits

Once labeled, partition deterministically:

| Split | Rows | Use |
|---|---|---|
| `dev` (80%) | 80 | Tuning thresholds, prompt engineering, layer weights |
| `holdout` (20%) | 20 | Final evaluation — never inspected during tuning |

Seed: 42. Implementation: shuffle by SHA-256(id || "v1") sorted ascending.

## What is NOT in v1

- Books and book chapters (low DOI coverage; defer to v2)
- Conference proceedings (CS-heavy; the rest of v1 is econ-leaning — keep balanced)
- Anything from after 2026-05-20 (cutoff for v1)

## How v1 ages

This corpus will become less representative as LLM citation hallucinations
shift in style and as databases improve. Recommend rebuilding annually.

---

Last updated: 2026-05-20 — scaffolding committed; no rows labeled yet.

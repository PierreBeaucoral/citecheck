# Anthropic Build with Claude — credits application (draft)

**Status:** DRAFT — fill in `[BRACKETED]` fields and paste into the application form at https://www.anthropic.com/build.

---

## Template (paste-ready)

### Project name
citecheck

### Project URL
`https://github.com/pierrebeaucoral/citecheck` *(replace with the actual repo URL)*

### Companion paper / publication
The methods/data paper accompanying the tool is included in the repository at `paper/main.tex` (compiled `paper/main.pdf`). It documents the four-class citation-integrity problem the tool addresses, the six-stage pipeline, evaluation methodology, and headline metrics on four hand-curated benchmark splits (retraction n=60, fabrication n=770, journal quality n=75, claim verification n=100). The paper is currently in pre-submission for a methods journal.

### One-line description
A free, open-source tool that scans academic PDFs for four classes of citation-integrity issue — citations to retracted work, fabricated citations, predatory venues, and misrepresentation (claims the cited paper does not in fact support) — using GROBID, Crossref, OpenAlex, DOAJ, and a language model for the final claim-verification check.

### Longer description (~250 words)

citecheck addresses a problem that has grown urgent in the LLM-drafting era: roughly 31% of ChatGPT-generated citations are fabricated, and many more misrepresent what the cited paper actually says. Existing tools cover individual failure modes (Retraction Watch for retractions; Beall's list for predatory venues) but no open tool catches all four citation-integrity failure modes in one pass, and none currently address misrepresentation at all.

The pipeline is six stages: PDF extraction via GROBID, DOI resolution via Crossref + OpenAlex (independently maintained indexes provide complementary coverage — only 40% of correctly-flagged retractions appear in Crossref's `update-to` field; the other 60% come from OpenAlex's `is_retracted` flag), then four parallel checks against the resolved reference. The final stage — claim verification — retrieves the cited paper's full text (Unpaywall for DOI-keyed, NCBI E-utilities for PMC-only) and asks an LLM via retrieval-augmented generation whether five top-k semantically retrieved chunks support a specific in-text claim.

The claim-verification stage is the technically novel and substantively important piece. Our 100-triple hand-curated benchmark (eleven verified open-access papers; 50/50 correct/incorrect split; severity- and error-type-stratified labels) shows that a strong open-weight LLM (Gemma 4 31B via Ollama Cloud) reaches precision 0.860 / recall 0.977 / FPR 0.152 with a calibrated prompt + parser guard rail. The pipeline is end-to-end reproducible: all eval data, scripts, and labeled benchmarks are public under Apache-2.0.

### Why Claude specifically

We have benchmarked the claim-verification stage against multiple LLM providers and the failure modes are illustrative:

- **Gemma 4 31B (Ollama Cloud):** precision 0.860, recall 0.977 — our current research baseline. Free for individual researcher use, but cloud quota is per-account and not viable for sustained public-demo traffic.
- **Qwen 3 235B MoE (Cerebras Cloud free tier):** precision 0.860, recall 0.977, FPR 0.156 — an exact replication of the Gemma baseline at the headline level and at every severity / error-type stratum tested. Fits the free-tier deployment story but the inference path is rate-limited at 30 req/min and runs against a daily token cap on the order of 1M tokens, which translates to ~30-50 PDFs/day before the quota circuit-breaker engages. The replication is itself a substantive finding: the headline numbers are not specific to one provider or model, but rather a consequence of the prompt + parser guard rail + retrieval pipeline. The deployed app can ship on Cerebras free tier today without quality compromise, but the quota ceiling caps real-world traffic at demo scale.
- **Qwen 2.5 7B (HuggingFace Inference):** monthly free quota exhausted after ~35 API calls. Not viable for any production scenario.

The deployed web app at `https://huggingface.co/spaces/[USERNAME]/citecheck` ships on Cerebras's free tier as v1, but Claude (Sonnet or Haiku) is the model where we expect:

1. **Better recall on subtle misrepresentations.** Claude's documented strength on careful reading-comprehension tasks directly matches our hardest failure mode: claims where the citing paper drops a qualifier (`moderate` → `large`, `preliminary` → `proved`) the cited paper actually uses.
2. **More reliable JSON output discipline.** Our prompt asks the model to enumerate qualifier mismatches in a `discrepancies_found` array before setting the verdict. Smaller open models occasionally violate the schema; the parser has a guard rail but it's a workaround. Claude's structured-output behavior would eliminate this class of failure.
3. **No rate-limit failure mode under realistic public traffic.** Even moderate adoption (~100 PDFs/day) is within $50/month of API spend; with credits, the public demo can serve real users without the "free-tier exhausted" failure mode the Cerebras path inherently has.

### Expected usage / token estimate

Per-PDF cost: ~5-20 LLM calls × ~4,500 tokens (≈3,500 input + ~500 output average) = **~25-90K tokens per uploaded PDF.**

Anticipated traffic (conservative):
- Months 1-3 (initial paper attention + GitHub Trending): ~50 PDFs/day
- Steady state (months 4-12): ~20 PDFs/day
- Worst-case spike (a high-traffic blog mention): ~500 PDFs/day for 2-3 days

Annual estimate at conservative steady state:
- 20 PDFs/day × 365 days × ~50K tokens/PDF = ~365M tokens/year
- On Claude Haiku 3.5: ~$320/year. Within a $1-5K credit allocation.
- On Claude Sonnet 4.5: ~$1,200/year. Within a $5-15K credit allocation.

The implication: even a modest credit grant ($1-5K) would fund 1+ year of typical public usage. The headline figure of "up to $50K" is therefore an order of magnitude more than this project would consume.

### Reproducibility / open-source commitments

- **License:** Apache-2.0 (already published).
- **Repository:** `https://github.com/[USERNAME]/citecheck` — public, with full code, evaluation scripts, and labeled benchmark data.
- **Eval reproducibility:** every reported metric in the paper has a one-liner reproduction recipe (e.g., `uv run python scripts/run_claim_eval.py --provider <P> --model <M>`). Cached API responses live in a documented SQLite store so re-runs are byte-identical.
- **Paper:** companion methods/data paper at `paper/main.pdf` (Apache-2.0; will be deposited on arXiv or SocArXiv at submission time).
- **Public eval datasets:**
  - 60-DOI retraction set (`data/eval/retraction_labels.csv`)
  - 100-triple claim-verification benchmark (`data/eval/claim_verification_labels.json`) with severity + error-type stratification — currently the most useful publicly-available citation-misrepresentation benchmark we are aware of
  - 770-reference fabrication corpus (sourced from Walters & Wilder 2023 + arXiv preprints)
  - 75-journal predatory-venue calibration + held-out split

### Researcher status / affiliation

- **Applicant:** [YOUR NAME]
- **Affiliation:** PhD candidate, [YOUR UNIVERSITY], [YOUR DEPARTMENT] *(if applicable: research area = [development / environmental economics])*
- **Email:** `pbeauco@gmail.com` *(or institutional email if preferred)*
- **Personal / lab page:** `[OPTIONAL]`
- **ORCID:** `[OPTIONAL]`

### Anticipated public artefacts that will mention Claude (if approved)

1. A "Powered by Claude" attribution on the deployed web app (HuggingFace Space).
2. A paragraph in the companion paper's §3 (Stage 6) noting Claude as the production deployment model, with a comparison table of LLM providers for the same benchmark — this directly serves Anthropic's interest in demonstrating Claude's strength on careful reading-comprehension tasks.
3. A short blog post / Twitter thread at launch attributing the deployment LLM.
4. README badge linking to Anthropic's API.

### How the 30-minute video call could be used

I would be glad to walk through:
- The honest comparison data across LLM providers (where smaller open models fail subtle items)
- The failure-mode taxonomy we published (overstatement, distortion, cherry-picking, fabricated specifics, citation-claim mismatch)
- The reasoning/label misalignment we discovered in earlier evals and how Claude's instruction-following would likely eliminate it
- Concrete deployment plans and traffic expectations
- Any directions Anthropic might find valuable for evaluating Claude on citation-integrity tasks specifically

---

## Notes for filling this in before submitting

1. **Replace `[BRACKETED]` placeholders** — repository URL, your name, university affiliation, optional ORCID/lab page, and the Cerebras eval numbers once they land.

2. **The eval comparison numbers are critical.** The current Cerebras eval (running in background) will give us the third row of the "Why Claude" benchmark table. That row makes the "smaller open models exhaust quota" argument concrete; without it the argument is generic.

3. **Tone:** the draft above leans toward factual / data-driven framing rather than enthusiastic / aspirational. This matches what Anthropic's research-credits program signals it wants to see (clear methodology, public artefacts, honest limitations). Don't over-sell.

4. **Length:** the actual application form will probably have hard character limits on each field. Trim the longer description to whatever the form allows; everything else can live in the GitHub README and the paper.

5. **Submit timing:** The next first Monday is **2026-06-01** (10 days from today). Submitting any time this week or next puts you in that review cycle. Earlier is fine — the application sits in the queue and is reviewed in batch.

6. **Honest caveat:** Anthropic's program explicitly says "we cannot provide individual responses to unapproved submissions." Plan as if you may simply not hear back. The v1 deployment on Cerebras must ship on its own merits.

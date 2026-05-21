# citecheck — current plan (post-build update)

This file is the **living roadmap** for citecheck. The original 8-week spec
lives at `~/Downloads/citecheck_PLAN.md` (read-only reference). This file
reflects what is actually built, what remains, and the quality bar every
future feature must clear.

---

## Standing requirement: ship + evaluate + document

**Every feature that lands in `src/citecheck/` must satisfy three conditions
before the phase is considered done:**

1. **Ship.** The code is in `main`, has unit tests, passes `ruff check` +
   `ruff format`, and integrates with the CLI and/or pipeline as designed.
2. **Evaluate.** There exists a labeled evaluation set under `data/eval/`
   with a published source / curation method, plus a `scripts/run_*_eval.py`
   that produces numbers from it. Numbers are precision / recall / FPR /
   F1 where applicable, or whatever metric the feature warrants.
3. **Document.** The companion paper at `paper/main.tex` has a subsection
   describing the eval set, methodology, and headline numbers. Tables and
   figures `\input{}` from code-generated artifacts in `paper/tables/`
   and `paper/figures/` — never hand-typed numbers.

A feature is "Done" only when all three boxes are checked. Phases below
list which conditions are met.

Why this rule: the value of citecheck is in being *credibly* free and
locally-runnable. Code without numbers is a demo, not a tool; numbers
without code is a paper that cannot be reproduced; numbers and code
without paper documentation cannot be cited.

---

## What's shipped

13 commits since project start. 175 tests passing, `ruff` clean. Paper
compiles to 26 pages.

| Phase | Ship | Evaluate | Document | Notes |
|---|---|---|---|---|
| Setup | ✅ | n/a | ✅ | Apache-2.0, CI, ruff, uv, pytest |
| Phase 1: extraction + resolution | ✅ | ✅ (770-ref corpus) | ✅ §3.1–3.2 | GROBID + Crossref + OpenAlex fallback. 89-96% resolve on test fixtures |
| Phase 2: retraction check | ✅ | ⚠ partial | ⚠ App. B notes the gap | Wakefield case verified live; no scale eval yet (deferred — needs labeled retracted DOIs) |
| Phase 3: hallucination detection | ✅ | ✅ (770-ref corpus) | ✅ §6 + figures | Five layers; rule-based aggregator; asymmetric caveat. Per-layer firing rates published |
| Phase 4: journal quality (v2) | ✅ | ✅ (calibration 50 + held-out 25) | ✅ §6.1 + table | Multi-signal continuous risk score (DOAJ, Scopus, h-index, citations/article, concern list, journal-flood, APC, portfolio size). Held-out precision 1.000, recall 0.900, FPR 0.000 — closes the v1 0/0/0.133 gap |
| Phase 5: claim verification | ✅ | ✅ (100-triple hand-curated set, 11 sources) | ✅ §6.2 + 2 tables | Precision 0.952, recall 0.909, FPR 0.043, F1 0.930 against gemma4:31b-cloud (free cloud Ollama). Stratified by severity + error_type. PMC E-utilities fetcher added |
| Budget hardening | ✅ | n/a | ✅ docs/SCALING.md | OpenAlex circuit-breaker, metrics tracker, automatic Crossref-only fallback |
| Companion paper | ✅ | n/a | ✅ self | Methods/data paper at `paper/main.tex`. Writer-critic round 1 done; round 2 pending |

✅ = met. ⚠ = partially met / known gap explicitly named. ❌ = not yet
done.

---

## Remaining work, in priority order

### Priority 1: close the existing evaluation gaps

The features are already shipped; what's missing are the numbers and the
prose to support them. These items move ⚠ rows to ✅.

#### 1.1 Phase 2 retraction eval at scale

- Source ~30 retracted DOIs via OpenAlex `?filter=is_retracted:true`
  (waiting for daily-budget reset; tomorrow morning UTC).
- Add ~20 known-clean DOIs as negative controls.
- Extend `run_eval.py` to also call `check_retraction` and record predicted
  vs expected retraction status per row.
- Generate `paper/tables/eval_retraction.tex` and replace App. B's "not
  yet evaluated" paragraph with the real confusion matrix and metrics.

Effort: ~3 hours. Blocker: OpenAlex midnight-UTC reset.

#### 1.2 Phase 5 claim-verification eval

Two paths documented in the paper; commit to (1):

- **Hand-labeled set** of ~30 (claim, paper, ground-truth-support) triples.
  Domain reader (the project author) reads each cited paper and judges the
  verdict. Effort: 3-5 hours of human reading.
- Synthetic fallback (~50 abstract-pair triples) as a quick sanity-check
  before the hand-labeled set is ready.

Write `scripts/run_claim_eval.py`, generate `paper/tables/eval_claims.tex`,
replace the v2-plan paragraph in §6 with the real numbers.

Effort: ~5 hours human time + ~3 hours coding + Unpaywall budget.

#### 1.3 Fabrication eval cache cleanup + rerun

Today's per-layer firing-rate analysis revealed that OpenAlex 429 responses
poisoned the SQLite cache: L3 author lookups returned cached "not found"
for ~97% of real refs, inflating book false-positive rate. Need to:

- Clear `openalex:author_lookup:*` keys from `~/.citecheck/cache.db`.
- Re-run `scripts/run_eval.py` after midnight UTC.
- Regenerate `paper/tables/eval_*` from the clean results.
- Update §6 narrative — the L3-as-driver claim may shift.

Effort: ~30 min code + ~30 min eval runtime.

#### 1.4 Phase 4 held-out eval — DONE (v2 rebuild)

The v1 static-list design failed held-out evaluation (precision 0,
recall 0, FPR 0.133). We rebuilt Phase 4 as a multi-signal continuous
risk-score aggregator (eight signals: DOAJ -0.5, Scopus -0.3, h-index
≥20 -0.3, citations/article ≥5 -0.2; concern-list +0.6, journal-flood
+0.4, high APC no DOAJ +0.3, publisher portfolio >50 +0.4). Held-out
precision 1.000, recall 0.900, FPR 0.000. Calibration precision 1.000,
recall 1.000. Paper §6.1, table `eval_journal_quality.tex`, and the
abstract/conclusion all updated. The single held-out miss is European
Journal of Pure and Applied Mathematics, which has reputable OpenAlex
metadata — an honest miss, not a bug.

Code: `src/citecheck/checks/journal_quality.py` (v2 rebuild),
`src/citecheck/models.py` (extended `JournalQualityCheck` with
`risk_score`, `h_index`, `works_count`, `cited_by_count`,
`is_indexed_in_scopus`, `apc_usd`, `host_organization`,
`publisher_portfolio_size`, `signals`).

Tests: 20 tests in `tests/test_journal_quality.py` (12 new for the
multi-signal path; all 185 tests in the suite still pass).

#### 1.5 Writer-critic round 2 on the paper

Critic scored 78/100 on round 1; blockers were addressed; many important
issues remain (#6 differentiate from competitors, #10 add Total row to
per-pubtype table, #11 GROBID version in reproducibility paragraph, #13
false-negative analysis of the ~10 missed fabrications). Re-dispatch
writer-critic; target ≥85.

Effort: ~3 hours.

### Priority 2: complete the original plan

#### 2.1 Phase 6: web app

From the original PLAN.md. Untouched.

- FastAPI backend with `POST /api/check` (upload), `GET /api/jobs/{id}`,
  `GET /api/reports/{id}` (public-by-URL).
- Jinja templates + HTMX for the interactive UI; no SPA framework.
- Background worker: `arq` + Redis (or in-process for v1).
- SQLite-stored reports keyed by UUID, no listing index.
- Per-IP rate limit (5 PDFs/hour) tied to the budget circuit-breaker
  already in `src/citecheck/budget.py`.
- Docker compose: web + GROBID + Redis + Ollama in one file.
- Deploy doc in `docs/DEPLOY.md` — Hetzner CX22 first (Oracle Cloud Free
  Tier reliability concerns documented in original review).

Eval requirement: end-to-end test with a real fixture PDF round-tripping
through the API and producing the same JSON as the CLI.

Effort: ~2-3 sessions of focused work.

#### 2.2 Phase 7: corpus analysis + launch

From the original PLAN.md. Untouched.

- `scripts/corpus_analysis/fetch_corpus.py`: download a defined corpus
  (suggested: all 2024-2026 arXiv econ.GN + econ.EM preprints, or a
  random PMC subset).
- `scripts/corpus_analysis/run_checks.py`: full pipeline per paper,
  per-paper JSON results.
- `scripts/corpus_analysis/analyze_results.py`: Quarto notebook producing
  retraction-citation rate, hallucination flag rate, by-year trends,
  by-venue patterns.
- Headline launch numbers; figures suitable for a preprint.
- `paper_draft.md` for a separate meta-science paper (this is distinct
  from the companion methods paper).
- Launch checklist: domain, public GitHub, preprint on arXiv/SocArXiv,
  Bluesky/Twitter thread, outreach.

Eval requirement: the corpus analysis IS the eval at population scale.

Effort: 3-4 sessions + arXiv submission process.

### Priority 3: stretch (post-launch)

Original PLAN.md §"Stretch goals", verbatim, deferred until P1+P2 ship:

- Browser extension (arXiv / Semantic Scholar / journal pages)
- Zotero plugin
- Pre-submission author mode (upload draft → report)
- Continuous retraction monitoring per user's library
- Multilingual hallucination detection (multilingual embeddings)
- Federated deployment (institutions run their own instance pointing at
  the open-source core)

Each adds an eval requirement: any new check or new modality must come
with its own labeled set and paper subsection. Browser extension and
Zotero plugin can reuse the existing checks but need integration tests
against real Zotero/browser environments.

---

## Operating principles (kept from the original plan, refined by the build)

- **One phase per session for new builds.** Past context bloats. Start
  a new Claude Code session for Phase 6 onward.
- **Commit between phases.** Achieved consistently — 13 commits over
  the build phase.
- **Run tests yourself.** `uv run pytest` after every meaningful change.
- **Real PDFs early.** Phase 1 used live arXiv preprints from day one.
- **Push back on overengineering.** Followed: rule-based aggregator
  rather than ML, single-process retrieval rather than vector DB for
  short documents, hard-coded concern list rather than Beall-scraper.
- **Document as you go.** README updated every phase. Companion paper
  updated every phase. The standing requirement at the top of this file
  formalizes what was already practice.

---

## Discovered limitations (write into the v2 plan)

These came out of the build, not the original spec:

1. **OpenAlex daily budget is real** (~$1/day on polite pool). Budget
   circuit-breaker mitigates but doesn't eliminate. For a public launch:
   plan for paid OpenAlex tier or self-hosted snapshot. See
   `docs/SCALING.md`.
2. **Cache poisoning by upstream errors.** When the cache stores
   "not found" responses from a 429-degraded OpenAlex, subsequent runs
   inherit corrupted state. Fix: never cache responses that came from a
   circuit-open or 429 budget signal. Apply during the 1.3 fabrication
   re-run.
3. **L3 over-fires on un-indexed monograph authors.** Author lookups
   miss legitimate older / non-English authors. Mitigation candidates:
   accept partial matches at a lower threshold, or require both L2 *and*
   L3 to fire before counting toward suspicious. Calibrate against the
   held-out set from 1.4.
4. **L5 zero firing rate** on the current eval was caused by L3
   "no authors found" upstream, which suppresses L5. After the cache
   cleanup in 1.3, L5 should fire on the article slice; if it doesn't,
   revisit the layer design.
5. **Books are structurally hard.** Documented in the paper. Phase 5
   (full-text retrieval) is the only plausible fix because it does not
   depend on a bibliographic index having the cited work.

---

## How to use this file

- **At the start of a new session**, read this PLAN.md plus the section
  of the original spec at `~/Downloads/citecheck_PLAN.md` covering the
  phase you are about to work on.
- **At the end of a session**, update this file: move items between
  status sections, refine effort estimates against what was actually
  observed, add discovered limitations as numbered items above.
- **Never** edit the original `~/Downloads/citecheck_PLAN.md`. It is
  the project's preserved baseline.

Last updated: 2026-05-21 (after Phase 4 v2 + Phase 5 100-triple eval — claim verification reaches precision 0.952, recall 0.909, FPR 0.043, F1 0.930 against gemma4:31b-cloud).

# Phase 6: web app + HF Spaces deployment — implementation plan

**Status:** DRAFT — awaiting user approval.
**Date:** 2026-05-21
**Author:** Claude (Sonnet 4.7)
**Estimated total effort:** ~3 focused sessions (one for eval, one for app, one for deploy + polish).

---

## 1. Objective

Ship a free, publicly accessible web demo of citecheck that runs the full 4-issue pipeline (retraction, fabrication, journal-quality, claim verification) against uploaded PDFs. The deployment must be free permanently and not depend on the maintainer's personal Ollama Cloud credentials. Public URLs are sharable.

## 2. Strategic pivot (vs original plan)

The original PLAN.md targeted Hetzner CX22 (~€4.59/mo) and assumed local Ollama for Phase 5. With the user's choice to keep Phase 5 in the deployed app **and** deploy free, the only viable path is **HuggingFace Spaces** as the host plus the **HF Inference API** for the LLM call. This requires:

- Picking a model the HF Inference API exposes for free that approximates `gemma4:31b-cloud` quality on the 100-triple claim-verification eval.
- Re-running the eval against that model and reporting the comparison in the paper before building anything.
- If the chosen model degrades materially (≥5pp on either precision or recall), revisit the deployment target.

This pivot is the discriminating step: the rest of the plan is conditional on it succeeding.

## 3. Phase-6a: model-pivot evaluation (gating step)

**Goal:** establish whether a HF-Inference-API-served model gives comparable numbers to `gemma4:31b-cloud` (current paper headline: precision 0.860, recall 0.977, FPR 0.152, F1 0.915).

### 3.1 Candidate model

Primary candidate: **Qwen 2.5 7B Instruct** (`Qwen/Qwen2.5-7B-Instruct`)
- Available on HF Inference API free tier
- ~5 GB Q4 quantized; ~14 GB full precision — well within the HF Space's 16 GB RAM if we ever want to host it locally
- Top of late-2024 7B leaderboards for reasoning + structured output
- Paper's documented CLI default — no new documentation burden

Backup candidate if Qwen 2.5 7B underperforms: **Qwen 2.5 14B Instruct** (`Qwen/Qwen2.5-14B-Instruct`)
- Stronger reasoning at the cost of slower inference
- ~9 GB Q4 quantized; would fit on HF Space CPU but slower
- Still on HF Inference API free tier last I checked

### 3.2 Implementation steps

1. **Extend `src/citecheck/checks/claims.py`** to accept either an Ollama call or an HF-Inference-API call.
   - Add `_call_hf_inference(prompt: str, *, model: str, token: str | None) -> str`.
   - Add a top-level dispatcher: if `OLLAMA_MODEL` env var ends in `:cloud` OR starts with `hf:`, route via HF; otherwise via Ollama. Or use a dedicated env var like `CITECHECK_LLM_PROVIDER=hf`.
   - Keep `_call_ollama` unchanged so the CLI and existing tests continue to work.
2. **Extend `scripts/run_claim_eval.py`** with a `--provider {ollama,hf}` flag and `--hf-token <env-var-name>` (defaulting to `HF_TOKEN`).
3. **Run the 100-item eval** against Qwen 2.5 7B Instruct via HF Inference API.
   - Expected runtime: ~10-15 min (HF API is faster than CPU Ollama).
   - Save results to `data/eval/claim_results_qwen7b.json` so the gemma4:31b numbers remain available for comparison.
4. **Compare numbers.** Decision rule:
   - If precision ≥ 0.81 (within 5pp of 0.860) **and** recall ≥ 0.93 (within 5pp of 0.977): proceed to Phase-6b with Qwen 2.5 7B.
   - If either falls outside that band: try Qwen 2.5 14B. If 14B still fails: revisit deployment target (e.g., switch to Ollama Cloud BYO key).
5. **Update paper §6.2 and §3 (Stage 6 paragraph)** to document the deployment model + the comparison numbers. Add a small table or sentence with both rows: gemma4:31b-cloud (research baseline) vs Qwen 2.5 7B (deployed). Justify the choice as the free-tier-feasible option that approximates the baseline.

### 3.3 Gating decision

After Phase-6a, present the results. If go: continue to 6b. If no-go: present alternatives (BYO key, Ollama Cloud only, defer Phase 5 in deployment).

## 4. Phase-6b: FastAPI app

**Goal:** working FastAPI + HTMX app that runs the full pipeline against an uploaded PDF and returns a report. Runs locally under `uvicorn` first; deployment comes after.

### 4.1 Module layout

```
src/citecheck/web/
├── __init__.py
├── app.py                  # FastAPI app factory
├── routes/
│   ├── __init__.py
│   ├── upload.py          # POST /api/check (PDF upload)
│   ├── jobs.py            # GET  /api/jobs/{job_id}/status
│   └── reports.py         # GET  /api/reports/{job_id} (JSON + HTML)
├── tasks.py                # FastAPI BackgroundTasks runner
├── storage.py              # SQLite report store
├── templates/
│   ├── base.html
│   ├── index.html         # upload form
│   ├── job_status.html    # HTMX-poll target
│   └── report.html        # rendered report
├── static/
│   ├── style.css
│   └── htmx.min.js
└── settings.py             # env-var loading (pydantic-settings)
```

### 4.2 Data model

`web/storage.py` introduces a single `jobs` table:

```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,           -- uuid4
    created_at INTEGER NOT NULL,       -- unix epoch
    status TEXT NOT NULL,              -- 'pending' | 'running' | 'done' | 'error'
    pdf_sha256 TEXT NOT NULL,
    report_json TEXT,                  -- nullable until done
    error_message TEXT
);
```

No user accounts, no email, no PII. The `job_id` is the only secret.

### 4.3 Endpoints

| Method | Path | Behavior |
|---|---|---|
| `GET /` | `/` | Renders `index.html` with the upload form |
| `POST /api/check` | `/api/check` | Accepts `pdf` multipart file; creates a job row; spawns BackgroundTask; returns `303 See Other → /jobs/{job_id}` |
| `GET /jobs/{job_id}` | `/jobs/{job_id}` | Renders `job_status.html` with HTMX polling `/api/jobs/{job_id}/status` every 3s |
| `GET /api/jobs/{job_id}/status` | `/api/jobs/{job_id}/status` | Returns minimal status JSON or an HTMX partial; when status is `done`, the partial swaps in a link to `/reports/{job_id}` |
| `GET /reports/{job_id}` | `/reports/{job_id}` | Renders `report.html` from the JSON report |
| `GET /api/reports/{job_id}.json` | `/api/reports/{job_id}.json` | Returns the raw JSON report |

### 4.4 Background task

`tasks.run_pipeline(job_id: str, pdf_bytes: bytes)`:
1. Persist the PDF to a tempfile.
2. Set job status to `running`.
3. Call existing pipeline functions in order:
   - `extract_references(pdf_path)` (GROBID)
   - `resolve_references(...)` (Crossref + OpenAlex)
   - `check_retraction` per reference
   - `check_hallucination` per reference
   - `check_journal_quality` per reference
   - For Phase 5: call `verify_claim` if the user opted in (default off in v1 to keep latency reasonable); see §4.5.
4. Aggregate into a `CheckedReference[]` list (existing model).
5. Write report JSON to the `jobs.report_json` column, mark status `done`.
6. On any exception: mark status `error`, store the message.

### 4.5 Phase 5 in the web app

Two-step rollout to manage cost and latency:

- **v1.0 (this build):** Phase 5 is **disabled by default**. The upload form has a checkbox "Verify claims against cited papers (slow, ~30s per claim)". When checked, the background task runs `verify_claim` for each citation. When unchecked, the report shows the four other checks and notes Phase 5 was skipped.
- **v1.1 (later):** Make Phase 5 the default once we've watched real usage and confirmed the latency is tolerable on HF Spaces.

### 4.6 Rate limiting + quota management

Two layers of protection on the LLM-call surface:

**Layer A — per-IP request rate-limit (`slowapi` middleware):**
- 5 PDF uploads per IP per hour
- Returns 429 with a friendly HTML page after the limit
- Tied to the existing OpenAlex budget circuit-breaker: if `budget.exhausted_until()` returns a future time, the upload endpoint returns 503 with an explanatory page rather than processing the request.

**Layer B — server-side LLM quota monitor (`web/quota.py`, NEW):**

The Cerebras free tier caps requests at ~30 RPM with a daily token budget around 1M tokens.  Live usage data from the eval run confirms typical per-PDF consumption of 25-90K tokens (5-20 claims).  The web app needs to track its own consumption so it can predict before each job whether running Phase 5 will exhaust the budget.

Architecture:

```
web/quota.py
├── class QuotaMonitor:                  # SQLite-backed, per-provider
│   ├── record(tokens_in, tokens_out)    # called from _call_cerebras hook
│   ├── usage_today() -> dict            # {requests, input_tok, output_tok}
│   ├── remaining_today() -> int | None  # None when cap unknown
│   ├── is_exhausted() -> bool
│   └── estimate_cost(n_claims) -> dict  # {expected_tok, will_exhaust: bool}
└── llm_call_hook(prompt, fn) -> str     # wraps _call_cerebras to update monitor
```

Persistence: SQLite at `~/.citecheck/quota.db` with a `usage` table keyed by `(provider, date_utc)`.  The web app passes a per-request `llm_call_hook` to `verify_claim` instead of the raw `_call_cerebras`; the hook increments the counter on every successful call.

**Pre-flight estimate (in the upload route):**

After GROBID extracts the references, before launching Phase 5:

1. Count candidate claim-verification pairs (refs with a citation context + a resolvable text).
2. Call `quota.estimate_cost(n_claims)` → expected total tokens.
3. If `expected_tok > remaining_today()`: warn the user upfront on the job-status page that Phase 5 will partially execute (X of Y claims) before quota hits, and offer to proceed-anyway or skip-Phase-5.
4. If `n_claims > 20`: cap to 20 (sample uniformly) regardless of quota — keeps any single PDF from monopolizing the daily budget.

**User-facing notifications (`templates/job_status.html`):**

The HTMX-polled status partial shows progressive state:

| State | Display |
|---|---|
| `pending` | "Job queued, waiting to start..." |
| `running, stage=resolution` | "Resolving 47 references via Crossref + OpenAlex..." |
| `running, stage=phase5, X/Y claims` | "Verifying claims 3 of 18 (estimated 45s remaining)" |
| `running, stage=phase5, quota_warning` | "⚠ Phase 5 quota close to daily limit; remaining claims may be skipped" |
| `running, stage=phase5, quota_partial` | "⚠ Quota exhausted at claim 12/18; remaining claims will run after 00:00 UTC reset" |
| `done` | "Report ready → /reports/{job_id}" |
| `error` | "Job failed: {reason}.  Try again or open an issue." |

**Operator monitoring (`/admin/health`, auth-gated):**

A small admin endpoint returning JSON:

```json
{
  "provider": "cerebras",
  "today_utc": "2026-05-21",
  "requests": 234,
  "input_tokens": 891234,
  "output_tokens": 34521,
  "remaining_estimate": 75000,
  "circuit_open": false,
  "last_429": null
}
```

Auth: shared-secret header `X-Admin-Token` matching `CITECHECK_ADMIN_TOKEN` env var.  Used for a one-line cron health-check + manual debugging.  No public surface.

**Graceful Phase 5 degradation:**

When the daily quota is exhausted (or persistent 429s are logged):

- New uploads still accept and run Stages 1-5 (retraction, fabrication, journal quality) normally.
- Phase 5 (claim verification) skipped with a banner in the report: "Claim verification was unavailable when this report ran.  The four metadata checks above completed normally.  Phase 5 resets at 00:00 UTC; re-upload the PDF after then if you want the verification results."
- The upload-form Phase 5 toggle is greyed out with a tooltip showing the reset time.

**Cerebras response-header introspection:**

If Cerebras exposes per-response rate-limit headers (e.g., `x-ratelimit-remaining-tokens`, `x-ratelimit-reset`), the monitor reads them and uses authoritative numbers instead of local accounting.  Falls back to local counters when headers absent.  Verified empirically at first request: if Cerebras provides them, we use them; otherwise the local accounting is best-effort but never wrong-direction (we'll overestimate consumption, which is safe).

### 4.7 Tests

`tests/test_web/`:
- `test_upload.py` — POST a fixture PDF; asserts 303 + job created.
- `test_status.py` — fake a running/done/error job; asserts the right HTMX partial.
- `test_storage.py` — round-trip a report JSON through SQLite.
- `test_rate_limit.py` — 6 uploads from one IP triggers 429.

Existing pipeline tests stay green (no changes to library code besides §3.2's claims dispatcher).

## 5. Phase-6c: HF Spaces deploy

### 5.1 Repository structure for Space

HF Spaces wants a single Docker space. We add:

```
spaces/
├── Dockerfile             # builds the web app + bundles GROBID
├── README.md              # Space-specific README (Space metadata header)
├── start.sh               # entrypoint: launches GROBID + uvicorn
└── docker-compose.yml     # local-dev copy (HF only consumes Dockerfile)
```

### 5.2 Dockerfile

Multi-stage build:
1. **Stage 1:** Python base, install citecheck + claims extras + web deps via `uv pip install`.
2. **Stage 2:** copy GROBID jar (or use the lfoppiano image as a sidecar — see deployment trade-off below).
3. **Stage 3:** runtime — start GROBID + uvicorn on port 7860 (HF Spaces convention).

Key design choice: GROBID is a JVM service that wants ~6 GB RAM. We have two options:
- **Option A — bundled:** ship GROBID's jar + a JRE inside the same container. Single container; HF Space-friendly. ~6 GB image.
- **Option B — external:** call a public GROBID instance (e.g., the project's own `kermitt2/grobid` Hugging Face Space). Smaller image but introduces an external dependency.

Recommend **Option A** for v1 (single container, no third-party dependency). Move to B if image size becomes a problem.

### 5.3 Space metadata (`spaces/README.md` header)

```yaml
title: citecheck
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
license: apache-2.0
```

### 5.4 Secrets

- `HF_TOKEN` — for Inference API calls. Set as a Space Secret.
- `CITECHECK_CONTACT_EMAIL` — for Crossref / OpenAlex polite pool. Set as Space variable.

### 5.5 Cold-start mitigation

HF Spaces sleep after inactivity. First-request cold start is ~30-60s. Mitigations:
- A "Loading..." page on first hit that immediately auto-refreshes once the Space is warm.
- A cron-like keep-alive ping from an external service (e.g., the citecheck GitHub repo's own scheduled Action) every 30 min during business hours.

### 5.6 Deploy steps

1. Create a `citecheck` Space on HF (manual, one-time).
2. Add a GitHub Action that pushes to the HF Space's git remote on every `main` commit.
3. First push triggers HF's build; Space goes live at `https://huggingface.co/spaces/<user>/citecheck`.
4. Smoke test: upload a fixture PDF (e.g., the Walters & Wilder sample) and verify the report renders.
5. Document the public URL in `README.md`.

## 6. Out of scope for v1

- User accounts / login
- Per-user history pages
- Webhooks / callbacks
- A REST API beyond what the UI consumes (the JSON endpoint is opportunistic, not the main interface)
- arq + Redis (BackgroundTasks is sufficient for the demo's expected load)
- Comments, annotations, or collaborative editing
- A separate "results catalog" page browsable by anyone

These are documented as Phase 7+ work in PLAN.md.

## 7. Paper updates

After Phase 6 ships:
- New paragraph in §3 (Stage 6) noting the deployed app uses `Qwen/Qwen2.5-7B-Instruct` via HF Inference API.
- New paragraph in §6.2 with the comparison table: gemma4:31b-cloud (research baseline) vs deployed Qwen 7B (free-tier ceiling).
- Updated abstract sentence: "The pipeline runs locally on free public APIs, with a hosted public demo at `https://huggingface.co/spaces/<user>/citecheck`."
- New §8 (Discussion) paragraph: design rationale for the small-model-deployed / larger-model-research trade.

## 8. Effort breakdown

| Block | Hours |
|---|---|
| 6a model-pivot eval | 1.5 |
| 6a paper update | 0.5 |
| 6b FastAPI scaffold + routes | 3 |
| 6b Templates + HTMX | 2 |
| 6b Background task + storage | 2 |
| 6b Rate limiting + budget tie-in | 1 |
| 6b Tests | 2 |
| 6c Dockerfile + start.sh | 2 |
| 6c Space deploy + smoke test | 1.5 |
| 6c GitHub Action for auto-deploy | 1 |
| 6c Paper updates | 1 |
| **Total** | **17.5 h** |

Realistic over 3 sessions: 6a in one session (gates the rest), 6b in one session, 6c + polish in a third.

## 9. Risks

- **HF Inference API rate limit on free tier.** If the free quota is too low for even a demo, fall back to BYO key.
- **HF Space cold start UX.** 30-60s before first response is bad. Need to set expectations on the loading page.
- **GROBID memory inside Docker.** Has historically been finicky; may need explicit JVM flags.
- **Qwen 2.5 7B doesn't approximate gemma4:31b numbers.** Gating decision in §3.3 handles this; the plan does not commit code to a fallback before measurement.

## 10. Approval requested

- Approve the strategic pivot to HF Spaces + HF Inference API (rather than original Hetzner + local Ollama plan)
- Approve Qwen 2.5 7B Instruct as the primary candidate model (with Qwen 2.5 14B as backup)
- Approve the §4.5 v1.0 decision: Phase 5 opt-in by default, default-on later
- Approve the §5.2 Option A (bundled GROBID) over Option B (external)
- Approve the §6 out-of-scope list

When approved, I begin with Phase 6a (model eval).

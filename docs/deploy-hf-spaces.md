# Deploying citecheck to HuggingFace Spaces

This is the one-time setup for the free public demo at
`https://huggingface.co/spaces/<your-username>/citecheck`.

## Prerequisites

- HuggingFace account.
- GitHub repo write access (for adding Secrets + Variables).
- Cerebras Cloud free-tier API key — https://cloud.cerebras.ai/.

## Step 1 — Create the HF Space (one-time, in the UI)

1. Go to https://huggingface.co/new-space.
2. Fill in:
   - **Space name:** `citecheck` (or anything; just match `HF_SPACE_NAME` below).
   - **License:** Apache 2.0.
   - **Space SDK:** **Docker** → **Blank**.
   - **Hardware:** CPU basic (free).
   - **Visibility:** Public.
3. Click **Create Space**.  You'll get an empty repo at
   `https://huggingface.co/spaces/<username>/citecheck`.

## Step 2 — Set Space Secrets (in the Space's Settings tab)

In the Space UI, **Settings → Variables and secrets**, add:

| Type | Name | Value |
|---|---|---|
| Secret | `CITECHECK_CONTACT_EMAIL` | your email (mandatory for Crossref/OpenAlex/Unpaywall polite pool) |
| Secret | `CEREBRAS_API_KEY` | your Cerebras Cloud key |

Optional (with sensible defaults; override only if needed):

| Type | Name | Default |
|---|---|---|
| Variable | `CITECHECK_WEB_RATE_LIMIT_PER_HOUR` | `5` |
| Variable | `CITECHECK_WEB_MAX_CLAIMS_PER_PDF` | `20` |
| Variable | `CITECHECK_WEB_LLM_PROVIDER` | `cerebras` |
| Variable | `CITECHECK_WEB_LLM_MODEL` | `qwen-3-235b-a22b-instruct-2507` |
| Secret | `CITECHECK_WEB_ADMIN_TOKEN` | (empty → admin endpoint disabled) |

## Step 3 — Wire up auto-deploy from GitHub (one-time)

In the GitHub repo, **Settings → Secrets and variables → Actions**:

**Secrets** (encrypted):

| Name | Value |
|---|---|
| `HF_TOKEN` | HF User Access Token with **write** scope on Spaces — generate at https://huggingface.co/settings/tokens |

**Variables** (plain text):

| Name | Value |
|---|---|
| `HF_USERNAME` | your HF username (e.g. `pierrebeaucoral`) |
| `HF_SPACE_NAME` | `citecheck` (must match Step 1) |

## Step 4 — Trigger the first deploy

Push any commit to `main`, or manually trigger the workflow:

```bash
# Manual trigger from CLI (requires `gh` CLI authenticated).
gh workflow run hf-deploy.yml
```

Or via the UI: **Actions → Deploy to HuggingFace Space → Run workflow**.

The workflow:
1. Checks out `main` with full history.
2. Swaps `spaces/README.md` into the repo-root `README.md` (so HF reads
   the YAML frontmatter — title, emoji, sdk, etc.).
3. Force-pushes the result to the Space's `main` branch.

HF Spaces then auto-rebuilds the Docker image (~10-15 min the first
time, ~3-5 min on subsequent rebuilds via layer cache).

## Step 5 — Smoke-test the live URL

Once HF reports "Running", open `https://huggingface.co/spaces/<username>/citecheck`.

Checklist:

- [ ] Index page loads (mascot rendering, upload form visible).
- [ ] Upload a small test PDF — `data/fixtures/` has a few options, or
      use any preprint with ≥5 references.
- [ ] Job-status page polls and progresses through resolution → checks.
- [ ] Report page renders with charts + per-reference cards.
- [ ] Phase 5 toggle: enable it on a small PDF and verify a claim runs.
- [ ] Quota indicator in the header shows non-zero usage after a run.

## Troubleshooting

**Build fails with "no space left on device":** HF free tier has 50 GB
disk; if the build pulls torch + sentence-transformers it may bloat
intermediate layers.  Add `--no-cache-dir` to pip installs or move to
the Tiny image variant of GROBID.

**Build succeeds but app returns 503:** GROBID didn't warm up within
120 s.  Check the Space's "Logs" tab for `[start.sh] Still waiting
for GROBID`.  Bump the timeout in `spaces/start.sh` (replace `for i in
$(seq 1 120)` with `seq 1 180`).

**Phase 5 always fails with quota error:** Daily Cerebras token cap hit.
Either wait until 00:00 UTC for the reset or upgrade to paid Cerebras.

**Quota indicator stuck at 0:** The Space's `/tmp/citecheck/data` was
wiped on the last restart.  Counters reset on every restart on the free
tier — this is expected.  For persistence, upgrade to paid hardware and
set `CITECHECK_WEB_DATA_DIR=/data`.

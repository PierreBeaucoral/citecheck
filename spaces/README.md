---
title: citecheck
emoji: ☕
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Detect retracted, fake, or predatory citations in PDFs.
tags:
  - academic-integrity
  - citations
  - retractions
  - meta-science
  - open-science
---

# citecheck — Hugging Face Space

This `README.md` (with the YAML frontmatter above) is what Hugging Face
Spaces reads to configure the Docker SDK build of [citecheck][repo].
The actual application code, Dockerfile, and entrypoint live at the
repo root; this file exists only so the Space gets the right title,
icon, and routing.

**Live demo:** https://huggingface.co/spaces/pierrebeaucoral/citecheck

**Source:** https://github.com/pierrebeaucoral/citecheck

[repo]: https://github.com/pierrebeaucoral/citecheck

## How the Space is wired

- **SDK:** `docker` — the Space builds the `Dockerfile` at the repo
  root, which layers Python 3.11 + uv + the `citecheck[web,claims]`
  package on top of `lfoppiano/grobid:0.8.2-crf`.  Single-container
  by design: GROBID and uvicorn run side-by-side inside the same
  Space, with GROBID bound to `127.0.0.1:8070` and uvicorn exposed on
  port `7860` (HF's public ingress).
- **Entrypoint:** [`spaces/start.sh`](../spaces/start.sh) launches
  GROBID in the background, waits for `/api/isalive`, then exec's
  uvicorn.
- **Auto-deploy:** Pushes to `main` on GitHub trigger
  [`.github/workflows/hf-deploy.yml`](../.github/workflows/hf-deploy.yml),
  which mirrors the repo into this Space.

## Required Secrets

Set these via **Settings → Variables and secrets** in the Space UI.
None of them are in the repo.

| Name | Purpose |
|---|---|
| `CITECHECK_CONTACT_EMAIL` | Polite-pool identifier sent to Crossref, OpenAlex, Unpaywall. Mandatory — the app refuses to start without it. |
| `CEREBRAS_API_KEY` | Phase-5 LLM provider (free tier).  Get one at https://cloud.cerebras.ai/. |

Optional:

| Name | Default | Purpose |
|---|---|---|
| `CITECHECK_WEB_LLM_PROVIDER` | `cerebras` | One of `cerebras`, `hf`, `ollama`, `anthropic`. |
| `CITECHECK_WEB_LLM_MODEL` | `qwen-3-235b-a22b-instruct-2507` | Provider-specific model id. |
| `CITECHECK_WEB_RATE_LIMIT_PER_HOUR` | `5` | PDFs per IP per hour. |
| `CITECHECK_WEB_MAX_CLAIMS_PER_PDF` | `20` | Hard cap on Phase 5 claims per PDF. |
| `CITECHECK_WEB_PHASE5_DEFAULT_ON` | `false` | Whether Phase 5 toggle defaults to checked on the upload form. |
| `CITECHECK_WEB_ADMIN_TOKEN` | `""` (disabled) | Set non-empty to enable `/admin/health`. |

## Storage and persistence

The Space uses `/tmp/citecheck/{uploads,data}` for transient state
(jobs SQLite DB, uploaded PDFs).  **The free HF tier wipes `/tmp` on
every restart** — so historical reports are not preserved across
restarts.  This is intentional for the public demo (no PII retention,
no per-user accounts).  For a persistent deployment, mount the Space's
`/data` volume (paid tier) and set `CITECHECK_WEB_DATA_DIR=/data`.

## License

Apache-2.0.  See [LICENSE](../LICENSE) in the source repo.

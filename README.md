# citecheck

**Citation integrity checker for academic papers.** Drop in a PDF; get a report of retracted, hallucinated, predatory, or misrepresented references.

Free, open-source, runs locally. No accounts, no API keys required for the core checks.

## Status

Pre-alpha. Setup + Phases 1, 2, 3 complete:

- **Phase 1** — PDF → resolved references via GROBID, Crossref, and OpenAlex fallback. 89–96% of refs resolve to canonical DOIs on test fixtures.
- **Phase 2** — Retraction check via Crossref `update-to` + OpenAlex `is_retracted` (which integrates the Retraction Watch dataset). Confirmed end-to-end against Wakefield's MMR paper.
- **Phase 3** — Five-layer hallucination detector: DOI integrity, cross-DB existence, author plausibility (OpenAlex `/authors`), venue plausibility (OpenAlex `/sources`), and a rule-based aggregator. Asymmetric low-coverage caveat (books, pre-2000) prevents false positives on legitimate older / un-indexed references. 0% FPR on the Callaway/Sant'Anna fixture; correctly flags fully-fabricated test references.

Phase 4 (predatory-journal / journal-quality flags) is the next milestone.

## What it does (when finished)

| Phase | Check | What it catches |
|---|---|---|
| 1 | Extraction | Parse references from a PDF; resolve to canonical DOIs |
| 2 | Retraction | References to retracted, expressed-concern, or corrected papers |
| 3 | Hallucination | Fabricated or non-existent citations (the LLM-era problem) |
| 4 | Journal quality | Predatory / questionable publication venues |
| 5 | Claim verification | In-text claims that the cited paper doesn't actually support |
| 6 | Web app | Drag-and-drop PDF check at a public URL |
| 7 | Corpus analysis | Aggregate rates of retraction citation and hallucination across a corpus |

## What this isn't

- **Not Scite.ai.** Scite tells you *how* a paper has been cited (supporting / contrasting). citecheck tells you whether the cited paper *exists*, was *retracted*, or *supports the claim being made*. Complementary, not competing.
- **Not Retraction Watch.** RW is the canonical retraction database. citecheck *uses* RW data via Crossref to check a paper's reference list.
- **Not a Semantic Scholar competitor.** SS has citation contexts; citecheck uses SS as one of several lookup sources for existence checks.
- **Not a plagiarism checker.** Different problem.
- **Not an automated reviewer.** It checks *citations*, not arguments.

## Quickstart

```bash
# Install
uv sync --all-extras

# Verify
uv run pytest
uv run citecheck version

# Phase 1 — extract references from a PDF
cp .env.example .env                              # then edit to set CITECHECK_CONTACT_EMAIL
docker compose up -d grobid                       # ~30s startup; healthcheck on :8070
uv run python scripts/fetch_test_fixtures.py 1803.09015     # or any arXiv ID
uv run citecheck extract data/fixtures/1803.09015.pdf       # Rich table output
uv run citecheck extract data/fixtures/1803.09015.pdf --json  # JSON for piping

# Phase 2 + 3 — extract + resolve + retraction + hallucination
uv run citecheck check data/fixtures/1803.09015.pdf            # all checks, color-coded table
uv run citecheck check data/fixtures/1803.09015.pdf --json     # machine-readable
uv run citecheck check <pdf> --only-hallucination              # skip retraction (faster)
uv run citecheck check <pdf> --skip-hallucination              # skip the 4 OpenAlex calls per ref
uv run citecheck check <pdf> --no-cache                        # bypass ~/.citecheck/cache.db

docker compose down                               # when finished
```

The CLI exits with code 3 if GROBID is unreachable (with a hint on how to start
it). Pass `--skip-liveness` to bypass the probe in CI.

**Resolution flow** (see [`src/citecheck/resolution/__init__.py`](src/citecheck/resolution/__init__.py)):
1. If the reference has a DOI (from GROBID's TEI or regex-extracted from raw text), query Crossref `/works/{doi}`.
2. Otherwise — or if (1) misses — query Crossref `/works?query.bibliographic=...` and score candidates against the input.
3. If Crossref returns no confident match, fall back to OpenAlex `/works?search=...` for independent indexing.

OpenAlex needs no API key. Crossref's polite pool is enabled when `CITECHECK_CONTACT_EMAIL` is set in `.env`.

## Repository layout

```
citecheck/
  src/citecheck/
    extraction/   # PDF -> raw references (GROBID, phase 1)
    resolution/   # Reference -> canonical metadata (Crossref/OpenAlex, phase 1)
    checks/       # cache.py + retractions.py (phase 2); hallucination, etc. in 3-5
    pipeline/     # Orchestrators: extract (phase 1) + check (phase 2)
    models.py     # Pydantic schemas
    cli.py        # Typer CLI entry
  tests/
  data/
    fixtures/     # Sample PDFs for unit tests (gitignored)
    eval/         # Labeled hallucination eval set (built during phase 1)
  docs/
    PRIVACY.md    # PDF retention policy for the eventual web app
  scripts/
  .github/workflows/ci.yml
```

## Privacy

If you use the public web app (phase 6), uploaded PDFs are deleted after 24 hours. Reports (text-only) persist by UUID. See [`docs/PRIVACY.md`](docs/PRIVACY.md).

## Contributing

Contributions welcome once Phase 1 lands. Until then the API is unstable.

## License

Apache-2.0 — see [`LICENSE`](LICENSE). The patent grant matters: institutions and forks are explicitly allowed to use this without IP risk.

## Citation

If you use citecheck for research, please cite the corpus analysis preprint (forthcoming, Phase 7).

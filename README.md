# citecheck

**Citation integrity checker for academic papers.** Drop in a PDF; get a report of retracted, hallucinated, predatory, or misrepresented references.

Free, open-source, runs locally. No accounts, no API keys required for the core checks.

## Status

Pre-alpha. Setup phase complete. Phase 1 (reference extraction) is the next milestone.

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
uv sync

# Verify
uv run pytest
uv run citecheck version
```

Phase 1+ commands will be added as functionality lands.

## Repository layout

```
citecheck/
  src/citecheck/
    extraction/   # PDF -> raw references (GROBID, phase 1)
    resolution/   # Reference -> canonical metadata (Crossref/OpenAlex, phase 1)
    checks/       # Retraction, hallucination, journal-quality, claims (phases 2-5)
    pipeline/     # Orchestration (phases 2+)
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

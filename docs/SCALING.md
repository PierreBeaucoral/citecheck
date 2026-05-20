# Scaling citecheck

This document covers operational concerns for deploying citecheck beyond a
single-researcher local install: API budgets, caching, fallback modes, and
the four paths to scale past the free-tier daily caps.

## API surface

Per typical paper (~50 references), citecheck makes:

| Provider | Endpoint | Calls per paper (uncached) | Notes |
|---|---|---|---|
| GROBID | `/api/processReferences` | 1 | Local; no rate limit |
| GROBID | `/api/processFulltextDocument` | 1 (if `--verify-claims`) | Local |
| Crossref | `/works/{doi}` | up to 50 | Free polite-pool; no daily cap |
| Crossref | `/works?query.bibliographic=` | up to 50 | Free; metadata fallback |
| OpenAlex | `/works?search=` | up to 50 | $0.0001/call; daily budget |
| OpenAlex | `/works/doi:{doi}` (retraction) | up to 50 | Same budget |
| OpenAlex | `/authors?search=` (L3) | up to 100 | 2 authors × 50 refs |
| OpenAlex | `/sources?search=` (L4) | up to 50 | Skipped if no journal |
| OpenAlex | `/works?filter=authorships.author.id` (L5) | up to 100 | Skipped if no journal |
| DOAJ | `/api/v3/search/journals/` | up to 50 | Free; no documented daily cap |
| Unpaywall | `/v2/{doi}` (if `--verify-claims`) | up to 50 | Free; requires mailto |
| Ollama | `/api/generate` (if `--verify-claims`) | up to 50 × N claims | Local; no API cost |

Total: ~6.5 OpenAlex calls per reference on first encounter, ~0 on repeat
once the SQLite cache is warm.

## Budget math

OpenAlex's polite-pool charges roughly $0.0001 per request and provides a
daily allowance (observed: ~$1/day per email, may vary). Typical paper:
~325 fresh OpenAlex calls = ~$0.03 of budget consumed.

| Cache state | Papers/day before cap |
|---|---|
| Cold (every ref new) | ~30 |
| Warm (~70% cache hit) | ~100 |
| Replays of same papers | unlimited |

For a web-app deployment (Phase 6) the relevant axis is daily new-paper
volume across all users.

## Built-in protections

Citecheck already ships three pieces that keep operating costs in check:

1. **SQLite cache with 7-day TTL** at `~/.citecheck/cache.db`. Every
   external API response is cached; subsequent calls within the TTL are
   free. Cache survives process restarts.

2. **Circuit-breaker** (`src/citecheck/budget.py`). When OpenAlex returns
   429 with body `Insufficient budget`, a state flag at
   `~/.citecheck/openalex_exhausted_until.iso` records the expiry
   (midnight UTC). Until then every OpenAlex call short-circuits to None;
   each layer's response treats None as "benefit of the doubt" so the
   tool continues to run in degraded mode rather than crashing.

3. **Automatic Crossref-only fallback**. When the circuit is open, layers
   that depend on OpenAlex (retraction is_retracted check, L3, L4, L5)
   return non-flagged results with a "OpenAlex daily budget exhausted"
   note. The CLI emits a yellow warning banner before the report. The
   user sees a meaningful but partial check rather than an error.

4. **API consumption tracker** (`src/citecheck/metrics.py`). Every API
   response writes one row to `~/.citecheck/metrics.db`. Inspect with:

   ```bash
   uv run citecheck stats
   ```

   Shows per-(source, endpoint) calls today and rolling 7-day total.

## Four scaling paths

### A. Cache warming (free, light effort)

For repeat papers or papers from a stable author community, the cache
absorbs most traffic. Optionally pre-warm:

```bash
# Hit the most-cited 10k authors and 1k journals once during deploy
uv run python scripts/warm_cache.py --top-authors 10000 --top-sources 1000
```

(script is a TODO; the cache will warm naturally over normal use.)

### B. OpenAlex paid tier (recommended for low-volume public deployment)

Pay-as-you-go at ~$0.0001/query. Adding $30/month covers ~300 papers/day
on cold cache, easily 10× that in steady state. Sign up at
<https://openalex.org/pricing>.

### C. DuckDB over OpenAlex parquet snapshot (single-machine, full local)

The OpenAlex snapshot is freely downloadable from S3. Load just the
slices we need (works + authors + sources) into DuckDB:

```bash
aws s3 sync s3://openalex/data/works/   ./openalex/works/   --no-sign-request
aws s3 sync s3://openalex/data/authors/ ./openalex/authors/ --no-sign-request
aws s3 sync s3://openalex/data/sources/ ./openalex/sources/ --no-sign-request

duckdb openalex.db <<'SQL'
CREATE TABLE works   AS SELECT * FROM read_json_auto('openalex/works/*/*.gz',   format='newline_delimited');
CREATE TABLE authors AS SELECT * FROM read_json_auto('openalex/authors/*/*.gz', format='newline_delimited');
CREATE TABLE sources AS SELECT * FROM read_json_auto('openalex/sources/*/*.gz', format='newline_delimited');
SQL
```

Then add a new `citecheck.resolution.openalex_local` module that translates
the same endpoints into DuckDB SQL. Set `OPENALEX_BACKEND=duckdb` to switch.

Disk: ~200 GB for the slices we need. Cost: ~$10 storage; ~zero per query.

### D. Full Postgres self-host (production)

Canonical answer for a public web app. Use the OpenAlex flatten-files
loader at <https://github.com/ourresearch/openalex-documentation-scripts>
to populate Postgres, then stand up a FastAPI wrapper around the five
endpoints citecheck uses:

- `GET /works/doi:{doi}`
- `GET /works?filter=is_retracted:true`
- `GET /works?filter=authorships.author.id:{id}&search=...`
- `GET /authors?search=...`
- `GET /sources?search=...`

Hardware: ~3-4 TB SSD, 32+ GB RAM. Cloud: Hetzner AX42 (~€60/mo) or
similar. Monthly snapshot refresh; ~12 hours each pull.

Point citecheck at it with `OPENALEX_BASE=http://your-server:8080`.

## Recommendations by deployment size

| Daily new papers | Recommended setup |
|---|---|
| <30 | Free tier; cache + circuit-breaker |
| 30–300 | OpenAlex paid (~$30/mo) |
| 300–3000 | DuckDB local (single machine) |
| 3000+ | Full Postgres self-host |

## Per-user rate limiting (Phase 6)

For a public web app, also enforce per-IP rate limits so a single user
cannot exhaust the daily allowance. Example with `slowapi` / `fastapi`:

```python
@app.post("/api/check", dependencies=[Depends(rate_limit("10/hour"))])
async def check_pdf(...) -> JobId: ...
```

Combined with the circuit-breaker, this gives graceful degradation:
1. Hit individual user limit → 429 with retry-after.
2. Hit collective OpenAlex limit → switch to Crossref-only mode, warn
   user in the report, accept reduced detection coverage until midnight UTC.

The current state file at `~/.citecheck/openalex_exhausted_until.iso` is
process-local; for a multi-replica web app, move it to Redis with a TTL
to the same midnight-UTC expiry.

"""Unpaywall client: fetch an open-access PDF URL for a DOI, download + cache.

Used in Phase 5: claim verification needs the actual text of the cited paper.
Unpaywall (https://unpaywall.org) is a free database of open-access locations
keyed by DOI; their REST API requires only an email parameter.

We cache the API response in the shared SQLite cache (7-day TTL) and the
downloaded PDF in ~/.citecheck/papers/<safe_doi>.pdf. Both keep us under
Unpaywall's rate limit (100k req/day, very lenient) and survive re-runs.

The download is bounded by `MAX_PDF_BYTES` (~25 MB) — enough for any real
article, small enough to avoid filling disk if a publisher serves a giant
PDF or supplementary bundle.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from citecheck.checks.cache import CacheStore

log = logging.getLogger(__name__)

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
PAPERS_DIR = Path(
    os.environ.get("CITECHECK_PAPERS_DIR", str(Path.home() / ".citecheck" / "papers"))
)
MAX_PDF_BYTES = 25 * 1024 * 1024
DEFAULT_TIMEOUT_S = 60.0


def _polite_email() -> str:
    email = os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()
    if not email or "@" not in email:
        # Unpaywall mandates an email; bail loudly rather than silently
        # making 4xx requests against their service.
        raise RuntimeError(
            "Unpaywall requires CITECHECK_CONTACT_EMAIL in .env. "
            "Add a valid email address before invoking the claim-verification flow."
        )
    return email


def _safe_doi(doi: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", doi.strip().lower())


def _papers_dir() -> Path:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    return PAPERS_DIR


@retry(
    retry=retry_if_exception_type(httpx.RequestError),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _lookup_unpaywall(doi: str, *, timeout_s: float) -> dict | None:
    """Ask Unpaywall for the OA location record for `doi`."""
    url = f"{UNPAYWALL_BASE}/{doi}"
    try:
        resp = httpx.get(url, params={"email": _polite_email()}, timeout=timeout_s)
    except httpx.RequestError as exc:
        log.warning("unpaywall: request failed for %s: %s", doi, exc)
        return None
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        log.warning("unpaywall: %s for %s: %s", resp.status_code, doi, resp.text[:120])
        return None
    return resp.json() or None


def best_oa_url(
    doi: str, *, cache: CacheStore | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> str | None:
    """Return the best OA PDF URL for `doi`, or None if no open-access location is known.

    Cache key: `unpaywall:<doi>`. The cached payload is the slim subset of the
    Unpaywall record we care about (so we never have to re-fetch the full
    record on cache hit).
    """
    doi = (doi or "").lower().strip()
    if not doi:
        return None

    key = f"unpaywall:{doi}"
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached.get("url") if cached else None

    record = _lookup_unpaywall(doi, timeout_s=timeout_s)
    if record is None:
        if cache is not None:
            cache.set(key, {})  # remember the miss
        return None

    # Unpaywall returns the recommended location at `best_oa_location`; fall
    # back to `oa_locations[0]` if best is null.
    best = record.get("best_oa_location") or {}
    if not best.get("url_for_pdf"):
        for loc in record.get("oa_locations") or []:
            if loc.get("url_for_pdf"):
                best = loc
                break
    url = best.get("url_for_pdf") or best.get("url")
    slim = {
        "url": url,
        "host_type": best.get("host_type"),
        "license": best.get("license"),
        "version": best.get("version"),
        "is_oa": record.get("is_oa", False),
    }
    if cache is not None:
        cache.set(key, slim)
    return url


def fetch_pdf(
    doi: str, *, cache: CacheStore | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> Path | None:
    """Download and cache the OA PDF for `doi`; return the local path.

    Returns None when:
    - Unpaywall has no record for the DOI
    - No OA PDF URL is available (paywalled)
    - The download exceeds MAX_PDF_BYTES or the response is not a PDF
    """
    doi = (doi or "").lower().strip()
    if not doi:
        return None

    target = _papers_dir() / f"{_safe_doi(doi)}.pdf"
    if target.exists() and target.stat().st_size > 0:
        return target

    url = best_oa_url(doi, cache=cache, timeout_s=timeout_s)
    if not url:
        return None

    try:
        with (
            httpx.Client(timeout=timeout_s, follow_redirects=True) as client,
            client.stream("GET", url) as resp,
        ):
            if resp.status_code != 200:
                log.warning("unpaywall: PDF fetch %s for %s -> %s", resp.status_code, doi, url)
                return None
            content_type = resp.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                log.warning("unpaywall: not a PDF (content-type=%s) for %s", content_type, doi)
                return None
            buf = bytearray()
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                buf.extend(chunk)
                if len(buf) > MAX_PDF_BYTES:
                    log.warning("unpaywall: PDF too large (>%d) for %s", MAX_PDF_BYTES, doi)
                    return None
    except httpx.RequestError as exc:
        log.warning("unpaywall: PDF download failed for %s: %s", doi, exc)
        return None

    if not bytes(buf).startswith(b"%PDF"):
        log.warning("unpaywall: downloaded bytes are not a PDF for %s", doi)
        return None

    target.write_bytes(bytes(buf))
    return target

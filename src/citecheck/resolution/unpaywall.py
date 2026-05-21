"""Unpaywall (and PMC) client: fetch an open-access PDF for a paper, cache it.

Used in Phase 5: claim verification needs the actual text of the cited paper.

Two paths:

* **DOI-keyed**: Unpaywall (https://unpaywall.org) is a free database of OA
  locations keyed by DOI; their REST API requires only an email parameter.
  We hit `/v2/<doi>` and read `best_oa_location.url_for_pdf`.

* **PMC-keyed**: when the source carries only a PubMed Central ID rather than
  a DOI (common for many biomedical papers and for the S6/S9 sources in the
  Phase 5 claim-verification eval), we skip Unpaywall and fetch directly from
  `https://www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxx/pdf/`. PMC's OA subset is
  one of the largest free-PDF sources in the world; no auth required.

We cache the Unpaywall response in the shared SQLite cache (7-day TTL) and
the downloaded PDF in `~/.citecheck/papers/<safe_id>.pdf`. Both keep us under
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


def _publisher_pdf_candidates(doi: str) -> list[str]:
    """Known-pattern direct-PDF URLs for publishers where Unpaywall sometimes
    returns the article landing page instead of the PDF itself.

    Order matters: the caller tries these in sequence after the Unpaywall URL
    fails (HTML response or 4xx).  PLOS is the only entry we needed for the
    Phase 5 eval, but the structure makes it trivial to add more publishers
    later (eLife, BMC, MDPI, etc.) when their patterns are encountered.
    """
    out: list[str] = []
    # PLOS journals — DOIs of the form 10.1371/journal.<jrnl>.<id> serve their
    # PDFs at this stable URL pattern.
    if doi.startswith("10.1371/journal."):
        out.append(
            f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable"
        )
    return out


def _download_pdf(
    url: str,
    target: Path,
    *,
    timeout_s: float,
) -> bool:
    """Stream a candidate PDF URL into `target`.  Return True on success.

    Factored out so fetch_pdf can try Unpaywall's URL first, then iterate the
    publisher fallbacks without duplicating the streaming + validation logic.
    """
    try:
        with (
            httpx.Client(timeout=timeout_s, follow_redirects=True) as client,
            client.stream("GET", url) as resp,
        ):
            if resp.status_code != 200:
                log.warning("pdf fetch %s for %s", resp.status_code, url)
                return False
            content_type = resp.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                log.warning("pdf fetch: not a PDF (content-type=%s) for %s", content_type, url)
                return False
            buf = bytearray()
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                buf.extend(chunk)
                if len(buf) > MAX_PDF_BYTES:
                    log.warning("pdf fetch: too large (>%d) for %s", MAX_PDF_BYTES, url)
                    return False
    except httpx.RequestError as exc:
        log.warning("pdf fetch failed for %s: %s", url, exc)
        return False
    if not bytes(buf).startswith(b"%PDF"):
        log.warning("pdf fetch: downloaded bytes are not a PDF for %s", url)
        return False
    target.write_bytes(bytes(buf))
    return True


def fetch_pdf(
    doi: str, *, cache: CacheStore | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> Path | None:
    """Download and cache the OA PDF for `doi`; return the local path.

    Returns None when:
    - Unpaywall has no record for the DOI
    - No OA PDF URL is available (paywalled)
    - The download exceeds MAX_PDF_BYTES or the response is not a PDF

    The lookup is Unpaywall-first; on failure (HTML response, 403, etc.) we
    try a small set of known publisher direct-PDF URL patterns before giving
    up.  This recovers the PLOS cases where Unpaywall hands us the article
    landing page instead of the PDF.
    """
    doi = (doi or "").lower().strip()
    if not doi:
        return None

    target = _papers_dir() / f"{_safe_doi(doi)}.pdf"
    if target.exists() and target.stat().st_size > 0:
        return target

    candidates: list[str] = []
    primary = best_oa_url(doi, cache=cache, timeout_s=timeout_s)
    if primary:
        candidates.append(primary)
    candidates.extend(_publisher_pdf_candidates(doi))
    if not candidates:
        return None
    for url in candidates:
        if _download_pdf(url, target, timeout_s=timeout_s):
            return target
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


# ---- PMC E-utilities text fetch -------------------------------------------

# NCBI's web frontend (pmc.ncbi.nlm.nih.gov) now serves a JavaScript proof-of-
# work interstitial before returning PDFs, which makes the direct-URL approach
# brittle for unattended scripts.  The E-utilities API path bypasses the POW
# entirely and returns JATS XML, which contains the article text in a much
# cleaner structured form than extracted PDF text.  No auth, no rate-limit
# token; only an email parameter to identify polite traffic (same convention
# as Unpaywall and Crossref).
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _normalize_pmc(pmc_id: str) -> str:
    """Accept 'PMC8076531', 'pmc8076531', or '8076531' and return 'PMC8076531'."""
    s = (pmc_id or "").strip().upper()
    if not s:
        return ""
    if not s.startswith("PMC"):
        s = "PMC" + s
    return s


def _strip_jats_to_text(xml_body: str) -> str:
    """Convert a JATS XML document into plain text suitable for the embedder.

    We strip XML tags but keep section headings and paragraph boundaries by
    inserting newlines.  ElementTree handles the parse; if it fails (NCBI
    sometimes returns multi-document XML for retractions and corrections) we
    fall back to a regex strip so the downstream pipeline still gets text.
    """
    import re as _re
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_body)
    except ET.ParseError:
        # Be lenient on the response: dump tags, collapse whitespace.
        text = _re.sub(r"<[^>]+>", " ", xml_body)
        return _re.sub(r"\s+", " ", text).strip()

    pieces: list[str] = []
    # Track headings and paragraphs so we keep some structure.
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # strip namespace
        if tag in {"title", "label"}:
            if elem.text:
                pieces.append("\n\n" + elem.text.strip())
        elif tag in {"p", "abstract", "sec", "caption"}:
            text = "".join(elem.itertext())
            if text and text.strip():
                pieces.append("\n" + text.strip())
    combined = "".join(pieces).strip()
    if combined:
        return combined
    # Fallback: dump every text node so we never return empty when the XML
    # had content but did not match the tags above.
    return "".join(root.itertext()).strip()


def fetch_pmc_text(
    pmc_id: str,
    *,
    cache: CacheStore | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str | None:
    """Return the article's full text (JATS-derived) for a PMC article.

    Returns None when the article is not in PMC's OA subset (no XML returned)
    or when the response is empty.  The fetched text is cached on disk at
    `~/.citecheck/papers/<pmcid_lower>.txt` so repeated eval runs are fast.

    Why text and not PDF: PMC's frontend now requires solving a JS proof-of-
    work challenge to download PDFs.  E-utilities' efetch endpoint is a
    documented programmatic interface (no POW) and returns JATS XML which is
    in fact a better fulltext source for our embedder than pypdf's PDF text
    extraction (no broken cross-page word boundaries, no header/footer noise).
    """
    pmcid = _normalize_pmc(pmc_id)
    if not pmcid:
        return None

    target = _papers_dir() / f"{pmcid.lower()}.txt"
    if target.exists() and target.stat().st_size > 0:
        return target.read_text(encoding="utf-8")

    bare_id = pmcid.removeprefix("PMC")
    params = {
        "db": "pmc",
        "id": bare_id,
        "rettype": "xml",
        "retmode": "xml",
    }
    email = os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()
    if email and "@" in email:
        params["email"] = email
        params["tool"] = "citecheck"

    try:
        resp = httpx.get(_EFETCH_URL, params=params, timeout=timeout_s)
    except httpx.RequestError as exc:
        log.warning("pmc: efetch request failed for %s: %s", pmcid, exc)
        return None
    if resp.status_code != 200:
        log.warning("pmc: efetch %s for %s", resp.status_code, pmcid)
        return None

    body = resp.text or ""
    if not body.strip() or "<error>" in body[:200].lower():
        # E-utilities returns a small <error> element when the ID is not in
        # the OA subset or is unknown.
        log.info("pmc: %s not in OA subset (empty/error body)", pmcid)
        return None

    text = _strip_jats_to_text(body)
    if not text or len(text) < 200:
        log.warning("pmc: %s returned suspiciously short text (%d chars)", pmcid, len(text))
        return None

    target.write_text(text, encoding="utf-8")
    return text


def _pmc_id_from_url(url: str | None) -> str | None:
    """Pull a PMC ID out of an Unpaywall URL like
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10569980/pdf/main.pdf."""
    if not url:
        return None
    import re as _re

    match = _re.search(r"/PMC(\d+)\b", url)
    return f"PMC{match.group(1)}" if match else None


def fetch_text_any(
    *,
    doi: str | None = None,
    pmc_id: str | None = None,
    cache: CacheStore | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[str | None, str | None]:
    """Return (text, source_label) for a paper using whichever ID is available.

    Source label is one of: "unpaywall_pdf", "pmc_xml", or None when neither
    route resolved.  The Phase 5 eval calls this to populate verify_claim's
    `text` argument and to record provenance in the per-item output.

    Routing logic:
      1. DOI path: try Unpaywall PDF + pypdf.
      2. If that fails AND Unpaywall's recommended OA location is on PMC
         (pmc.ncbi.nlm.nih.gov, which now serves a JS proof-of-work page for
         PDFs), extract the PMC ID from the URL and fall through to the PMC
         text route.
      3. PMC path: fetch JATS XML via E-utilities and strip to text.
    """
    if doi:
        pdf_path = fetch_pdf(doi, cache=cache, timeout_s=timeout_s)
        if pdf_path is not None:
            try:
                from pypdf import PdfReader  # type: ignore[import-not-found]

                reader = PdfReader(str(pdf_path))
                text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
                if text and text.strip():
                    return text, "unpaywall_pdf"
            except Exception as exc:  # noqa: BLE001
                log.warning("pdf text extraction failed for %s: %s", doi, exc)
        # Fall through: see if Unpaywall handed us a PMC URL we can route
        # through E-utilities instead.
        if pmc_id is None:
            cached_url = best_oa_url(doi, cache=cache, timeout_s=timeout_s)
            pmc_id = _pmc_id_from_url(cached_url)
    if pmc_id:
        text = fetch_pmc_text(pmc_id, cache=cache, timeout_s=timeout_s)
        if text is not None:
            return text, "pmc_xml"
    # Last resort: abstract-only verification.  OpenAlex stores abstracts as
    # `abstract_inverted_index` (token positions → token) for almost every
    # indexed work; we reconstruct it into running text.  This dramatically
    # extends Phase 5 coverage to paywalled refs but with lower confidence,
    # so the source label is distinct so the report can warn the user.
    if doi:
        abstract = _fetch_openalex_abstract(doi, cache=cache, timeout_s=timeout_s)
        if abstract:
            return abstract, "openalex_abstract"
    return None, None


def _fetch_openalex_abstract(
    doi: str,
    *,
    cache: CacheStore | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str | None:
    """Fetch the abstract for `doi` from OpenAlex; reconstruct from inverted index.

    OpenAlex returns abstracts as an inverted index of `{token: [positions]}`
    rather than running text (a copyright-friendly format that's lossless
    enough for full-text reconstruction).  We rebuild the sentence(s).

    Returns None when OpenAlex doesn't know the DOI, has no abstract, the
    daily budget is exhausted (circuit-breaker), or the response is
    malformed.  Cache key: `openalex:abstract:<doi>`.
    """
    from citecheck import budget

    doi = (doi or "").lower().strip()
    if not doi:
        return None

    key = f"openalex:abstract:{doi}"
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached.get("abstract") or None if isinstance(cached, dict) else None

    url = f"https://api.openalex.org/works/doi:{doi}"
    params = {"select": "abstract_inverted_index"}
    email = os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()
    if email and "@" in email:
        params["mailto"] = email

    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = budget.openalex_get(client, url, **params)
    except httpx.RequestError as exc:
        log.warning("openalex abstract fetch failed for %s: %s", doi, exc)
        return None
    if resp is None or resp.status_code != 200:
        return None

    body = resp.json() or {}
    inv = body.get("abstract_inverted_index") or {}
    if not inv:
        if cache is not None:
            cache.set(key, {"abstract": None})
        return None

    # Reconstruct: invert the index, sort by position, join.  Some tokens
    # repeat at multiple positions; the natural order returns the original.
    positions: list[tuple[int, str]] = []
    for token, posns in inv.items():
        if not isinstance(posns, list):
            continue
        for p in posns:
            try:
                positions.append((int(p), str(token)))
            except (TypeError, ValueError):
                continue
    positions.sort(key=lambda x: x[0])
    abstract = " ".join(token for _, token in positions).strip()
    if cache is not None:
        cache.set(key, {"abstract": abstract})
    return abstract or None

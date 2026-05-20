"""GROBID client: PDF -> list of RawReference via /api/processReferences.

GROBID returns TEI XML; we parse the subset of <biblStruct> children needed to
populate RawReference and ignore the rest. We deliberately do NOT ask GROBID
to consolidate citations with Crossref (consolidateCitations=0): resolution is
this project's job and we want it under our control with caching, polite
headers, and our own scoring.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from citecheck.models import Author, RawReference, extract_doi_from_text

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"tei": TEI_NS}

DEFAULT_HOST = "http://localhost:8070"
DEFAULT_TIMEOUT_S = 120.0


class GrobidError(RuntimeError):
    """Raised when GROBID returns an unexpected status or unparseable TEI."""


def _grobid_host() -> str:
    return os.environ.get("GROBID_HOST", DEFAULT_HOST).rstrip("/")


@retry(
    retry=retry_if_exception_type((httpx.RequestError, GrobidError)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _post_pdf(pdf_path: Path, *, host: str, timeout_s: float) -> str:
    """POST the PDF to /api/processReferences and return the TEI body as text."""
    url = f"{host}/api/processReferences"
    with pdf_path.open("rb") as fh:
        files = {"input": (pdf_path.name, fh, "application/pdf")}
        data = {
            # Raw citation strings are needed for the raw_text field — they survive
            # round-trips and let downstream layers fall back to fuzzy matching.
            "includeRawCitations": "1",
            # We resolve via Crossref separately; do not let GROBID call out.
            "consolidateCitations": "0",
        }
        resp = httpx.post(url, files=files, data=data, timeout=timeout_s)
    if resp.status_code >= 500:
        raise GrobidError(f"GROBID returned {resp.status_code}: {resp.text[:200]}")
    if resp.status_code != 200:
        # 4xx is not retryable — bad input, not a transient failure.
        raise GrobidError(f"GROBID returned {resp.status_code}: {resp.text[:200]}")
    return resp.text


def _text(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    # itertext joins text inside nested tags (e.g., <hi>) which TEI sprinkles freely.
    text = "".join(elem.itertext()).strip()
    return text or None


def _first(elem: ET.Element, path: str) -> ET.Element | None:
    return elem.find(path, NS)


def _parse_author(persname: ET.Element) -> Author | None:
    surname = _text(_first(persname, "tei:surname"))
    if not surname:
        return None
    given_parts = [_text(e) for e in persname.findall("tei:forename", NS) if _text(e) is not None]
    given = " ".join(p for p in given_parts if p) or None
    return Author(family=surname, given=given)


def _parse_year(monogr: ET.Element | None) -> int | None:
    """TEI dates live under monogr/imprint/date[@when]. Be lenient about format."""
    if monogr is None:
        return None
    date = _first(monogr, "tei:imprint/tei:date")
    if date is None:
        return None
    when = date.get("when") or _text(date)
    if not when:
        return None
    match = re.match(r"(\d{4})", when)
    if not match:
        return None
    year = int(match.group(1))
    return year if 1500 <= year <= 2100 else None


def _parse_pages(monogr: ET.Element | None) -> str | None:
    if monogr is None:
        return None
    scope = monogr.find("tei:imprint/tei:biblScope[@unit='page']", NS)
    if scope is None:
        return None
    text = _text(scope)
    if text:
        return text
    frm, to = scope.get("from"), scope.get("to")
    if frm and to:
        return f"{frm}-{to}"
    return frm or to


def _parse_scope(monogr: ET.Element | None, unit: str) -> str | None:
    if monogr is None:
        return None
    scope = monogr.find(f"tei:imprint/tei:biblScope[@unit='{unit}']", NS)
    return _text(scope)


def _parse_biblstruct(node: ET.Element) -> RawReference:
    """Map a single <biblStruct> element to a RawReference."""
    ref_id = node.get(f"{{{XML_NS}}}id")
    analytic = _first(node, "tei:analytic")
    monogr = _first(node, "tei:monogr")

    # Title: article title under analytic; otherwise monograph (book/proceedings) title.
    # Use explicit `is None` rather than truth-testing Elements (deprecated in 3.12+).
    title = None
    if analytic is not None:
        title_elem = analytic.find("tei:title[@type='main']", NS)
        if title_elem is None:
            title_elem = _first(analytic, "tei:title")
        title = _text(title_elem)
    if not title and monogr is not None:
        title = _text(_first(monogr, "tei:title"))

    # Authors: prefer analytic/author, fall back to monogr/author.
    persnames = []
    if analytic is not None:
        persnames = analytic.findall("tei:author/tei:persName", NS)
    if not persnames and monogr is not None:
        persnames = monogr.findall("tei:author/tei:persName", NS)
    authors = [a for a in (_parse_author(p) for p in persnames) if a is not None]

    # Journal: <title level='j'> under monogr.
    journal = None
    if monogr is not None:
        journal = _text(monogr.find("tei:title[@level='j']", NS))

    # DOI: usually under analytic/idno[@type='DOI'], occasionally under monogr.
    doi = None
    if analytic is not None:
        doi = _text(analytic.find("tei:idno[@type='DOI']", NS))
    if not doi and monogr is not None:
        doi = _text(monogr.find("tei:idno[@type='DOI']", NS))

    # Raw citation — captured because includeRawCitations=1.
    raw_text = _text(_first(node, "tei:note[@type='raw_reference']"))
    if not raw_text:
        # Fallback: assemble what we have so RawReference still has a non-empty raw_text.
        fallback_parts = [p for p in (title, journal, str(_parse_year(monogr) or "")) if p]
        raw_text = " · ".join(fallback_parts) or "<unparsed reference>"

    # Last-chance DOI: when GROBID's structured field is empty, the raw text may
    # still contain a DOI inline (e.g. "... doi:10.1234/abc"). Cheap; only fires
    # when we'd otherwise have nothing.
    if not doi:
        doi = extract_doi_from_text(raw_text)

    return RawReference(
        ref_id=ref_id,
        raw_text=raw_text,
        title=title,
        authors=authors,
        year=_parse_year(monogr),
        journal=journal,
        volume=_parse_scope(monogr, "volume"),
        issue=_parse_scope(monogr, "issue"),
        pages=_parse_pages(monogr),
        doi=doi,
    )


def parse_tei_references(tei_xml: str) -> list[RawReference]:
    """Parse the TEI body returned by /api/processReferences into RawReference objects."""
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        raise GrobidError(f"unparseable TEI: {exc}") from exc

    # References live under teiHeader/.../listBibl OR under text/back/div/listBibl.
    # Searching descendant-wise picks up both.
    return [_parse_biblstruct(b) for b in root.iter(f"{{{TEI_NS}}}biblStruct")]


def extract_references(
    pdf_path: str | Path, *, host: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> list[RawReference]:
    """Extract references from a PDF via GROBID.

    Raises:
        FileNotFoundError: if pdf_path does not exist.
        GrobidError: if GROBID returns non-200 or malformed TEI.
        httpx.RequestError: if the GROBID host is unreachable after retries.
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    tei = _post_pdf(path, host=host or _grobid_host(), timeout_s=timeout_s)
    return parse_tei_references(tei)


@retry(
    retry=retry_if_exception_type((httpx.RequestError, GrobidError)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _post_citation(citation: str, *, host: str, timeout_s: float) -> str:
    """POST a single citation string to /api/processCitation."""
    url = f"{host}/api/processCitation"
    data = {"citations": citation, "consolidateCitations": "0"}
    resp = httpx.post(url, data=data, timeout=timeout_s)
    if resp.status_code >= 500:
        raise GrobidError(f"GROBID returned {resp.status_code}: {resp.text[:200]}")
    if resp.status_code != 200:
        raise GrobidError(f"GROBID returned {resp.status_code}: {resp.text[:200]}")
    return resp.text


def parse_citation_string(
    citation: str, *, host: str | None = None, timeout_s: float = 60.0
) -> RawReference:
    """Parse a single plain-text citation via GROBID's /api/processCitation.

    Used by the eval-set builder (scripts/build_eval_set.py) to convert raw
    citation strings into RawReference objects without needing a full PDF.
    The returned TEI is a single <biblStruct>; we wrap it in a minimal TEI
    document so the existing parser can handle it.
    """
    tei_fragment = _post_citation(citation, host=host or _grobid_host(), timeout_s=timeout_s)
    wrapped = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">'
        "<text><back><div><listBibl>" + tei_fragment + "</listBibl></div></back></text></TEI>"
    )
    refs = parse_tei_references(wrapped)
    if not refs:
        # GROBID returned a bare biblStruct outside the TEI root we synthesized;
        # try parsing the fragment directly.
        refs = parse_tei_references(tei_fragment)
    if not refs:
        # Fallback: return a RawReference with raw_text only so downstream resolution
        # can still try metadata search on the original string.
        return RawReference(raw_text=citation)
    out = refs[0]
    # Preserve the original citation string verbatim — GROBID's raw_reference note
    # is sometimes empty when input was a free-text citation.
    return out.model_copy(update={"raw_text": citation or out.raw_text})


def is_alive(host: str | None = None, *, timeout_s: float = 5.0) -> bool:
    """Quick liveness probe — used by the CLI to give a friendly error if GROBID is down."""
    url = f"{(host or _grobid_host())}/api/isalive"
    try:
        resp = httpx.get(url, timeout=timeout_s)
    except httpx.RequestError:
        return False
    return resp.status_code == 200 and resp.text.strip().lower() == "true"

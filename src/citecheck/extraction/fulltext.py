"""Full-text extraction via GROBID for Phase 5 (claim verification).

`extract_references` (sibling module) returns only the reference list; the
hallucination and retraction checks need nothing more. The claim-verification
check needs, for each in-text citation, the *sentence* in which the citation
appears, so we can later ask "does this cited paper actually support that
sentence?".

GROBID's `/api/processFulltextDocument` endpoint returns TEI XML containing
the full body of the article with `<ref target="#b<N>">` pointers wherever
a citation occurs in the running text. We walk the TEI tree, collect each
citation's enclosing sentence, and emit a list of `(ref_id, claim_sentence)`
tuples for downstream processing.

The implementation is intentionally simple: we treat the parent paragraph
as the sentence neighborhood and trim to the surrounding two sentences when
the paragraph is long. A more rigorous sentence splitter (spaCy / pysbd) is
future work; the current approach is good enough for the verifier to have
something to match against, and the verifier sees three sentences anyway.
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

from citecheck.extraction.grobid_client import DEFAULT_HOST, DEFAULT_TIMEOUT_S, GrobidError

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}


def _grobid_host() -> str:
    return os.environ.get("GROBID_HOST", DEFAULT_HOST).rstrip("/")


@retry(
    retry=retry_if_exception_type((httpx.RequestError, GrobidError)),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _post_fulltext(pdf_path: Path, *, host: str, timeout_s: float) -> str:
    """POST to /api/processFulltextDocument; return the TEI body."""
    url = f"{host}/api/processFulltextDocument"
    with pdf_path.open("rb") as fh:
        files = {"input": (pdf_path.name, fh, "application/pdf")}
        # consolidateCitations=0: the resolver already does this with our cache.
        # teiCoordinates not requested — claim extraction does not need them.
        resp = httpx.post(url, files=files, data={"consolidateCitations": "0"}, timeout=timeout_s)
    if resp.status_code >= 500:
        raise GrobidError(f"GROBID returned {resp.status_code}: {resp.text[:200]}")
    if resp.status_code != 200:
        raise GrobidError(f"GROBID returned {resp.status_code}: {resp.text[:200]}")
    return resp.text


# Splits on '. ' or '? ' or '! ' followed by a capital letter. Naive but
# enough for English-language scholarly prose. Avoids common abbreviations
# (e.g., 'Fig.', 'et al.') by requiring a space then uppercase.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return _SENTENCE_SPLIT.split(text) if text else []


def _para_text(p: ET.Element) -> str:
    """Concatenate the visible text of a <p>, replacing <ref> with [REF]."""
    parts: list[str] = []
    if p.text:
        parts.append(p.text)
    for child in p:
        if child.tag == f"{{{TEI_NS}}}ref":
            parts.append("[REF]")
        else:
            txt = "".join(child.itertext())
            if txt:
                parts.append(txt)
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def parse_fulltext(tei_xml: str) -> list[tuple[str, str]]:
    """From GROBID fulltext TEI, return [(ref_id, claim_sentence), ...].

    `ref_id` matches the xml:id of the corresponding <biblStruct> in the
    reference list (e.g. 'b0', 'b1'), so the caller can look up the cited
    work in the Reference list and then fetch its full text for verification.
    """
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        raise GrobidError(f"unparseable fulltext TEI: {exc}") from exc

    out: list[tuple[str, str]] = []
    # Body paragraphs live under text/body/div/p or just text/body/p.
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return out

    for p in body.iter(f"{{{TEI_NS}}}p"):
        # Collect (ref_id, position) pairs by walking children with text accumulator.
        # Then, when we know the full paragraph text, find the sentence each
        # ref falls in.
        refs: list[tuple[str, int]] = []  # (ref_id, char_offset_in_paragraph)
        cursor = 0
        if p.text:
            cursor += len(p.text)
        for child in p:
            if child.tag == f"{{{TEI_NS}}}ref":
                target = (child.get("target") or "").lstrip("#")
                if target:
                    refs.append((target, cursor))
                cursor += len("[REF]")
            else:
                cursor += len("".join(child.itertext()))
            if child.tail:
                cursor += len(child.tail)

        if not refs:
            continue

        para = _para_text(p)
        sentences = _split_sentences(para)
        if not sentences:
            continue

        # Map each ref's char offset to a sentence.
        sentence_offsets: list[tuple[int, int, str]] = []
        offset = 0
        for s in sentences:
            sentence_offsets.append((offset, offset + len(s), s))
            offset += len(s) + 1  # +1 for the space we collapsed

        for ref_id, char_pos in refs:
            picked = None
            for start, end, s in sentence_offsets:
                if start <= char_pos <= end:
                    picked = s
                    break
            if picked is None and sentence_offsets:
                picked = sentence_offsets[-1][2]
            if picked:
                # Clean up [REF] tokens in the rendered claim sentence.
                claim = re.sub(r"\s*\[REF\]\s*", " ", picked).strip()
                claim = re.sub(r"\s+", " ", claim)
                if claim:
                    out.append((ref_id, claim))
    return out


def extract_claims(
    pdf_path: str | Path, *, host: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> list[tuple[str, str]]:
    """Extract (ref_id, claim_sentence) tuples from the PDF body.

    The list may contain the same ref_id multiple times when a reference is
    cited in more than one sentence; downstream deduplicates per-reference
    when running the verifier.
    """
    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    tei = _post_fulltext(path, host=host or _grobid_host(), timeout_s=timeout_s)
    return parse_fulltext(tei)

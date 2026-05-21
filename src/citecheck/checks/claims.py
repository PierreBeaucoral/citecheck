"""Claim verification: does the cited paper actually support the in-text claim?

Phase 5. The check that no metadata-only layer can perform: it retrieves the
cited paper's text, finds the passages most semantically similar to the
claim sentence, and asks a local LLM whether the passages support the claim.

Pipeline per (claim_sentence, reference):

    1. fetch_pdf(reference.resolved_doi) via Unpaywall.
       Paywalled? -> ClaimCheck(status=UNVERIFIABLE).
    2. Read PDF text with pypdf.
    3. Chunk into ~500-token windows with 50-token overlap.
    4. Embed with BAAI/bge-small-en-v1.5 (free, CPU-only, ~90 MB model).
    5. Embed the claim sentence.
    6. Cosine-similarity retrieve top-5 chunks from the document.
    7. Format Ollama prompt: claim + chunks. Request JSON verdict.
    8. Parse + validate -> ClaimCheck.

Heavy dependencies (sentence-transformers, lancedb, ollama) live in the
`claims` optional install. The module top is import-light; the heavy
imports happen lazily inside each function so non-claim users do not pay
for them. If the deps are missing and the user calls this module's
entrypoint, they get a clear "uv sync --extra claims" message.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from citecheck.checks.cache import CacheStore
from citecheck.models import ClaimCheck, ClaimStatus, Reference
from citecheck.resolution.unpaywall import fetch_pdf

log = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = os.environ.get("CITECHECK_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 5


class ClaimsExtraNotInstalled(RuntimeError):
    """Raised when the heavy deps for Phase 5 are absent."""

    def __init__(self, missing: str) -> None:
        super().__init__(
            f"Phase 5 requires the '{missing}' package. Install it with:\n"
            "    uv sync --extra claims\n"
            "and ensure Ollama is running with:\n"
            "    ollama pull qwen2.5:7b-instruct"
        )


# --- PDF text extraction ----------------------------------------------------


def _read_pdf_text(pdf_path: Path) -> str:
    """Extract concatenated text from a PDF. Lazy import of pypdf."""
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ClaimsExtraNotInstalled("pypdf") from exc
    reader = PdfReader(str(pdf_path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


# --- Chunking ---------------------------------------------------------------


def _chunk_text(
    text: str, *, tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Approximate token chunking via whitespace splitting.

    Real tokenization belongs in the embedder, but for chunk boundaries
    whitespace is good enough — the embedder will retokenize internally.
    Chunks of ~500 tokens (~2000 chars) with 50-token overlap give good
    semantic-search recall on academic text.
    """
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= tokens:
        # Short text: one chunk, no sliding. Saves the sliding-window logic
        # from emitting near-duplicate chunks on small inputs (and makes the
        # function obvious to reason about in tests).
        return [" ".join(words)]
    out: list[str] = []
    i = 0
    step = max(1, tokens - overlap)
    while i < len(words):
        out.append(" ".join(words[i : i + tokens]))
        i += step
    return out


# --- Embeddings + retrieval -------------------------------------------------


def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ClaimsExtraNotInstalled("sentence-transformers") from exc
    return SentenceTransformer(DEFAULT_EMBED_MODEL)


def _retrieve_top_chunks(
    chunks: list[str], claim: str, *, embedder, top_k: int = RETRIEVAL_TOP_K
) -> list[str]:
    """Cosine-similarity retrieve top-k chunks for `claim`.

    We embed in-process and do the cosine math in numpy. A vector DB
    (lancedb) is overkill for the small chunk counts of a single paper;
    keeping it in-process avoids the on-disk index churn and lets the
    function be cleanly mocked in tests.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ClaimsExtraNotInstalled("numpy") from exc

    if not chunks:
        return []
    chunk_emb = embedder.encode(chunks, normalize_embeddings=True)
    claim_emb = embedder.encode([claim], normalize_embeddings=True)[0]
    sims = chunk_emb @ claim_emb
    top_idx = np.argsort(-sims)[:top_k]
    return [chunks[i] for i in top_idx]


# --- LLM verifier -----------------------------------------------------------


_PROMPT_TEMPLATE = """You are a citation auditor. The user is checking whether a cited paper supports a specific claim made about it in another paper.

CLAIM (from the citing paper):
{claim}

PASSAGES (from the cited paper; these are the chunks most semantically similar to the claim):
{passages}

TASK: Decide whether the passages above support the claim, paying close attention to qualifiers, quantifiers, and strength of language.

A claim is NOT supported when the citing paper inflates or drops a qualifier the cited paper actually uses. Common patterns of misrepresentation:

- The claim says "large" / "strong" / "dramatic" but the paper says "moderate" / "small" / "modest".
- The claim says "proved" / "established" / "showed" but the paper says "preliminary" / "suggests" / "may indicate".
- The claim says "all" / "every" / "always" but the paper says "most" / "many" / "in most cases".
- The claim asserts causation but the paper reports an association or correlation.
- The claim presents a finding as universal but the paper restricts it to a subgroup.
- The claim omits an exception the paper explicitly states (e.g., "no effect on X" where X was the one null result).
- The claim invents specific numbers (effect sizes, percentages, sample sizes) the paper does not report.

Procedure:

1. First, list any qualifier / quantifier / strength mismatches you detect between the claim and the passages, in the `discrepancies_found` array. Use one short sentence per mismatch. Use an empty array `[]` only when you have actively checked and found none.
2. Then set `supported`:
   - "yes" only if the passages clearly state the claim or a direct consequence AND `discrepancies_found` is empty.
   - "partial" if the passages are topically consistent with the claim but do not state it directly, AND `discrepancies_found` is empty.
   - "no" whenever `discrepancies_found` is non-empty, OR none of the passages support the claim.

Reply ONLY with this JSON object:

{{
  "discrepancies_found": ["short sentence per mismatch, or empty array"],
  "supported": "yes" | "partial" | "no",
  "quote": "<exact text from one passage that supports the claim, or null>",
  "confidence": "low" | "medium" | "high",
  "reasoning": "<one short sentence explaining your decision>"
}}

- "quote" must be EXACT text from the passages; if no good quote, use null.
- Reply with only the JSON. No prose before or after."""


def _call_ollama(
    prompt: str, *, model: str = DEFAULT_OLLAMA_MODEL, host: str = DEFAULT_OLLAMA_HOST
) -> str:
    try:
        import ollama  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ClaimsExtraNotInstalled("ollama") from exc
    client = ollama.Client(host=host)
    resp = client.generate(model=model, prompt=prompt, options={"temperature": 0.0})
    # ollama-python changed return type around 2025: older versions returned a
    # dict-like with `"response"`; newer versions return a `GenerateResponse`
    # pydantic object that exposes `.response` as an attribute.  Handle both
    # without depending on a specific version.
    if isinstance(resp, dict):
        return resp.get("response", "")
    if hasattr(resp, "response"):
        return getattr(resp, "response", "") or ""
    return str(resp)


def _parse_verdict(raw: str) -> dict[str, Any] | None:
    """Extract the JSON object from the LLM's reply. Returns None on parse failure."""
    raw = (raw or "").strip()
    # Models sometimes wrap JSON in ``` fences or add prose; try to be lenient.
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def _status_from_verdict(parsed: dict[str, Any]) -> ClaimStatus:
    s = (parsed.get("supported") or "").lower().strip()
    # Guard rail for the overstatement failure mode observed in the v1 Phase 5
    # eval: when the model fills `discrepancies_found` with a non-empty list
    # (i.e., it detected a qualifier/strength mismatch in its own reasoning)
    # but still sets `supported = yes`, force the verdict to NOT_SUPPORTED.
    # The reasoning/label misalignment was the cause of the 0.600 recall on
    # overstatement items; the prompt now explicitly instructs the model to
    # set supported = no whenever discrepancies are found, and this guard is
    # a safety net for models that follow the field convention but miss the
    # rule about how it should map to `supported`.
    discrepancies = parsed.get("discrepancies_found") or []
    if isinstance(discrepancies, list) and any(
        isinstance(d, str) and d.strip() for d in discrepancies
    ):
        return ClaimStatus.NOT_SUPPORTED
    if s == "yes":
        return ClaimStatus.SUPPORTED
    if s == "partial":
        return ClaimStatus.PARTIAL
    if s == "no":
        return ClaimStatus.NOT_SUPPORTED
    return ClaimStatus.ERROR


# --- Public entrypoint ------------------------------------------------------


def verify_claim(
    reference: Reference,
    claim_sentence: str,
    *,
    cache: CacheStore | None = None,
    embedder=None,
    ollama_call=_call_ollama,
    text: str | None = None,
) -> ClaimCheck:
    """Run the verification pipeline for one (reference, claim) pair.

    `embedder` and `ollama_call` are dependency-injected to make unit tests
    cheap: substitute a fake embedder and a function that returns a canned
    JSON string. The defaults wire up the real models when invoked from the
    CLI.

    `text`: when provided, the function skips the Unpaywall fetch + pypdf
    extraction and uses the supplied paper text directly.  Used by the Phase 5
    eval runner, which has its own text-acquisition step (Unpaywall PDF for
    DOIs, NCBI E-utilities for PMC IDs).
    """
    if text is not None:
        # Pre-supplied text path: skip the DOI/PDF acquisition entirely.
        if not text.strip():
            return ClaimCheck(
                status=ClaimStatus.UNVERIFIABLE,
                claim_sentence=claim_sentence,
                notes=["supplied paper text was empty"],
            )
    else:
        doi = (reference.resolved_doi or "").lower().strip()
        if not doi:
            return ClaimCheck(
                status=ClaimStatus.UNCHECKED,
                claim_sentence=claim_sentence,
                notes=["no resolved DOI"],
            )

        pdf_path = fetch_pdf(doi, cache=cache)
        if pdf_path is None:
            return ClaimCheck(
                status=ClaimStatus.UNVERIFIABLE,
                claim_sentence=claim_sentence,
                notes=["no OA PDF available via Unpaywall"],
            )

        try:
            text = _read_pdf_text(pdf_path)
        except (ClaimsExtraNotInstalled, RuntimeError) as exc:
            # Re-raise the install instruction; for other errors, surface as ERROR.
            if isinstance(exc, ClaimsExtraNotInstalled):
                raise
            return ClaimCheck(
                status=ClaimStatus.ERROR,
                claim_sentence=claim_sentence,
                notes=[f"PDF text extraction failed: {exc}"],
            )

    chunks = _chunk_text(text)
    if not chunks:
        return ClaimCheck(
            status=ClaimStatus.UNVERIFIABLE,
            claim_sentence=claim_sentence,
            notes=["PDF contained no extractable text"],
        )

    if embedder is None:
        embedder = _load_embedder()

    try:
        top_chunks = _retrieve_top_chunks(chunks, claim_sentence, embedder=embedder)
    except ClaimsExtraNotInstalled:
        raise
    except Exception as exc:
        log.warning("retrieval failed for %s: %s", doi, exc)
        return ClaimCheck(
            status=ClaimStatus.ERROR,
            claim_sentence=claim_sentence,
            notes=[f"retrieval failed: {exc}"],
        )

    prompt = _PROMPT_TEMPLATE.format(
        claim=claim_sentence,
        passages="\n\n---\n\n".join(top_chunks),
    )
    raw = ollama_call(prompt)
    parsed = _parse_verdict(raw)
    if parsed is None:
        return ClaimCheck(
            status=ClaimStatus.ERROR,
            claim_sentence=claim_sentence,
            notes=[f"verifier returned unparseable JSON: {raw[:80]!r}"],
        )

    return ClaimCheck(
        status=_status_from_verdict(parsed),
        claim_sentence=claim_sentence,
        quote=parsed.get("quote") or None,
        confidence=parsed.get("confidence") or None,
        reasoning=parsed.get("reasoning") or None,
    )


def check_claims(
    reference: Reference,
    claim_sentences: list[str],
    *,
    cache: CacheStore | None = None,
    embedder=None,
    ollama_call=_call_ollama,
    max_claims: int = 5,
) -> list[ClaimCheck]:
    """Verify multiple (reference, claim) pairs. Deduplicates and caps at max_claims."""
    if not claim_sentences:
        return []
    # Dedupe verbatim claims (a reference cited twice in the same sentence shows up twice).
    seen: set[str] = set()
    unique = []
    for s in claim_sentences:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)
        if len(unique) >= max_claims:
            break
    return [
        verify_claim(reference, c, cache=cache, embedder=embedder, ollama_call=ollama_call)
        for c in unique
    ]

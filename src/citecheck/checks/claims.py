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
# HuggingFace serverless Router (OpenAI-compatible). Used by the deployed
# web app on HF Spaces and by Phase-6a eval comparisons.  The Router auto-
# selects the best inference provider for the model id and supports the
# /v1/chat/completions schema; no extra SDK needed.
DEFAULT_HF_MODEL = os.environ.get("CITECHECK_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_HF_BASE_URL = os.environ.get(
    "CITECHECK_HF_BASE_URL", "https://router.huggingface.co/v1"
)

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


def _call_openai_compatible(
    prompt: str,
    *,
    model: str,
    base_url: str,
    token: str,
    provider_name: str = "OpenAI-compatible",
    timeout_s: float = 120.0,
    max_retries: int = 4,
    backoff_base_s: float = 8.0,
) -> str:
    """Call any OpenAI-compatible /chat/completions endpoint with backoff on 429.

    HuggingFace's serverless Router, Cerebras Cloud, Groq, Together AI,
    OpenRouter, and (obviously) OpenAI itself all accept the same chat-
    completions schema, so a single client suffices for all of them.  The
    distinguishing parameters are the base URL and the model id.

    On a 429 response, the function sleeps for `backoff_base_s * 2**attempt`
    seconds (exponential backoff) and retries up to `max_retries` times.
    This lets the eval runner saturate a provider's per-minute rate limit
    without manual throttling.  Returns the raw assistant `content` string
    so the existing `_parse_verdict` works unchanged.
    """
    import time

    import httpx

    if not token:
        raise RuntimeError(
            f"{provider_name} API token not set; cannot call the LLM provider."
        )
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # 1024 tokens is comfortably above what our JSON schema produces
        # (typically <300 tokens) and leaves headroom for verbose reasoning.
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    last_status = None
    last_body = ""
    for attempt in range(max_retries + 1):
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(url, headers=headers, json=body)
        if resp.status_code == 200:
            data = resp.json() or {}
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"{provider_name} API returned no choices: {data}")
            return (choices[0].get("message") or {}).get("content") or ""
        last_status = resp.status_code
        last_body = resp.text[:200]
        # Retry only on rate-limit / transient server errors; bail on auth /
        # bad-request / not-found / payment-required.
        if resp.status_code not in (429, 500, 502, 503, 504):
            break
        if attempt == max_retries:
            break
        sleep_s = backoff_base_s * (2 ** attempt)
        log.warning(
            "%s API %s on attempt %d/%d, sleeping %.1fs before retry",
            provider_name,
            resp.status_code,
            attempt + 1,
            max_retries + 1,
            sleep_s,
        )
        time.sleep(sleep_s)
    raise RuntimeError(
        f"{provider_name} API returned {last_status}: {last_body}"
    )


def _call_hf_chat(
    prompt: str,
    *,
    model: str = DEFAULT_HF_MODEL,
    base_url: str = DEFAULT_HF_BASE_URL,
    token: str | None = None,
    timeout_s: float = 120.0,
) -> str:
    """HuggingFace serverless Router wrapper around `_call_openai_compatible`."""
    return _call_openai_compatible(
        prompt,
        model=model,
        base_url=base_url,
        token=(token or os.environ.get("HF_TOKEN", "").strip()),
        provider_name="HF Inference",
        timeout_s=timeout_s,
    )


# Cerebras Cloud (OpenAI-compatible).  Free tier offers qwen-3-32b,
# llama-3.3-70b, and llama-3.1-8b with no monthly cap (30 req/min, 60k
# tokens/min) -- substantially more generous than HF's free Inference tier.
DEFAULT_CEREBRAS_MODEL = os.environ.get("CITECHECK_CEREBRAS_MODEL", "qwen-3-32b")
DEFAULT_CEREBRAS_BASE_URL = os.environ.get(
    "CITECHECK_CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"
)


def _call_cerebras(
    prompt: str,
    *,
    model: str = DEFAULT_CEREBRAS_MODEL,
    base_url: str = DEFAULT_CEREBRAS_BASE_URL,
    token: str | None = None,
    timeout_s: float = 120.0,
) -> str:
    """Cerebras Cloud wrapper around `_call_openai_compatible`."""
    return _call_openai_compatible(
        prompt,
        model=model,
        base_url=base_url,
        token=(token or os.environ.get("CEREBRAS_API_KEY", "").strip()),
        provider_name="Cerebras",
        timeout_s=timeout_s,
    )


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

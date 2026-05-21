"""Background pipeline runner for the citecheck web app.

The web endpoints kick off `run_pipeline(job_id, pdf_bytes, ...)` as a
FastAPI BackgroundTask.  The runner orchestrates the same pipeline
stages the CLI uses, updates `job_status` as it progresses, and writes
the final report JSON back into the job store.

This module is the thin layer that adapts citecheck's library code
(designed for batch CLI use) to the web app's needs:

* Status updates after each stage so the HTMX-polled job page shows
  meaningful progress.
* Pre-flight quota check before Phase 5; sample / truncate when the
  budget won't cover all claims.
* Graceful Phase 5 degradation: when the LLM provider exhausts quota
  mid-job, the partial claim results are kept and the report is
  delivered with a clear "Phase 5 partial" banner.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from citecheck.checks.cache import CacheStore
from citecheck.checks.hallucination import check_hallucination
from citecheck.checks.journal_quality import check_journal_quality
from citecheck.checks.retractions import check_retraction
from citecheck.web.quota import QuotaMonitor
from citecheck.web.settings import WebSettings
from citecheck.web.storage import JobStore

log = logging.getLogger(__name__)


def run_pipeline(
    job_id: str,
    pdf_bytes: bytes,
    *,
    phase5_enabled: bool,
    store: JobStore,
    quota: QuotaMonitor,
    settings: WebSettings,
) -> None:
    """Run the full citecheck pipeline against `pdf_bytes`; update job state.

    Catches all exceptions — anything raised here would otherwise be
    silently swallowed by FastAPI BackgroundTasks.  The job is moved to
    STATUS_ERROR with the message preserved.
    """
    try:
        _run_pipeline_inner(
            job_id=job_id,
            pdf_bytes=pdf_bytes,
            phase5_enabled=phase5_enabled,
            store=store,
            quota=quota,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 — entry point for background work
        log.exception("pipeline failed for job %s", job_id)
        store.set_error(job_id, f"{type(exc).__name__}: {exc}")


def _run_pipeline_inner(
    job_id: str,
    pdf_bytes: bytes,
    *,
    phase5_enabled: bool,
    store: JobStore,
    quota: QuotaMonitor,
    settings: WebSettings,
) -> None:
    # Stage 0: persist PDF to a tempfile so GROBID can read it.
    store.set_status(job_id, "running", progress_stage="upload", progress_text="Saving upload...")
    settings.ensure_directories()
    pdf_path = _write_temp_pdf(pdf_bytes, settings.upload_dir)
    cache = CacheStore()

    try:
        # Stage 1: extraction (GROBID).
        store.set_status(
            job_id,
            "running",
            progress_stage="extraction",
            progress_text="Extracting references via GROBID...",
        )
        references = _extract_references(pdf_path)
        n_refs = len(references)

        # Stage 2: resolution (Crossref + OpenAlex).
        store.set_status(
            job_id,
            "running",
            progress_stage="resolution",
            progress_text=f"Resolving {n_refs} references via Crossref + OpenAlex...",
        )
        references = _resolve_references(references)

        # Stages 3-5: retraction, fabrication, journal-quality (parallel-able
        # in principle but run sequentially here; each is fast at this scale).
        store.set_status(
            job_id,
            "running",
            progress_stage="metadata",
            progress_text=f"Running metadata checks (retraction, fabrication, journal quality) on {n_refs} references...",
        )
        per_ref_results = []
        for i, ref in enumerate(references, 1):
            retr = check_retraction(ref, cache=cache)
            hall = check_hallucination(ref, cache=cache)
            jq = check_journal_quality(ref, cache=cache)
            per_ref_results.append(
                {
                    "raw": _serialize_ref(ref),
                    "retraction": retr.model_dump(mode="json"),
                    "hallucination": hall.model_dump(mode="json"),
                    "journal_quality": jq.model_dump(mode="json"),
                    "claims": [],  # filled in by Phase 5 below
                }
            )
            if i % 5 == 0 or i == n_refs:
                store.set_status(
                    job_id,
                    "running",
                    progress_stage="metadata",
                    progress_text=f"Metadata checks: {i}/{n_refs} references done...",
                )

        # Stage 6: Phase 5 claim verification (opt-in).
        phase5_summary: dict[str, Any] = {
            "enabled": phase5_enabled,
            "claims_verified": 0,
            "claims_skipped_quota": 0,
            "claims_skipped_no_text": 0,
            "claims_skipped_cap": 0,
            "model": settings.llm_model,
            "provider": settings.llm_provider,
            "partial": False,
            "skipped_reason": None,
        }
        if phase5_enabled:
            _run_phase5(
                per_ref_results=per_ref_results,
                cache=cache,
                store=store,
                quota=quota,
                settings=settings,
                job_id=job_id,
                phase5_summary=phase5_summary,
            )
        else:
            phase5_summary["skipped_reason"] = "Claim verification was not requested for this run."

        # Persist the report.
        report = {
            "n_references": n_refs,
            "references": per_ref_results,
            "phase5": phase5_summary,
        }
        store.set_report(job_id, report)
    finally:
        cache.close()
        # Tempfile cleanup; not critical if it fails.
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_temp_pdf(pdf_bytes: bytes, upload_dir: Path) -> Path:
    """Write the bytes to a uniquely-named tempfile under upload_dir."""
    fd = tempfile.NamedTemporaryFile(
        dir=str(upload_dir), suffix=".pdf", delete=False
    )
    fd.write(pdf_bytes)
    fd.close()
    return Path(fd.name)


def _extract_references(pdf_path: Path) -> list:
    """Wrap the citecheck.pipeline extraction step.

    Imported lazily because GROBID + lxml are heavy and the test suite
    benefits from being able to mock the entrypoint without importing
    the full extraction stack at module load.
    """
    from citecheck.extraction.grobid_client import extract_references_from_pdf

    return extract_references_from_pdf(pdf_path)


def _resolve_references(references: list) -> list:
    """Run the existing batch resolver from citecheck.resolution."""
    from citecheck.resolution.crossref import resolve

    return [resolve(ref.raw) if hasattr(ref, "raw") else resolve(ref) for ref in references]


def _serialize_ref(ref) -> dict[str, Any]:
    """Reduce a Reference / RawReference object to a JSON-serializable dict."""
    if hasattr(ref, "model_dump"):
        return ref.model_dump(mode="json")
    if hasattr(ref, "__dict__"):
        return {k: v for k, v in ref.__dict__.items() if not k.startswith("_")}
    return {"repr": repr(ref)}


def _run_phase5(
    *,
    per_ref_results: list[dict],
    cache: CacheStore,
    store: JobStore,
    quota: QuotaMonitor,
    settings: WebSettings,
    job_id: str,
    phase5_summary: dict[str, Any],
) -> None:
    """Run claim verification for resolvable references, with quota checks.

    Skips references without a resolved DOI or PMC ID; caps the total
    claims processed at `settings.max_claims_per_pdf`; checks quota
    before each call and degrades gracefully when exhausted.
    """
    from citecheck.checks.claims import (
        _call_cerebras,
        _call_hf_chat,
        _call_ollama,
        _load_embedder,
        verify_claim,
    )
    from citecheck.resolution.unpaywall import fetch_text_any

    # Pick the LLM call function based on settings.llm_provider.
    if settings.llm_provider == "cerebras":
        def llm_call(prompt: str) -> str:
            return _call_cerebras(prompt, model=settings.llm_model)
    elif settings.llm_provider == "hf":
        def llm_call(prompt: str) -> str:
            return _call_hf_chat(prompt, model=settings.llm_model)
    else:
        def llm_call(prompt: str) -> str:
            return _call_ollama(prompt, model=settings.llm_model)

    # Identify candidate refs: have a resolved DOI/PMC and a clear text-claim.
    # The "claim" we verify is the citation context — in v1 we use the raw
    # citation string itself as a proxy; v1.1 will extract the in-text
    # sentence that cites this ref via GROBID's coordinates.
    candidates: list[tuple[int, dict, str]] = []
    for idx, r in enumerate(per_ref_results):
        raw_text = (r.get("raw") or {}).get("raw_text") or ""
        doi = (r.get("raw") or {}).get("doi") or ""
        if not raw_text or not doi:
            continue
        candidates.append((idx, r, raw_text))

    # Cap to max_claims_per_pdf to prevent monopolizing the daily budget.
    capped = candidates[: settings.max_claims_per_pdf]
    phase5_summary["claims_skipped_cap"] = max(0, len(candidates) - len(capped))

    # Pre-flight quota check.
    estimate = quota.estimate(settings.llm_provider, len(capped))
    if estimate.will_exhaust:
        # Reduce the in-scope set to what fits in remaining quota.
        if estimate.safe_claim_count == 0:
            phase5_summary["partial"] = True
            phase5_summary["skipped_reason"] = (
                f"{settings.llm_provider} quota exhausted for today; Phase 5 deferred until 00:00 UTC reset."
            )
            phase5_summary["claims_skipped_quota"] = len(capped)
            return
        capped = capped[: estimate.safe_claim_count]
        phase5_summary["partial"] = True
        phase5_summary["claims_skipped_quota"] = len(candidates) - len(capped)

    # Load embedder once (heavy ~30s on first use).
    store.set_status(
        job_id,
        "running",
        progress_stage="phase5",
        progress_text=f"Loading retrieval model (~30s) before verifying {len(capped)} claims...",
    )
    embedder = _load_embedder()

    # Iterate claims with progress updates.
    n_verified = 0
    for i, (idx, r, claim_text) in enumerate(capped, 1):
        store.set_status(
            job_id,
            "running",
            progress_stage="phase5",
            progress_text=f"Verifying claim {i}/{len(capped)}...",
        )
        doi = (r.get("raw") or {}).get("doi") or ""
        text, _provenance = fetch_text_any(doi=doi, cache=cache)
        if text is None:
            phase5_summary["claims_skipped_no_text"] += 1
            continue
        try:
            from citecheck.models import RawReference, Reference, ResolutionStatus

            ref_obj = Reference(
                raw=RawReference(raw_text=claim_text, doi=doi),
                status=ResolutionStatus.RESOLVED,
                resolved_doi=doi,
            )
            result = verify_claim(
                ref_obj,
                claim_text,
                cache=cache,
                embedder=embedder,
                ollama_call=llm_call,
                text=text,
            )
        except Exception as exc:  # noqa: BLE001 — must not crash the job
            log.warning("verify_claim failed for ref %d: %s", idx, exc)
            r["claims"].append({"status": "error", "reasoning": repr(exc)[:200]})
            # If the error looks like a quota-exhaustion 429, stop early.
            if "429" in repr(exc):
                phase5_summary["partial"] = True
                phase5_summary["claims_skipped_quota"] += len(capped) - i
                break
            continue

        # Best-effort token accounting: assume average per-call cost.
        # v1.1 will read actual token counts from response headers.
        from citecheck.web.quota import ESTIMATED_TOKENS_PER_CALL

        quota.record(
            settings.llm_provider,
            input_tokens=int(ESTIMATED_TOKENS_PER_CALL * 0.9),
            output_tokens=int(ESTIMATED_TOKENS_PER_CALL * 0.1),
        )
        r["claims"].append(result.model_dump(mode="json"))
        n_verified += 1

    phase5_summary["claims_verified"] = n_verified

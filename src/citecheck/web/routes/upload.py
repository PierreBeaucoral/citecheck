"""POST /api/check — PDF upload endpoint.

Accepts a multipart form with one PDF and an optional phase5 checkbox.
Creates a job row, kicks off a BackgroundTask, and 303-redirects the
browser to the job-status page.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse

from citecheck.web.quota import QuotaMonitor
from citecheck.web.settings import WebSettings
from citecheck.web.storage import JobStore
from citecheck.web.tasks import run_pipeline

router = APIRouter()


# 25 MB is the same cap the Unpaywall PDF downloader uses; we keep this
# in sync so anything we accept here can also be parsed downstream.
MAX_PDF_BYTES = 25 * 1024 * 1024


@router.post("/api/check")
async def upload_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    verify_claims: str | None = Form(None),
) -> RedirectResponse:
    """Accept one PDF; create a job; redirect to /jobs/{job_id}.

    `verify_claims` arrives as the string "on" when the checkbox is
    ticked, else None.  Anything truthy is treated as opt-in.
    """
    settings: WebSettings = request.app.state.settings
    store: JobStore = request.app.state.store
    quota: QuotaMonitor = request.app.state.quota

    content_type = (pdf.content_type or "").lower()
    if "pdf" not in content_type and not (pdf.filename or "").lower().endswith(".pdf"):
        return _error_redirect("Upload must be a PDF.")

    raw = await pdf.read()
    if len(raw) == 0:
        return _error_redirect("Uploaded PDF is empty.")
    if len(raw) > MAX_PDF_BYTES:
        return _error_redirect(
            f"PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB limit."
        )

    phase5_enabled = bool(verify_claims) and settings.enable_phase5_toggle

    job_id = store.create_job(pdf_bytes=raw, phase5_enabled=phase5_enabled)

    # FastAPI's BackgroundTasks runs after the response is sent.  We capture
    # the heavy parameters (settings, store, quota) by reference rather
    # than re-resolving them inside the task; this is intentional because
    # the request scope is gone by the time the task starts.
    background_tasks.add_task(
        run_pipeline,
        job_id=job_id,
        pdf_bytes=raw,
        phase5_enabled=phase5_enabled,
        store=store,
        quota=quota,
        settings=settings,
    )

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


def _error_redirect(message: str) -> RedirectResponse:
    """Redirect back to / with an error query-string for the form to render."""
    from urllib.parse import quote

    return RedirectResponse(url=f"/?error={quote(message)}", status_code=303)

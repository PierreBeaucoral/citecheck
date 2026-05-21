"""GET /jobs/{job_id} and /api/jobs/{job_id}/status.

The HTML route renders the job status page with an HTMX polling loop;
the JSON route is what HTMX polls every 3s.  When the job reaches
'done', the polled partial swaps in a link to the report page.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from citecheck.web.storage import STATUS_DONE, STATUS_ERROR, JobStore

router = APIRouter()


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_status_page(request: Request, job_id: str) -> HTMLResponse:
    """Render the HTMX-polling status page for a job."""
    store: JobStore = request.app.state.store
    templates: Jinja2Templates = request.app.state.templates
    job = store.get_job(job_id)
    if job is None:
        return HTMLResponse(
            content="<h1>Job not found</h1>", status_code=404
        )
    return templates.TemplateResponse(
        request=request,
        name="job_status.html",
        context={"job": job, "settings": request.app.state.settings},
    )


@router.get("/api/jobs/{job_id}/status")
async def job_status_partial(request: Request, job_id: str):
    """HTMX-target endpoint.  Returns an HTML partial reflecting current state.

    Returns:
      - 200 with an HTML partial (the polled target) while the job is
        running or pending.
      - 200 with a redirect-style partial when done, instructing HTMX to
        swap the whole page to the report URL.
      - 404 when the job ID is unknown.
    """
    store: JobStore = request.app.state.store
    templates: Jinja2Templates = request.app.state.templates
    job = store.get_job(job_id)
    if job is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return templates.TemplateResponse(
        request=request,
        name="_job_status_partial.html",
        context={
            "job": job,
            "done": job.status == STATUS_DONE,
            "error": job.status == STATUS_ERROR,
        },
    )

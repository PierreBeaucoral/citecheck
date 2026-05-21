"""GET /reports/{job_id} (HTML) and /api/reports/{job_id}.json.

The HTML route renders the report.html template against the stored
report JSON.  The JSON route returns the raw report dict for users who
want to consume the report programmatically.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from citecheck.web.storage import JobStore

router = APIRouter()


@router.get("/reports/{job_id}", response_class=HTMLResponse)
async def report_page(request: Request, job_id: str) -> HTMLResponse:
    """Render the rich-HTML view of a finished report."""
    store: JobStore = request.app.state.store
    templates: Jinja2Templates = request.app.state.templates
    job = store.get_job(job_id)
    if job is None:
        return HTMLResponse(content="<h1>Report not found</h1>", status_code=404)
    if job.report_json is None:
        return HTMLResponse(
            content="<h1>Report not ready</h1>"
            f"<p>Job is still {job.status}. Try <a href='/jobs/{job_id}'>the status page</a>.</p>",
            status_code=425,  # Too Early
        )
    report = store.report_dict(job_id) or {}
    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={"job": job, "report": report},
    )


@router.get("/api/reports/{job_id}.json")
async def report_json(request: Request, job_id: str):
    """Raw report JSON; suitable for `curl | jq`."""
    store: JobStore = request.app.state.store
    report = store.report_dict(job_id)
    if report is None:
        return JSONResponse({"error": "not found or not ready"}, status_code=404)
    return JSONResponse(report)

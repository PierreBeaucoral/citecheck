"""GET / — landing page with the upload form."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from citecheck.web.quota import QuotaMonitor
from citecheck.web.settings import WebSettings

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the upload form.

    Includes a live snapshot of the LLM provider's remaining quota so
    visitors see whether Phase 5 will run today or be deferred.  Also
    surfaces any `?error=...` query string from the upload route's
    validation rejections.
    """
    settings: WebSettings = request.app.state.settings
    quota: QuotaMonitor = request.app.state.quota
    templates: Jinja2Templates = request.app.state.templates

    error_message = request.query_params.get("error")
    quota_state = quota.state(settings.llm_provider)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "settings": settings,
            "quota": quota_state,
            "error_message": error_message,
        },
    )

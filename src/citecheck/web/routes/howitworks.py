"""GET /how — explainer page describing the citecheck pipeline.

A reader-facing tour of the six-stage pipeline + the four issue classes,
intended for a casual visitor who wants to know what citecheck is doing
to their PDF before they upload one.  Mirrors the structure of the
companion paper's §3 (pipeline) and §6 (results) without the academic
formality.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from citecheck.web.settings import WebSettings

router = APIRouter()


@router.get("/how", response_class=HTMLResponse)
async def how_it_works(request: Request) -> HTMLResponse:
    """Render the static explainer page.

    Static in the sense that the content is the same for every visitor;
    we still go through Jinja so the site_title / nav active-state work
    consistently.
    """
    settings: WebSettings = request.app.state.settings
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="how.html",
        context={"settings": settings},
    )

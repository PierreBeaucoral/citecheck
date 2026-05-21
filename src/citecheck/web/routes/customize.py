"""GET /customize — coffee-person customizer page.

A pure-frontend page; the server only serves the template.  Color
choices and toggles are stored in the visitor's localStorage by
`/static/mascot.js` and applied to every `.coffee-person` SVG on the
site.  No accounts, no server-side state.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from citecheck.web.settings import WebSettings

router = APIRouter()


@router.get("/customize", response_class=HTMLResponse)
async def customize(request: Request) -> HTMLResponse:
    settings: WebSettings = request.app.state.settings
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="customize.html",
        context={"settings": settings},
    )

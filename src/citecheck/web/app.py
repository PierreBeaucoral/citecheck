"""FastAPI app factory for the citecheck web demo.

Wires together: settings, SQLite job store, quota monitor, slowapi rate
limiter, Jinja2 templates, and the four route modules (index, upload,
jobs, reports) plus the auth-gated admin endpoint.

The app is exposed both as the module-level `app` (for uvicorn /
HuggingFace Spaces) and as a `create_app()` callable so tests can spin
up isolated instances with custom settings and tmpdir-backed storage.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from citecheck.web.quota import QuotaMonitor
from citecheck.web.routes import admin, index, jobs, reports, upload
from citecheck.web.settings import WebSettings, get_settings
from citecheck.web.storage import JobStore


def create_app(settings: WebSettings | None = None) -> FastAPI:
    """Build a fully wired FastAPI app.

    Pass an explicit `settings` for tests; otherwise loads from env.
    """
    settings = settings or get_settings()
    settings.ensure_directories()

    # Rate limiter — per-IP cap on uploads.
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[],
    )

    app = FastAPI(
        title=settings.site_title,
        description=settings.site_tagline,
        version="0.6.0",
    )
    app.state.settings = settings
    app.state.store = JobStore(settings.data_dir / "jobs.sqlite3")
    app.state.quota = QuotaMonitor(settings.data_dir / "quota.sqlite3")

    # Templates + static files.
    web_pkg_dir = Path(__file__).parent
    app.state.templates = Jinja2Templates(directory=str(web_pkg_dir / "templates"))
    app.mount(
        "/static",
        StaticFiles(directory=str(web_pkg_dir / "static")),
        name="static",
    )

    # slowapi wiring.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Apply the per-hour cap to the upload endpoint via slowapi's decorator
    # interface; we re-decorate at registration time so the cap reads from
    # settings.
    _apply_upload_rate_limit(upload.router, limiter, settings)

    # Mount routes.
    app.include_router(index.router)
    app.include_router(upload.router)
    app.include_router(jobs.router)
    app.include_router(reports.router)
    app.include_router(admin.router)

    # Friendly favicon shim so the browser stops 404'ing it.
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():  # noqa: D401
        return HTMLResponse(status_code=204)

    return app


def _apply_upload_rate_limit(router, limiter: Limiter, settings: WebSettings) -> None:
    """Wrap the POST /api/check endpoint with a per-IP rate limit.

    Done after `include_router` because slowapi expects the limit to
    apply to the bound endpoint; we look it up by name on the router.
    """
    rate_string = f"{settings.rate_limit_per_hour}/hour"
    for route in router.routes:
        if getattr(route, "name", None) == "upload_pdf":
            # slowapi expects a callable with .__wrapped__ semantics; we
            # use the decorator form directly on the endpoint.
            route.endpoint = limiter.limit(rate_string)(route.endpoint)


# Module-level app for `uvicorn citecheck.web.app:app` and HF Spaces.
app = create_app()

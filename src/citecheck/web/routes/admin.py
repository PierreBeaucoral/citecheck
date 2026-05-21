"""GET /admin/health — operator-only quota + jobs snapshot.

Auth: shared-secret header `X-Admin-Token` matching the admin_token
setting.  Setting admin_token to the empty string disables this
endpoint entirely (returns 404).  No public surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from citecheck.web.quota import QuotaMonitor
from citecheck.web.settings import WebSettings

router = APIRouter()


@router.get("/admin/health")
async def admin_health(request: Request):
    """Return live quota + recent-jobs snapshot for the operator.

    Hidden behind a shared-secret header to avoid leaking internal
    rate-limit numbers to public visitors (and to discourage scraping).
    Disabled when `settings.admin_token` is empty.
    """
    settings: WebSettings = request.app.state.settings
    quota: QuotaMonitor = request.app.state.quota

    if not settings.admin_token:
        return JSONResponse({"error": "admin endpoint disabled"}, status_code=404)
    if request.headers.get("X-Admin-Token") != settings.admin_token:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    snapshot = {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "quota": quota.to_health_dict(settings.llm_provider),
    }
    return JSONResponse(snapshot)

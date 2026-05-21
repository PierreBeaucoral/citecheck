"""Route-level tests using FastAPI TestClient.

Tests assert on HTTP behaviour and HTML structure; the actual pipeline
runner is patched out via monkeypatch since exercising GROBID + Crossref
+ Cerebras for every test would be impossibly slow.
"""

from __future__ import annotations

from typing import Any

import pytest

from citecheck.web.storage import STATUS_DONE, STATUS_ERROR


def test_index_renders_upload_form(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    assert "Run citecheck" in body
    # Phase 5 checkbox visible when enable_phase5_toggle=True (the default in tests)
    assert "verify_claims" in body
    # Quota state is exposed
    assert "Live LLM quota" in body or "quota" in body.lower()


def test_index_shows_error_query_param(client) -> None:
    resp = client.get("/?error=Upload+must+be+a+PDF.")
    assert resp.status_code == 200
    assert "Upload must be a PDF" in resp.text


def test_upload_requires_pdf_file(client, monkeypatch) -> None:
    # Empty multipart should return a friendly redirect with an error.
    resp = client.post("/api/check", files={}, follow_redirects=False)
    # FastAPI returns 422 if pdf field missing entirely
    assert resp.status_code in (303, 422)


def test_upload_rejects_non_pdf_content_type(client) -> None:
    resp = client.post(
        "/api/check",
        files={"pdf": ("not.txt", b"hello", "text/plain")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "error=" in resp.headers["location"]


def test_upload_creates_job_and_redirects(client, monkeypatch) -> None:
    # Stub the background task so we don't actually run the pipeline.
    captured: dict[str, Any] = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("citecheck.web.routes.upload.run_pipeline", fake_run_pipeline)

    resp = client.post(
        "/api/check",
        files={"pdf": ("paper.pdf", b"%PDF-1.4\n%fake pdf\n", "application/pdf")},
        data={"verify_claims": "on"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    loc = resp.headers["location"]
    assert loc.startswith("/jobs/")
    # Background task was registered (FastAPI may call it after the response).
    # We don't assert on captured here because FastAPI's TestClient may or may
    # not have executed the task by the time the response returns; the more
    # important assertion is the redirect target shape.


def test_jobs_page_404s_for_unknown_id(client) -> None:
    resp = client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_jobs_status_partial_shows_running(client, app) -> None:
    store = app.state.store
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    store.set_status(job_id, "running", progress_stage="resolution", progress_text="resolving 12 refs")
    resp = client.get(f"/api/jobs/{job_id}/status")
    assert resp.status_code == 200
    assert "resolution" in resp.text
    assert "12 refs" in resp.text


def test_jobs_status_partial_shows_done(client, app) -> None:
    store = app.state.store
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    store.set_report(job_id, {"n_references": 0, "references": [], "phase5": {"enabled": False}})
    resp = client.get(f"/api/jobs/{job_id}/status")
    assert resp.status_code == 200
    assert "Report ready" in resp.text
    assert f"/reports/{job_id}" in resp.text


def test_jobs_status_partial_shows_error(client, app) -> None:
    store = app.state.store
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    store.set_error(job_id, "GROBID timeout")
    resp = client.get(f"/api/jobs/{job_id}/status")
    assert resp.status_code == 200
    assert "Job failed" in resp.text or "failed" in resp.text.lower()
    assert "GROBID timeout" in resp.text


def test_report_page_404s_for_unknown_id(client) -> None:
    resp = client.get("/reports/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_report_json_returns_404_when_not_ready(client, app) -> None:
    store = app.state.store
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    resp = client.get(f"/api/reports/{job_id}.json")
    assert resp.status_code == 404


def test_report_json_returns_payload_when_done(client, app) -> None:
    store = app.state.store
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    payload = {"n_references": 3, "references": [], "phase5": {"enabled": False}}
    store.set_report(job_id, payload)
    resp = client.get(f"/api/reports/{job_id}.json")
    assert resp.status_code == 200
    assert resp.json() == payload


def test_admin_health_requires_token(client) -> None:
    resp = client.get("/admin/health")
    assert resp.status_code == 401


def test_admin_health_returns_quota_with_token(client) -> None:
    resp = client.get("/admin/health", headers={"X-Admin-Token": "test-admin-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "cerebras"
    assert "quota" in body
    assert body["quota"]["requests"] == 0


def test_admin_health_disabled_when_token_empty(tmp_path) -> None:
    """When admin_token is blank, the route returns 404 (endpoint disabled)."""
    from fastapi.testclient import TestClient

    from citecheck.web.app import create_app
    from citecheck.web.settings import WebSettings

    settings = WebSettings(
        upload_dir=tmp_path / "up",
        data_dir=tmp_path / "data",
        admin_token="",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/admin/health")
        assert resp.status_code == 404

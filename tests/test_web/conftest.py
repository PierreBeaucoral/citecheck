"""Shared fixtures for the web-app test suite.

Each test gets a fresh FastAPI app backed by a tmpdir, so SQLite stores
and uploads never leak between tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from citecheck.web.app import create_app
from citecheck.web.settings import WebSettings


@pytest.fixture
def settings(tmp_path: Path) -> WebSettings:
    """Tmpdir-backed settings with everything pinned to known values."""
    return WebSettings(
        upload_dir=tmp_path / "up",
        data_dir=tmp_path / "data",
        rate_limit_per_hour=5,
        max_claims_per_pdf=20,
        enable_phase5_toggle=True,
        phase5_default_on=False,
        llm_provider="cerebras",
        llm_model="qwen-3-235b-a22b-instruct-2507",
        admin_token="test-admin-token",
    )


@pytest.fixture
def app(settings: WebSettings):
    """Fully wired FastAPI app instance with tmpdir-backed storage."""
    return create_app(settings)


@pytest.fixture
def client(app):
    """FastAPI TestClient — synchronous, in-process."""
    with TestClient(app) as c:
        yield c

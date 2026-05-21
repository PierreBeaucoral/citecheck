"""Environment-variable-driven settings for the citecheck web app.

A single pydantic-settings model loads every knob that may differ between
local dev (Docker compose on developer machine), CI (test fixtures), and
deployed (HuggingFace Space, Oracle Cloud, etc.).  All defaults are
production-safe; nothing here requires manual configuration to boot.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSettings(BaseSettings):
    """Web-app-level configuration.

    Reads from environment variables prefixed `CITECHECK_WEB_` (so e.g.
    `CITECHECK_WEB_LLM_PROVIDER=cerebras` sets `llm_provider`).  Also
    reads a `.env` file in the repo root for local development.

    The LLM-provider knobs are also accessible via the generic env vars
    (`CEREBRAS_API_KEY`, `HF_TOKEN`, `OLLAMA_HOST`, `OLLAMA_MODEL`) so the
    CLI and the web app share the same configuration when run on the
    same machine.
    """

    model_config = SettingsConfigDict(
        env_prefix="CITECHECK_WEB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Deployment surface --------------------------------------------------
    # Site title shown on the upload page and in browser tabs.
    site_title: str = "citecheck"
    # Tagline shown on the upload page.
    site_tagline: str = "Scan academic PDFs for retracted, fabricated, predatory, and misrepresented citations."

    # --- Storage paths -------------------------------------------------------
    # Directory for uploaded PDFs (kept only while the job runs).
    upload_dir: Path = Path("/tmp/citecheck/uploads")
    # Directory for the job-store SQLite database.
    data_dir: Path = Path("/tmp/citecheck/data")

    # --- Rate limiting -------------------------------------------------------
    # PDFs per IP per hour.  5 is the documented public-demo cap.
    rate_limit_per_hour: int = 5
    # Hard cap on the number of claims any single PDF can submit to Phase 5,
    # regardless of how many references it has.  Keeps one big PDF from
    # monopolizing the daily LLM budget.
    max_claims_per_pdf: int = 20

    # --- Pipeline behaviour --------------------------------------------------
    # Whether Phase 5 (claim verification) is offered as an opt-in toggle
    # on the upload form.  Off in production v1 until we've confirmed
    # latency + quota behaviour under real traffic.
    enable_phase5_toggle: bool = True
    # Default state of the Phase 5 toggle when the form first loads.
    phase5_default_on: bool = False

    # --- LLM provider routing -----------------------------------------------
    # Provider name forwarded to claims.py at runtime.  One of
    # {ollama, cerebras, hf, anthropic}.  Cerebras is the v1 deploy default.
    llm_provider: str = "cerebras"
    # Provider-specific model id.  Filled in by the corresponding env var
    # below if unset.
    llm_model: str = "qwen-3-235b-a22b-instruct-2507"

    # --- Operator monitoring ------------------------------------------------
    # Shared-secret token for /admin/health.  Server returns 401 when the
    # request's X-Admin-Token header doesn't match.  Setting this to the
    # empty string disables the admin endpoint entirely.
    admin_token: str = ""

    def ensure_directories(self) -> None:
        """Create the configured directories if they don't already exist.

        Called once during app startup.  Tolerates missing parents.
        """
        for d in (self.upload_dir, self.data_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_settings() -> WebSettings:
    """Singleton accessor.  Importable from any module."""
    global _CACHED_SETTINGS
    if _CACHED_SETTINGS is None:
        _CACHED_SETTINGS = WebSettings()
    return _CACHED_SETTINGS


_CACHED_SETTINGS: WebSettings | None = None

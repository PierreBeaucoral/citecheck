"""SQLite-backed job store for the citecheck web app.

A single `jobs` table holds every uploaded PDF's processing state plus
the final report JSON.  The schema is intentionally minimal: no user
accounts, no per-job ACLs, no PII.  The `job_id` (uuid4) is the only
secret; reports are public-by-URL.

Why SQLite rather than a "real" database:
* The deployed-app traffic budget is on the order of tens of PDFs per
  day; concurrent writes are not the bottleneck.
* HuggingFace Spaces, Oracle Cloud Free, and local-dev docker compose
  all give us a writable filesystem; no external DB to provision.
* The whole store fits in a single file we can back up or wipe in one
  shell command.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Job:
    """In-memory view of a `jobs` row."""

    job_id: str
    created_at: int  # unix epoch seconds
    status: str  # 'pending' | 'running' | 'done' | 'error'
    pdf_sha256: str
    phase5_enabled: bool
    progress_stage: str | None  # short label e.g. 'resolution', 'phase5'
    progress_text: str | None  # human-readable, e.g. 'verified 3/18 claims'
    report_json: str | None
    error_message: str | None


# Status strings used in `jobs.status`.  Keep these in sync with the
# template + the routes; tests assert against them by name.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    created_at      INTEGER NOT NULL,
    status          TEXT NOT NULL,
    pdf_sha256      TEXT NOT NULL,
    phase5_enabled  INTEGER NOT NULL DEFAULT 0,
    progress_stage  TEXT,
    progress_text   TEXT,
    report_json     TEXT,
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs(created_at);
"""


class JobStore:
    """Thin wrapper around the `jobs` table.

    One JobStore instance per process is fine — SQLite handles internal
    locking and we always open short connections via the contextmanager.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- write -----------------------------------------------------------

    def create_job(
        self,
        *,
        pdf_bytes: bytes,
        phase5_enabled: bool,
    ) -> str:
        """Insert a new pending job; return its job_id."""
        job_id = str(uuid.uuid4())
        sha = hashlib.sha256(pdf_bytes).hexdigest()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, created_at, status, pdf_sha256, phase5_enabled) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, int(time.time()), STATUS_PENDING, sha, 1 if phase5_enabled else 0),
            )
        return job_id

    def set_status(
        self,
        job_id: str,
        status: str,
        *,
        progress_stage: str | None = None,
        progress_text: str | None = None,
    ) -> None:
        """Move a job to a new status, optionally with progress annotation."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress_stage = ?, progress_text = ? "
                "WHERE job_id = ?",
                (status, progress_stage, progress_text, job_id),
            )

    def set_report(self, job_id: str, report: dict[str, Any]) -> None:
        """Attach the final report JSON and mark the job done."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, report_json = ?, progress_stage = NULL, "
                "progress_text = NULL WHERE job_id = ?",
                (STATUS_DONE, json.dumps(report, default=str), job_id),
            )

    def set_error(self, job_id: str, message: str) -> None:
        """Mark a job as errored with a human-readable message."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error_message = ? WHERE job_id = ?",
                (STATUS_ERROR, message[:1000], job_id),
            )

    # ---- read ------------------------------------------------------------

    def get_job(self, job_id: str) -> Job | None:
        """Fetch a job by id; None when missing."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return Job(
            job_id=row["job_id"],
            created_at=row["created_at"],
            status=row["status"],
            pdf_sha256=row["pdf_sha256"],
            phase5_enabled=bool(row["phase5_enabled"]),
            progress_stage=row["progress_stage"],
            progress_text=row["progress_text"],
            report_json=row["report_json"],
            error_message=row["error_message"],
        )

    def report_dict(self, job_id: str) -> dict[str, Any] | None:
        """Convenience: return the report JSON parsed back into a dict."""
        job = self.get_job(job_id)
        if job is None or job.report_json is None:
            return None
        try:
            return json.loads(job.report_json)
        except json.JSONDecodeError:
            return None

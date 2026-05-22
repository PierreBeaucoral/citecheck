"""JobStore round-trip tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from citecheck.web.storage import (
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_PENDING,
    STATUS_RUNNING,
    JobStore,
)


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs.sqlite3")


def test_create_job_assigns_unique_ids(store: JobStore) -> None:
    a = store.create_job(pdf_bytes=b"%PDF-A", phase5_enabled=False)
    b = store.create_job(pdf_bytes=b"%PDF-B", phase5_enabled=True)
    assert a != b
    assert len(a) == 36  # UUIDv4 string format


def test_new_job_is_pending(store: JobStore) -> None:
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    job = store.get_job(job_id)
    assert job is not None
    assert job.status == STATUS_PENDING
    assert job.phase5_enabled is False
    assert job.report_json is None


def test_phase5_flag_persists(store: JobStore) -> None:
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=True)
    assert store.get_job(job_id).phase5_enabled is True


def test_status_transitions(store: JobStore) -> None:
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    store.set_status(
        job_id, STATUS_RUNNING, progress_stage="resolution", progress_text="resolving 12 refs"
    )
    job = store.get_job(job_id)
    assert job.status == STATUS_RUNNING
    assert job.progress_stage == "resolution"
    assert "12" in job.progress_text


def test_report_round_trip(store: JobStore) -> None:
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    report = {"n_references": 5, "references": [{"id": 1, "verdict": "clean"}]}
    store.set_report(job_id, report)
    job = store.get_job(job_id)
    assert job.status == STATUS_DONE
    assert store.report_dict(job_id) == report


def test_error_records_message(store: JobStore) -> None:
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    store.set_error(job_id, "GROBID timeout after 30s")
    job = store.get_job(job_id)
    assert job.status == STATUS_ERROR
    assert "GROBID" in job.error_message


def test_get_missing_job_returns_none(store: JobStore) -> None:
    assert store.get_job("does-not-exist") is None


def test_long_error_message_truncated(store: JobStore) -> None:
    job_id = store.create_job(pdf_bytes=b"%PDF-X", phase5_enabled=False)
    store.set_error(job_id, "x" * 5000)
    assert len(store.get_job(job_id).error_message) == 1000

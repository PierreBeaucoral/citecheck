"""Unit tests for citecheck.extraction.fulltext."""

from __future__ import annotations

import pytest

from citecheck.extraction.fulltext import parse_fulltext
from citecheck.extraction.grobid_client import GrobidError


def _tei(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
<text><body>{body}</body></text>
</TEI>"""


def test_single_citation_in_sentence() -> None:
    body = '<p>Citations are fabricated <ref target="#b0">Walters 2023</ref>. The rate is high.</p>'
    out = parse_fulltext(_tei(body))
    assert out == [("b0", "Citations are fabricated .")]


def test_multiple_citations_different_sentences() -> None:
    body = (
        '<p>Recent work shows X <ref target="#b0">A 2023</ref>. '
        'Buchanan et al. confirm <ref target="#b1">B 2024</ref> in economics.</p>'
    )
    out = parse_fulltext(_tei(body))
    refs = {r for r, _ in out}
    assert refs == {"b0", "b1"}


def test_no_citations_returns_empty() -> None:
    body = "<p>A paragraph without any citations at all.</p>"
    assert parse_fulltext(_tei(body)) == []


def test_ref_without_target_skipped() -> None:
    body = '<p>Stray ref <ref>X</ref> with no target attribute.</p>'
    assert parse_fulltext(_tei(body)) == []


def test_malformed_tei_raises() -> None:
    with pytest.raises(GrobidError):
        parse_fulltext("not xml")

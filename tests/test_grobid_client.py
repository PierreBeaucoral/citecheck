"""Unit tests for citecheck.extraction.grobid_client.

We test the TEI parser with synthetic XML — no network, no GROBID required.
"""

from __future__ import annotations

import httpx
import pytest

from citecheck.extraction.grobid_client import (
    GrobidError,
    extract_references,
    is_alive,
    parse_tei_references,
)


def _tei(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><back><div><listBibl>{body}</listBibl></div></back></text>
</TEI>"""


JOURNAL_REF = """
<biblStruct xml:id="b0">
  <analytic>
    <title level="a" type="main">The role of institutions in growth and development</title>
    <author><persName><forename type="first">Daron</forename><surname>Acemoglu</surname></persName></author>
    <author><persName><forename type="first">James</forename><surname>Robinson</surname></persName></author>
    <idno type="DOI">10.1257/aer.91.5.1369</idno>
  </analytic>
  <monogr>
    <title level="j">American Economic Review</title>
    <imprint>
      <biblScope unit="volume">91</biblScope>
      <biblScope unit="issue">5</biblScope>
      <biblScope unit="page" from="1369" to="1401"/>
      <date type="published" when="2001"/>
    </imprint>
  </monogr>
  <note type="raw_reference">Acemoglu, D., &amp; Robinson, J. (2001). The role of institutions...</note>
</biblStruct>
"""

BOOK_REF = """
<biblStruct xml:id="b1">
  <monogr>
    <title>Why Nations Fail</title>
    <author><persName><surname>Acemoglu</surname></persName></author>
    <imprint>
      <publisher>Crown</publisher>
      <date type="published" when="2012"/>
    </imprint>
  </monogr>
  <note type="raw_reference">Acemoglu, D. (2012). Why Nations Fail. Crown.</note>
</biblStruct>
"""

NO_DOI_REF = """
<biblStruct xml:id="b2">
  <analytic>
    <title level="a">Some paper without a DOI</title>
    <author><persName><surname>Smith</surname></persName></author>
  </analytic>
  <monogr>
    <imprint><date when="1985"/></imprint>
  </monogr>
  <note type="raw_reference">Smith, A. (1985). Some paper without a DOI.</note>
</biblStruct>
"""


class TestParseTeiReferences:
    def test_journal_article(self) -> None:
        refs = parse_tei_references(_tei(JOURNAL_REF))
        assert len(refs) == 1
        r = refs[0]
        assert r.ref_id == "b0"
        assert r.title == "The role of institutions in growth and development"
        assert [a.family for a in r.authors] == ["Acemoglu", "Robinson"]
        assert r.authors[0].given == "Daron"
        assert r.year == 2001
        assert r.journal == "American Economic Review"
        assert r.volume == "91"
        assert r.issue == "5"
        assert r.pages == "1369-1401"
        assert r.doi == "10.1257/aer.91.5.1369"
        assert "Acemoglu" in r.raw_text

    def test_book_falls_back_to_monogr_title(self) -> None:
        refs = parse_tei_references(_tei(BOOK_REF))
        assert len(refs) == 1
        r = refs[0]
        assert r.title == "Why Nations Fail"
        assert r.year == 2012
        assert r.doi is None

    def test_reference_without_doi(self) -> None:
        refs = parse_tei_references(_tei(NO_DOI_REF))
        assert refs[0].doi is None
        assert refs[0].title == "Some paper without a DOI"

    def test_multiple_references(self) -> None:
        refs = parse_tei_references(_tei(JOURNAL_REF + BOOK_REF + NO_DOI_REF))
        assert [r.ref_id for r in refs] == ["b0", "b1", "b2"]

    def test_empty_listbibl(self) -> None:
        assert parse_tei_references(_tei("")) == []

    def test_doi_fallback_from_raw_text(self) -> None:
        # GROBID did not extract <idno type="DOI"> but the raw citation text
        # contains a DOI inline — _parse_biblstruct should recover it.
        body = """
<biblStruct xml:id="b9">
  <analytic>
    <title level="a">Paper with DOI in the raw text only</title>
    <author><persName><surname>Brown</surname></persName></author>
  </analytic>
  <monogr><imprint><date when="2015"/></imprint></monogr>
  <note type="raw_reference">Brown, A. (2015). Paper... https://doi.org/10.1086/261876.</note>
</biblStruct>
"""
        refs = parse_tei_references(_tei(body))
        assert refs[0].doi == "10.1086/261876"

    def test_malformed_xml_raises(self) -> None:
        with pytest.raises(GrobidError):
            parse_tei_references("not xml at all")


class TestExtractReferencesHttp:
    """Uses the respx_mock fixture; @respx.mock on a class silently breaks pytest collection."""

    def test_posts_and_parses(self, tmp_path, respx_mock) -> None:
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")

        respx_mock.post("http://localhost:8070/api/processReferences").mock(
            return_value=httpx.Response(200, text=_tei(JOURNAL_REF))
        )

        refs = extract_references(pdf)
        assert len(refs) == 1
        assert refs[0].title.startswith("The role of institutions")

    def test_404_raises_grobid_error(self, tmp_path, respx_mock) -> None:
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        respx_mock.post("http://localhost:8070/api/processReferences").mock(
            return_value=httpx.Response(415, text="unsupported")
        )
        with pytest.raises(GrobidError):
            extract_references(pdf)

    def test_missing_pdf_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_references(tmp_path / "nope.pdf")


class TestIsAlive:
    def test_alive(self, respx_mock) -> None:
        respx_mock.get("http://localhost:8070/api/isalive").mock(
            return_value=httpx.Response(200, text="true")
        )
        assert is_alive() is True

    def test_dead(self, respx_mock) -> None:
        respx_mock.get("http://localhost:8070/api/isalive").mock(
            return_value=httpx.Response(503, text="false")
        )
        assert is_alive() is False

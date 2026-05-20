"""Integration tests for citecheck.cli — end-to-end with mocked HTTP."""

from __future__ import annotations

import json

import httpx
import respx
from typer.testing import CliRunner

from citecheck.cli import app

runner = CliRunner()

TEI_TWO_REFS = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><back><div><listBibl>
    <biblStruct xml:id="b0">
      <analytic>
        <title level="a" type="main">Paper A</title>
        <author><persName><surname>Doe</surname></persName></author>
        <idno type="DOI">10.1234/a</idno>
      </analytic>
      <monogr><imprint><date when="2020"/></imprint></monogr>
      <note type="raw_reference">Doe (2020). Paper A.</note>
    </biblStruct>
    <biblStruct xml:id="b1">
      <analytic>
        <title level="a" type="main">Paper B without DOI</title>
        <author><persName><surname>Roe</surname></persName></author>
      </analytic>
      <monogr><imprint><date when="2019"/></imprint></monogr>
      <note type="raw_reference">Roe (2019). Paper B.</note>
    </biblStruct>
  </listBibl></div></back></text>
</TEI>"""


def _doi_response(doi: str, title: str, year: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "ok",
            "message": {
                "DOI": doi,
                "title": [title],
                "issued": {"date-parts": [[year]]},
                "container-title": ["J"],
                "type": "journal-article",
            },
        },
    )


@respx.mock
def test_extract_table_output(tmp_path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    respx.post("http://localhost:8070/api/processReferences").mock(
        return_value=httpx.Response(200, text=TEI_TWO_REFS)
    )
    respx.get("https://api.crossref.org/works/10.1234/a").mock(
        return_value=_doi_response("10.1234/a", "Paper A", 2020)
    )
    respx.get("https://api.crossref.org/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "ok",
                "message": {
                    "items": [
                        {
                            "DOI": "10.1234/b",
                            "title": ["Paper B without DOI"],
                            "issued": {"date-parts": [[2019]]},
                            "container-title": ["J"],
                            "type": "journal-article",
                        }
                    ]
                },
            },
        )
    )
    # OpenAlex mock is unused here (Crossref metadata succeeds for both refs)
    # but the route must exist so unmocked-call assertions don't fire if Crossref
    # ever returns a non-resolved status during refactors.
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    result = runner.invoke(app, ["extract", str(pdf), "--skip-liveness"])
    assert result.exit_code == 0, result.stdout
    # Two references in the table; one resolved by DOI, one by metadata.
    assert "Paper A" in result.stdout
    assert "Paper B without DOI" in result.stdout
    assert "resolved" in result.stdout


@respx.mock
def test_extract_json_output(tmp_path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    respx.post("http://localhost:8070/api/processReferences").mock(
        return_value=httpx.Response(200, text=TEI_TWO_REFS)
    )
    respx.get("https://api.crossref.org/works/10.1234/a").mock(
        return_value=_doi_response("10.1234/a", "Paper A", 2020)
    )
    respx.get("https://api.crossref.org/works").mock(
        return_value=httpx.Response(200, json={"status": "ok", "message": {"items": []}})
    )
    # Crossref returned no metadata candidates -> dispatch falls back to OpenAlex.
    # Mock it to also return nothing so the reference ends UNRESOLVED.
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )

    result = runner.invoke(app, ["extract", str(pdf), "--json", "--skip-liveness"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert len(payload) == 2
    assert payload[0]["resolved_doi"] == "10.1234/a"
    assert payload[0]["status"] == "resolved"
    assert payload[1]["status"] == "unresolved"


def test_extract_missing_file_exits(tmp_path) -> None:
    result = runner.invoke(app, ["extract", str(tmp_path / "nope.pdf"), "--skip-liveness"])
    assert result.exit_code == 2
    # typer.secho(err=True) writes to stderr; CliRunner exposes it via .stderr
    # when mix_stderr is False (the click 8.2+ default).
    combined = (result.stdout or "") + (result.stderr or "")
    assert "not found" in combined.lower()

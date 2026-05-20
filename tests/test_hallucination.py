"""Unit tests for citecheck.checks.hallucination — five layers + aggregator."""

from __future__ import annotations

import httpx

from citecheck.checks.hallucination import (
    _aggregate,
    _downgrade,
    _layer1_doi_integrity,
    _layer2_cross_db,
    _low_coverage_caveats,
    check_hallucination,
)
from citecheck.models import (
    Author,
    HallucinationVerdict,
    LayerSignal,
    RawReference,
    Reference,
    ResolutionStatus,
)


def _ref(
    *,
    title: str | None = "A real paper",
    doi: str | None = None,
    year: int | None = 2020,
    authors: list[Author] | None = None,
    journal: str | None = "Important Journal",
    status: ResolutionStatus = ResolutionStatus.RESOLVED,
    resolved_doi: str | None = None,
    resolved_metadata: dict | None = None,
) -> Reference:
    return Reference(
        raw=RawReference(
            raw_text="x",
            title=title,
            doi=doi,
            year=year,
            authors=authors if authors is not None else [Author(family="Smith", given="Jane")],
            journal=journal,
        ),
        status=status,
        resolved_doi=resolved_doi,
        resolved_metadata=resolved_metadata,
    )


# ----- Layer 1 ----------------------------------------------------------------


class TestLayer1DoiIntegrity:
    def test_no_doi_not_flagged(self) -> None:
        s = _layer1_doi_integrity(_ref(doi=None))
        assert s.flagged is False

    def test_resolved_metadata_matches(self) -> None:
        s = _layer1_doi_integrity(
            _ref(
                title="On the role of institutions in growth",
                doi="10.1234/abc",
                resolved_doi="10.1234/abc",
                status=ResolutionStatus.RESOLVED,
                resolved_metadata={
                    "title": "On the role of institutions in growth",
                    "issued_year": 2020,
                },
            )
        )
        assert s.flagged is False
        assert s.score is not None and s.score > 0.9

    def test_title_diverges_flagged(self) -> None:
        s = _layer1_doi_integrity(
            _ref(
                title="On the quantum properties of bananas",
                doi="10.1234/abc",
                resolved_doi="10.1234/abc",
                status=ResolutionStatus.RESOLVED,
                resolved_metadata={
                    "title": "A completely different topic in chemistry",
                    "issued_year": 2020,
                },
            )
        )
        assert s.flagged is True
        assert "diverges" in s.reasoning

    def test_year_far_off_flagged(self) -> None:
        s = _layer1_doi_integrity(
            _ref(
                title="Same title",
                doi="10.1234/abc",
                year=2020,
                resolved_doi="10.1234/abc",
                status=ResolutionStatus.RESOLVED,
                resolved_metadata={"title": "Same title", "issued_year": 2010},
            )
        )
        assert s.flagged is True


# ----- Layer 2 ----------------------------------------------------------------


class TestLayer2CrossDb:
    def test_resolved_not_flagged(self) -> None:
        assert _layer2_cross_db(_ref(status=ResolutionStatus.RESOLVED)).flagged is False

    def test_ambiguous_not_flagged(self) -> None:
        assert _layer2_cross_db(_ref(status=ResolutionStatus.AMBIGUOUS)).flagged is False

    def test_unresolved_flagged(self) -> None:
        assert _layer2_cross_db(_ref(status=ResolutionStatus.UNRESOLVED)).flagged is True


# ----- Layer 3 (network) ------------------------------------------------------


class TestLayer3Authors:
    @staticmethod
    def _mock_sources_known(respx_mock) -> None:
        """Each L3 test runs check_hallucination, which also fires L4. Stub the
        venue lookup so it returns a hit and L4 stays out of the flagged count."""
        respx_mock.get("https://api.openalex.org/sources").mock(
            return_value=httpx.Response(
                200, json={"results": [{"display_name": "Important Journal"}]}
            )
        )

    def test_both_authors_found(self, respx_mock) -> None:
        self._mock_sources_known(respx_mock)
        ref = _ref(
            authors=[
                Author(family="Acemoglu", given="Daron"),
                Author(family="Robinson", given="James"),
            ],
            status=ResolutionStatus.RESOLVED,
        )
        respx_mock.get("https://api.openalex.org/authors").mock(
            return_value=httpx.Response(200, json={"results": [{"display_name": "Daron Acemoglu"}]})
        )
        result = check_hallucination(ref)
        layer_states = {s.layer: s.flagged for s in result.signals}
        assert layer_states["L3_authors"] is False

    def test_neither_author_found(self, respx_mock) -> None:
        self._mock_sources_known(respx_mock)
        ref = _ref(
            authors=[
                Author(family="Anthropic", given="Q"),
                Author(family="Imaginarius", given="Z"),
            ],
            status=ResolutionStatus.RESOLVED,
        )
        respx_mock.get("https://api.openalex.org/authors").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        result = check_hallucination(ref)
        layer_states = {s.layer: s.flagged for s in result.signals}
        assert layer_states["L3_authors"] is True

    def test_partial_match_not_flagged(self, respx_mock) -> None:
        self._mock_sources_known(respx_mock)
        ref = _ref(
            authors=[Author(family="Acemoglu"), Author(family="Imaginarius")],
            status=ResolutionStatus.RESOLVED,
        )

        def _resolver(request):
            search = request.url.params.get("search", "")
            if "Acemoglu" in search:
                return httpx.Response(200, json={"results": [{"display_name": "Daron Acemoglu"}]})
            return httpx.Response(200, json={"results": []})

        respx_mock.get("https://api.openalex.org/authors").mock(side_effect=_resolver)
        result = check_hallucination(ref)
        layer_states = {s.layer: s.flagged for s in result.signals}
        assert layer_states["L3_authors"] is False


# ----- Layer 4 (network) ------------------------------------------------------


class TestLayer4Venue:
    def test_known_venue_not_flagged(self, respx_mock) -> None:
        ref = _ref(journal="Nature", status=ResolutionStatus.RESOLVED)
        respx_mock.get("https://api.openalex.org/authors").mock(
            return_value=httpx.Response(200, json={"results": [{"display_name": "Jane Smith"}]})
        )
        respx_mock.get("https://api.openalex.org/sources").mock(
            return_value=httpx.Response(200, json={"results": [{"display_name": "Nature"}]})
        )
        result = check_hallucination(ref)
        layer_states = {s.layer: s.flagged for s in result.signals}
        assert layer_states["L4_venue"] is False

    def test_unknown_venue_flagged(self, respx_mock) -> None:
        ref = _ref(journal="Journal of Imaginary Studies", status=ResolutionStatus.RESOLVED)
        respx_mock.get("https://api.openalex.org/authors").mock(
            return_value=httpx.Response(200, json={"results": [{"display_name": "Jane Smith"}]})
        )
        respx_mock.get("https://api.openalex.org/sources").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        result = check_hallucination(ref)
        layer_states = {s.layer: s.flagged for s in result.signals}
        assert layer_states["L4_venue"] is True

    def test_no_journal_layer_inapplicable(self, respx_mock) -> None:
        ref = _ref(journal=None, status=ResolutionStatus.RESOLVED)
        respx_mock.get("https://api.openalex.org/authors").mock(
            return_value=httpx.Response(200, json={"results": [{"display_name": "Jane Smith"}]})
        )
        result = check_hallucination(ref)
        layer_states = {s.layer: s.flagged for s in result.signals}
        assert layer_states["L4_venue"] is False


# ----- Aggregator -------------------------------------------------------------


def _signals(*flag_layers: str) -> list[LayerSignal]:
    """Build a 4-signal list (L1-L4) flagging only the named layers."""
    out = []
    for name in ["L1_doi_integrity", "L2_cross_db", "L3_authors", "L4_venue"]:
        out.append(LayerSignal(layer=name, flagged=name in flag_layers, reasoning=""))
    return out


class TestAggregate:
    def test_zero_flags_with_doi_resolution_is_high_confidence(self) -> None:
        ref = _ref(
            doi="10.1234/x",
            resolved_doi="10.1234/x",
            status=ResolutionStatus.RESOLVED,
            year=2020,
        )
        result = _aggregate(ref, _signals())
        assert result.verdict == HallucinationVerdict.REAL_HIGH_CONFIDENCE
        assert result.red_flag_count == 0

    def test_zero_flags_without_doi_is_low_confidence(self) -> None:
        ref = _ref(doi=None, status=ResolutionStatus.RESOLVED, year=2020)
        result = _aggregate(ref, _signals())
        assert result.verdict == HallucinationVerdict.REAL_LOW_CONFIDENCE

    def test_two_flags_is_suspicious(self) -> None:
        ref = _ref(status=ResolutionStatus.UNRESOLVED, year=2020)
        result = _aggregate(ref, _signals("L2_cross_db", "L3_authors"))
        assert result.verdict == HallucinationVerdict.SUSPICIOUS

    def test_four_flags_is_likely_hallucinated(self) -> None:
        ref = _ref(status=ResolutionStatus.UNRESOLVED, year=2020)
        result = _aggregate(
            ref, _signals("L1_doi_integrity", "L2_cross_db", "L3_authors", "L4_venue")
        )
        assert result.verdict == HallucinationVerdict.LIKELY_HALLUCINATED


class TestLowCoverageCaveats:
    def test_pre_2000_paper_caveat(self) -> None:
        ref = _ref(year=1985, status=ResolutionStatus.RESOLVED)
        assert any("pre-2000" in c for c in _low_coverage_caveats(ref))

    def test_no_journal_caveat(self) -> None:
        ref = _ref(journal=None, status=ResolutionStatus.RESOLVED)
        assert any("no journal" in c for c in _low_coverage_caveats(ref))

    def test_caveat_downgrades_real_high(self) -> None:
        # Use a valid DOI registrant (>=4 digits). 10.1/x normalizes to None per
        # the strict Crossref pattern and would silently flip the test scenario.
        ref = _ref(
            year=1985, doi="10.1234/x", resolved_doi="10.1234/x", status=ResolutionStatus.RESOLVED
        )
        result = _aggregate(ref, _signals())
        # Pre-2000 + would-be high-confidence -> downgraded to low-confidence.
        assert result.verdict == HallucinationVerdict.REAL_LOW_CONFIDENCE
        assert result.caveats

    def test_caveat_does_not_change_suspicious(self) -> None:
        ref = _ref(year=1985, status=ResolutionStatus.UNRESOLVED)
        result = _aggregate(ref, _signals("L2_cross_db", "L3_authors"))
        # Caveats only downgrade real_high; suspicious stays suspicious.
        assert result.verdict == HallucinationVerdict.SUSPICIOUS

    def test_caveat_does_not_escalate_real_low(self) -> None:
        # A book reference (no journal field) with one cross-DB miss would be
        # real_low_confidence under the rule alone. The no-journal caveat must
        # NOT push it into 'suspicious' — that was the false-positive trigger
        # for the Neumark/Wascher case on the Callaway/Sant'Anna fixture.
        ref = _ref(
            doi=None,
            journal=None,
            year=2008,
            status=ResolutionStatus.UNRESOLVED,
        )
        result = _aggregate(ref, _signals("L2_cross_db"))
        assert result.verdict == HallucinationVerdict.REAL_LOW_CONFIDENCE
        assert result.caveats  # the caveat was recorded but didn't escalate the verdict


class TestDowngrade:
    def test_each_step(self) -> None:
        assert (
            _downgrade(HallucinationVerdict.REAL_HIGH_CONFIDENCE)
            == HallucinationVerdict.REAL_LOW_CONFIDENCE
        )
        assert (
            _downgrade(HallucinationVerdict.REAL_LOW_CONFIDENCE) == HallucinationVerdict.SUSPICIOUS
        )
        assert (
            _downgrade(HallucinationVerdict.SUSPICIOUS) == HallucinationVerdict.LIKELY_HALLUCINATED
        )
        assert (
            _downgrade(HallucinationVerdict.LIKELY_HALLUCINATED)
            == HallucinationVerdict.LIKELY_HALLUCINATED
        )


# ----- End-to-end with all layers ---------------------------------------------


class TestEndToEnd:
    def test_likely_hallucinated_e2e(self, respx_mock) -> None:
        # Crafted fake: real-sounding title, fake DOI that "resolved" to a
        # paper with a completely unrelated title (mocked); both authors absent
        # in OpenAlex; venue absent. Three+ red flags -> likely_hallucinated
        # (or suspicious after caveat downgrade).
        ref = _ref(
            title="On the quantum economics of imaginary derivatives",
            doi="10.9999/fake",
            year=2024,
            authors=[
                Author(family="Anthropic", given="Q"),
                Author(family="Imaginarius", given="Z"),
            ],
            journal="Journal of Imaginary Studies",
            status=ResolutionStatus.RESOLVED,
            resolved_doi="10.9999/fake",
            resolved_metadata={
                "title": "Statistical methods for protein folding",
                "issued_year": 2024,
            },
        )
        respx_mock.get("https://api.openalex.org/authors").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        respx_mock.get("https://api.openalex.org/sources").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        result = check_hallucination(ref)
        assert result.verdict in (
            HallucinationVerdict.SUSPICIOUS,
            HallucinationVerdict.LIKELY_HALLUCINATED,
        )
        assert result.red_flag_count >= 3

    def test_skip_network_layers(self) -> None:
        # When network layers are skipped, L3 + L4 record skipped reasoning
        # and never count as red flags. No HTTP mock is needed.
        ref = _ref(status=ResolutionStatus.UNRESOLVED, year=2020)
        result = check_hallucination(ref, skip_network_layers=True)
        # Only L2 (UNRESOLVED) is flagged out of the 4 layers.
        assert result.red_flag_count == 1
        assert result.verdict == HallucinationVerdict.REAL_LOW_CONFIDENCE

"""Unit tests for citecheck.checks.claims.

Heavy deps (sentence-transformers, ollama) are dependency-injected on the
public functions, so we mock them with simple fakes and don't need the
`claims` optional extra installed in CI.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from citecheck.checks.claims import (
    _chunk_text,
    _parse_verdict,
    _retrieve_top_chunks,
    _status_from_verdict,
    check_claims,
    verify_claim,
)
from citecheck.models import (
    ClaimStatus,
    RawReference,
    Reference,
    ResolutionStatus,
)

# ---- chunking ---------------------------------------------------------------


class TestChunking:
    def test_short_text_one_chunk(self) -> None:
        out = _chunk_text("a b c d e", tokens=10)
        assert out == ["a b c d e"]

    def test_overlap_correct(self) -> None:
        out = _chunk_text("a b c d e f g h", tokens=4, overlap=2)
        # step = 4 - 2 = 2, so windows start at 0, 2, 4, 6
        assert out == ["a b c d", "c d e f", "e f g h", "g h"]

    def test_empty_returns_empty(self) -> None:
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []


# ---- JSON parsing ------------------------------------------------------------


class TestVerdictParsing:
    def test_clean_json(self) -> None:
        v = _parse_verdict('{"supported":"yes","quote":"x","confidence":"high","reasoning":"r"}')
        assert v == {"supported": "yes", "quote": "x", "confidence": "high", "reasoning": "r"}

    def test_json_with_prose_around(self) -> None:
        raw = 'Sure! Here is the JSON: {"supported":"no","quote":null,"confidence":"low","reasoning":"r"} done'
        v = _parse_verdict(raw)
        assert v is not None
        assert v["supported"] == "no"

    def test_json_with_code_fences(self) -> None:
        raw = '```json\n{"supported":"partial","quote":"y","confidence":"medium","reasoning":"r"}\n```'
        v = _parse_verdict(raw)
        assert v is not None
        assert v["supported"] == "partial"

    def test_unparseable_returns_none(self) -> None:
        assert _parse_verdict("no json here at all") is None
        assert _parse_verdict("") is None

    def test_status_from_verdict(self) -> None:
        assert _status_from_verdict({"supported": "yes"}) == ClaimStatus.SUPPORTED
        assert _status_from_verdict({"supported": "partial"}) == ClaimStatus.PARTIAL
        assert _status_from_verdict({"supported": "no"}) == ClaimStatus.NOT_SUPPORTED
        assert _status_from_verdict({"supported": "weird"}) == ClaimStatus.ERROR


# ---- retrieval --------------------------------------------------------------


class _FakeEmbedder:
    """Cosine-similarity-friendly fake embedder for tests.

    `mapping` is a list of (substring_marker, embedding) pairs. The embedder
    looks for any substring marker in the input text and returns the matching
    embedding. Unknown text gets `default` (or a zero vector). This is lenient
    enough to handle whatever exact chunk text the production code produces,
    while still being deterministic.
    """

    def __init__(
        self,
        mapping: dict[str, list[float]],
        *,
        default: list[float] | None = None,
    ) -> None:
        self.mapping = mapping
        self.default = default

    def _embed_one(self, text: str):
        import numpy as np

        for marker, vec in self.mapping.items():
            if marker in text:
                return np.array(vec, dtype=float)
        if self.default is not None:
            return np.array(self.default, dtype=float)
        # Fall back to zero vector — caller's cosine sim will be 0.
        return np.zeros(len(next(iter(self.mapping.values()))), dtype=float)

    def encode(self, texts, normalize_embeddings: bool = False):
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        if isinstance(texts, str):
            return self._embed_one(texts)
        return np.array([self._embed_one(t) for t in texts], dtype=float)


class TestRetrieval:
    def test_top_k_returns_most_similar(self) -> None:
        np = pytest.importorskip("numpy")  # noqa: F841
        chunks = ["alpha", "beta", "gamma"]
        emb = _FakeEmbedder(
            {
                "alpha": [1.0, 0.0],
                "beta": [0.0, 1.0],
                "gamma": [0.7, 0.7],
                "query": [0.95, 0.05],  # closest to alpha, then gamma
            }
        )
        out = _retrieve_top_chunks(chunks, "query", embedder=emb, top_k=2)
        assert out[0] == "alpha"
        assert out[1] == "gamma"


# ---- verify_claim end-to-end (mocks all heavy bits) -------------------------


def _resolved(doi: str = "10.1234/abc") -> Reference:
    return Reference(
        raw=RawReference(raw_text="x", title="A real paper", doi=doi),
        status=ResolutionStatus.RESOLVED,
        resolved_doi=doi,
        resolved_metadata={"title": "A real paper", "issued_year": 2020},
    )


class TestVerifyClaim:
    def test_no_doi_returns_unchecked(self) -> None:
        ref = Reference(raw=RawReference(raw_text="x"), status=ResolutionStatus.UNRESOLVED)
        result = verify_claim(ref, "any claim")
        assert result.status == ClaimStatus.UNCHECKED

    def test_paywalled_returns_unverifiable(self) -> None:
        ref = _resolved()
        with patch("citecheck.checks.claims.fetch_pdf", return_value=None):
            result = verify_claim(ref, "any claim")
        assert result.status == ClaimStatus.UNVERIFIABLE

    def test_supported_verdict_from_ollama(self, tmp_path) -> None:
        pytest.importorskip("numpy")
        ref = _resolved()
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        embedder = _FakeEmbedder(
            {
                "chunk one about institutions": [1.0, 0.0],
                "chunk two about cooking": [0.0, 1.0],
                "Institutions matter for growth": [0.95, 0.05],
            }
        )

        def fake_ollama(prompt: str) -> str:
            return (
                '{"supported":"yes","quote":"chunk one about institutions",'
                '"confidence":"high","reasoning":"matches"}'
            )

        with (
            patch("citecheck.checks.claims.fetch_pdf", return_value=fake_pdf),
            patch(
                "citecheck.checks.claims._read_pdf_text",
                return_value="chunk one about institutions. chunk two about cooking.",
            ),
        ):
            result = verify_claim(
                ref,
                "Institutions matter for growth",
                embedder=embedder,
                ollama_call=fake_ollama,
            )
        assert result.status == ClaimStatus.SUPPORTED
        assert result.quote == "chunk one about institutions"
        assert result.confidence == "high"

    def test_not_supported_verdict(self, tmp_path) -> None:
        pytest.importorskip("numpy")
        ref = _resolved()
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        embedder = _FakeEmbedder(
            {
                "irrelevant content": [0.0, 1.0],
                "Institutions matter": [1.0, 0.0],
            }
        )

        def fake_ollama(prompt: str) -> str:
            return '{"supported":"no","quote":null,"confidence":"high","reasoning":"unrelated"}'

        with (
            patch("citecheck.checks.claims.fetch_pdf", return_value=fake_pdf),
            patch("citecheck.checks.claims._read_pdf_text", return_value="irrelevant content"),
        ):
            result = verify_claim(
                ref, "Institutions matter", embedder=embedder, ollama_call=fake_ollama
            )
        assert result.status == ClaimStatus.NOT_SUPPORTED

    def test_unparseable_ollama_response_is_error(self, tmp_path) -> None:
        pytest.importorskip("numpy")
        ref = _resolved()
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        embedder = _FakeEmbedder({"text": [1.0, 0.0], "claim": [1.0, 0.0]})

        def garbage_ollama(prompt: str) -> str:
            return "I cannot answer this question."

        with (
            patch("citecheck.checks.claims.fetch_pdf", return_value=fake_pdf),
            patch("citecheck.checks.claims._read_pdf_text", return_value="text"),
        ):
            result = verify_claim(ref, "claim", embedder=embedder, ollama_call=garbage_ollama)
        assert result.status == ClaimStatus.ERROR


class TestCheckClaimsMulti:
    def test_dedupes_repeated_claims(self, tmp_path) -> None:
        pytest.importorskip("numpy")
        ref = _resolved()
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        embedder = _FakeEmbedder({"x": [1.0], "claim A": [1.0], "claim B": [1.0]})

        calls = {"n": 0}

        def fake_ollama(_prompt: str) -> str:
            calls["n"] += 1
            return '{"supported":"yes","quote":null,"confidence":"low","reasoning":""}'

        with (
            patch("citecheck.checks.claims.fetch_pdf", return_value=fake_pdf),
            patch("citecheck.checks.claims._read_pdf_text", return_value="x"),
        ):
            out = check_claims(
                ref,
                ["claim A", "claim A", "claim B", "claim A"],  # 'A' three times
                embedder=embedder,
                ollama_call=fake_ollama,
            )
        # Dedup -> 2 unique -> 2 ollama calls.
        assert calls["n"] == 2
        assert len(out) == 2

    def test_empty_input(self) -> None:
        ref = _resolved()
        assert check_claims(ref, []) == []

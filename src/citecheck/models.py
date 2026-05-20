"""Pydantic schemas for references and check results.

Composition over inheritance: Reference holds a RawReference plus resolution
metadata. This keeps the GROBID-side and the Crossref-side concerns separate
and lets us serialize either independently.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# DOI grammar per Crossref: starts with "10.", a registrant prefix, "/", a suffix.
# We strip common URL/scheme prefixes before validating.
_DOI_PREFIX_RE = re.compile(r"^\s*(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)
_DOI_PATTERN_RE = re.compile(r"^10\.\d{4,9}/\S+$")
# Loose DOI finder for free-text raw citation strings. Stops at whitespace and at
# common citation punctuation so we don't eat the trailing "." of a sentence or
# a "," before the next field. Lowercased + revalidated via normalize_doi.
_DOI_IN_TEXT_RE = re.compile(r"10\.\d{4,9}/[^\s,;)\]<>\"']+", re.IGNORECASE)
# Trailing punctuation that often sticks to a DOI inside a citation string.
_DOI_TRAIL_RE = re.compile(r"[.,;:)\]\s]+$")


def normalize_doi(value: str | None) -> str | None:
    """Strip prefixes, lowercase, validate against the Crossref pattern.

    Returns None for empty / invalid input so resolution can downgrade rather
    than crash on garbage.
    """
    if value is None:
        return None
    stripped = _DOI_PREFIX_RE.sub("", value).strip().lower()
    if not stripped:
        return None
    return stripped if _DOI_PATTERN_RE.match(stripped) else None


def extract_doi_from_text(text: str | None) -> str | None:
    """Find the first DOI inside arbitrary text and return it normalized.

    Used as a fallback when GROBID's structured `idno[@type='DOI']` is missing
    but the raw citation string still contains a DOI (common in older PDFs and
    when GROBID's CRF variant under-extracts).
    """
    if not text:
        return None
    match = _DOI_IN_TEXT_RE.search(text)
    if not match:
        return None
    candidate = _DOI_TRAIL_RE.sub("", match.group(0))
    return normalize_doi(candidate)


class Author(BaseModel):
    """A reference author. TEI sometimes gives only a surname."""

    model_config = ConfigDict(frozen=True)

    family: str
    given: str | None = None

    def display(self) -> str:
        return f"{self.given} {self.family}".strip() if self.given else self.family


class RawReference(BaseModel):
    """A reference as extracted from the PDF, before any external resolution.

    Fields mirror GROBID's TEI biblStruct output. All optional except raw_text
    because GROBID sometimes returns a citation it could not parse.
    """

    model_config = ConfigDict(extra="forbid")

    ref_id: str | None = Field(
        default=None,
        description="GROBID's xml:id (e.g. 'b0'). Needed in Phase 5 to map in-text "
        "citation pointers back to references.",
    )
    raw_text: str
    title: str | None = None
    authors: list[Author] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1500, le=2100)
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None

    @field_validator("doi", mode="before")
    @classmethod
    def _normalize_doi(cls, v: Any) -> str | None:
        return normalize_doi(v) if isinstance(v, str) else None


class ResolutionStatus(StrEnum):
    """How we ended up resolving (or failing to resolve) a RawReference."""

    RESOLVED = "resolved"  # Confident Crossref/OpenAlex match
    AMBIGUOUS = "ambiguous"  # Multiple plausible candidates, no clear winner
    UNRESOLVED = "unresolved"  # No DOI and no high-confidence match
    ERROR = "error"  # Upstream API failure, malformed response, etc.


class Reference(BaseModel):
    """A RawReference plus the result of resolving it against external databases."""

    model_config = ConfigDict(extra="forbid")

    raw: RawReference
    status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    resolved_doi: str | None = None
    resolved_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Subset of the Crossref work record we care about. Full record is "
        "not stored to keep reports compact.",
    )
    resolution_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fuzzy match score for metadata-based resolution; None for DOI-based.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Free-text annotations from the resolution step (e.g., 'fell back to "
        "OpenAlex after Crossref timeout').",
    )

    @field_validator("resolved_doi", mode="before")
    @classmethod
    def _normalize_resolved_doi(cls, v: Any) -> str | None:
        return normalize_doi(v) if isinstance(v, str) else None


class RetractionStatus(StrEnum):
    """Outcome of a per-reference retraction check.

    Values map to Crossref's `update-to` / `relation` types. We treat
    `withdrawal` as a synonym of `retracted` (Crossref uses both) and group
    `correction` separately because it does not invalidate the cited work.
    """

    CLEAN = "clean"
    RETRACTED = "retracted"
    EXPRESSION_OF_CONCERN = "expression_of_concern"
    CORRECTION = "correction"
    UNCHECKED = "unchecked"  # no DOI to look up
    ERROR = "error"  # upstream lookup failed


class RetractionCheck(BaseModel):
    """Verdict for a single reference's retraction check."""

    model_config = ConfigDict(extra="forbid")

    status: RetractionStatus = RetractionStatus.UNCHECKED
    notice_doi: str | None = Field(
        default=None,
        description="DOI of the retraction notice (or expression of concern / correction).",
    )
    notice_date: str | None = Field(
        default=None,
        description="ISO date of the notice when Crossref provides it.",
    )
    reason: str | None = None
    source_url: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("notice_doi", mode="before")
    @classmethod
    def _normalize_notice_doi(cls, v: Any) -> str | None:
        return normalize_doi(v) if isinstance(v, str) else None


class HallucinationVerdict(StrEnum):
    """Aggregate verdict from the 5-layer hallucination detector."""

    REAL_HIGH_CONFIDENCE = "real_high_confidence"
    REAL_LOW_CONFIDENCE = "real_low_confidence"
    SUSPICIOUS = "suspicious"
    LIKELY_HALLUCINATED = "likely_hallucinated"
    UNCHECKED = "unchecked"


class LayerSignal(BaseModel):
    """Per-layer result emitted by the hallucination detector.

    `flagged` is True when the layer found something suspicious. The detector
    aggregates the count of flagged signals into a verdict.
    """

    model_config = ConfigDict(extra="forbid")

    layer: str
    flagged: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str


class HallucinationCheck(BaseModel):
    """Verdict for a single reference's hallucination check."""

    model_config = ConfigDict(extra="forbid")

    verdict: HallucinationVerdict = HallucinationVerdict.UNCHECKED
    red_flag_count: int = Field(default=0, ge=0)
    signals: list[LayerSignal] = Field(default_factory=list)
    reasoning: str = ""
    caveats: list[str] = Field(
        default_factory=list,
        description="Conditions that warranted downgrading the verdict (e.g., pre-2000 paper).",
    )


class ClaimStatus(StrEnum):
    """Verdict of the Phase 5 claim-verification check."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    NOT_SUPPORTED = "not_supported"
    UNVERIFIABLE = "unverifiable"  # paywalled / no OA PDF available / no text retrieved
    UNCHECKED = "unchecked"  # no DOI to fetch, or feature not requested
    ERROR = "error"


class ClaimCheck(BaseModel):
    """Per-citation verification of whether the cited paper supports the claim."""

    model_config = ConfigDict(extra="forbid")

    status: ClaimStatus = ClaimStatus.UNCHECKED
    claim_sentence: str | None = None
    quote: str | None = Field(
        default=None,
        description="Supporting passage from the cited paper, when the verifier found one.",
    )
    confidence: str | None = Field(
        default=None,
        description="LLM's self-reported confidence: 'low' | 'medium' | 'high'.",
    )
    reasoning: str | None = None
    notes: list[str] = Field(default_factory=list)


class JournalRiskLevel(StrEnum):
    """Crude risk classification for the cited venue."""

    LOW = "low"  # in DOAJ and not on any concern list
    MEDIUM = "medium"  # not in DOAJ, not flagged — likely a small or non-OA journal
    HIGH = "high"  # publisher name matches a known predatory list
    UNCHECKED = "unchecked"  # no journal field to look up


class JournalQualityCheck(BaseModel):
    """Per-reference assessment of the cited journal's reputational signals."""

    model_config = ConfigDict(extra="forbid")

    risk_level: JournalRiskLevel = JournalRiskLevel.UNCHECKED
    doaj_listed: bool | None = Field(
        default=None,
        description="True if DOAJ indexes the journal; None if not checked.",
    )
    on_concern_list: bool = Field(
        default=False,
        description="True if the journal/publisher matches our hard-coded predatory list.",
    )
    matched_publisher: str | None = Field(
        default=None,
        description="The concern-list entry that matched, if any.",
    )
    notes: list[str] = Field(default_factory=list)


class CheckedReference(BaseModel):
    """A Reference paired with all the per-reference check results.

    Phase 2 introduces `retraction`. Phase 3 adds `hallucination`.
    Phase 4 adds `journal_quality`. Phase 5 adds `claims` (a list because
    a reference may be cited at multiple claim sentences).
    """

    model_config = ConfigDict(extra="forbid")

    reference: Reference
    retraction: RetractionCheck = Field(default_factory=RetractionCheck)
    hallucination: HallucinationCheck = Field(default_factory=HallucinationCheck)
    journal_quality: JournalQualityCheck = Field(default_factory=JournalQualityCheck)
    claims: list[ClaimCheck] = Field(
        default_factory=list,
        description="One entry per place this reference is cited in the source PDF.",
    )

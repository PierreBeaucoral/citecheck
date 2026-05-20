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


class CheckedReference(BaseModel):
    """A Reference paired with all the per-reference check results.

    Phase 2 introduces `retraction`. Phases 3+ will append `hallucination`,
    `journal_quality`, `claim` to this model.
    """

    model_config = ConfigDict(extra="forbid")

    reference: Reference
    retraction: RetractionCheck = Field(default_factory=RetractionCheck)

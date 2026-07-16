"""Conservative evidence-quality checks for recorded job descriptions.

The check deliberately does not look for technologies, industries, seniority,
or a fixed set of role-section headings.  It only rejects sources that have no
substantive descriptive body after the already-known identity metadata is
removed.  Semantic judgment about whether a real JD is worth applying to stays
with JobDiscoveryAgent.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


class ThinJobDescriptionError(ValueError):
    """A source contains identity/card text but not a usable job description."""


_SEGMENT_BOUNDARY_RE = re.compile(r"[\n\r]+|(?<=[。！？!?；;])")
_MIN_SEGMENT_ALNUM = 10
_MIN_BODY_ALNUM = 28
_MIN_SINGLE_PARAGRAPH_ALNUM = 72
_LATIN_OR_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9_.+#/\-]*|\d+(?:[.,%]\d+)*)"
    r"(?![A-Za-z0-9_])"
)
_QUOTE_ELLIPSIS_RE = re.compile(r"(?:\.{3,}|\u2026+)")
_TYPOGRAPHIC_EQUIVALENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


def require_substantive_job_description(
    text: str,
    *,
    identity_values: Iterable[str | None] = (),
) -> None:
    """Reject unmistakably thin cards while accepting varied real-JD formats.

    A source passes when, after removing known company/title/location metadata,
    it contains either multiple substantive pieces of descriptive prose or one
    substantial paragraph.  This is a lower evidence boundary, not a writing
    formula and not a target length for the Agent.
    """

    value = unicodedata.normalize("NFKC", str(text or ""))
    for identity in identity_values:
        normalized = unicodedata.normalize("NFKC", str(identity or "")).strip()
        if normalized:
            value = value.replace(normalized, " ")

    segment_lengths = [
        _alnum_length(segment)
        for segment in _SEGMENT_BOUNDARY_RE.split(value)
    ]
    substantive = [
        length for length in segment_lengths if length >= _MIN_SEGMENT_ALNUM
    ]
    enough_body = sum(substantive) >= _MIN_BODY_ALNUM and len(substantive) >= 2
    one_complete_paragraph = max(substantive, default=0) >= _MIN_SINGLE_PARAGRAPH_ALNUM
    if enough_body or one_complete_paragraph:
        return
    raise ThinJobDescriptionError(
        "job source contains only identity/card text or an insufficient description body"
    )


def _alnum_length(value: str) -> int:
    return sum(character.isalnum() for character in value)


def require_grounded_selection_reason(
    reason: str,
    *,
    grounding_quotes: Iterable[str],
    grounding_source_texts: Iterable[str] = (),
    job_text: str,
) -> None:
    """Bind visible recommendation claims to the evidence actually cited.

    The Agent remains free to explain a semantic connection in natural
    language. Deterministic code prevents specific names/numbers that occur in
    neither the cited passages nor the JD. Quotes are rendered beside the
    reason, so prose does not duplicate long resume passages just to pass.
    """

    quotes = [
        unicodedata.normalize("NFKC", quote).strip()
        for quote in grounding_quotes
    ]
    support = unicodedata.normalize(
        "NFKC", "\n".join([job_text, *quotes, *grounding_source_texts])
    ).casefold()
    unsupported = sorted(
        {
            token
            for token in _LATIN_OR_NUMBER_RE.findall(reason)
            if _is_specific_literal(token)
            and token.casefold() not in support
        },
        key=str.casefold,
    )
    if unsupported:
        raise ValueError(
            "recommendation reason contains specific names or numbers absent from "
            "its cited evidence and JD: " + ", ".join(unsupported)
        )


def evidence_quote_matches_source(quote: str, source: str) -> bool:
    """Match copied evidence despite PDF line wraps and typographic punctuation."""
    normalized_source = _canonical_evidence_text(source)
    parts = [
        _canonical_evidence_text(part)
        for part in _QUOTE_ELLIPSIS_RE.split(quote)
        if _canonical_evidence_text(part)
    ]
    if not parts:
        return False
    cursor = 0
    for part in parts:
        position = normalized_source.find(part, cursor)
        if position < 0:
            return False
        cursor = position + len(part)
    return True


def _canonical_evidence_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).translate(
        _TYPOGRAPHIC_EQUIVALENTS
    )
    return "".join(character for character in normalized if not character.isspace())


def _is_specific_literal(token: str) -> bool:
    if any(character.isdigit() for character in token):
        return True
    if any(character in "._+#/-" for character in token):
        return True
    letters = "".join(character for character in token if character.isalpha())
    if len(letters) >= 3 and letters.isupper():
        return True
    return any(character.isupper() for character in letters[1:])

"""Lightweight text normalization and filtering helpers."""

from __future__ import annotations

import re


SUSPECT_PATTERNS: tuple[str, ...] = (
    r"\bportrait(s)?\b",
    r"\bmodels?\b",
    r"\bpeople\b",
    r"\bpark(?:ing|ed)\b",
    r"\ban?\s+(?:view|picture|image|photo)\s+of\b",
    r"\bpainting\b",
    r"\bpicture of\b",
    r"\ban?\s+image of\b",
    r"\bcars?\b",
    r"\bmaps?\b",
    r"\badvert(?:ising|isement)?\b",
    r"\bмем(?!ориал)\b",
    r"\bmemes?\b",
    r"\bбаннер(?:ы)?\b",
    r"\bposter\b",
    r"\bbillboards?\b",
    r"\binstagram\b",
    r"\bvk\.com\b",
    r"\btiktok\b",
)
COMPILED_SUSPECT_PATTERNS = tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in SUSPECT_PATTERNS)


def normalize_text(text: object) -> str:
    """Normalize whitespace and casing for multilingual text fields."""

    if not isinstance(text, str):
        return ""
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def has_suspect_text(text: str) -> bool:
    """Return True when text matches known non-landmark noise patterns."""

    normalized = (text or "").lower()
    return any(pattern.search(normalized) for pattern in COMPILED_SUSPECT_PATTERNS)


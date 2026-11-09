"""Shared English tokenization and sentence handling."""

import re

_STOP = frozenset(
    """
    a an and are as at be been by can could did do does during for from had has have
    how i if in into is it its may me might must of on or our should so than that the
    their them there these they this to up us was we were what when where which who
    why will with would you your many much long
    """.split()
)
_INSTRUCTION = re.compile(
    r"\b(?:ignore|override|disregard|forget)\b.{0,60}\b(?:instructions?|prompts?|rules?)\b"
    r"|\b(?:reply|respond|answer|output|print)\s+(?:with|only|exactly)\b",
    re.IGNORECASE,
)


def tokens(text: str) -> list[str]:
    """Return lowercase content tokens with simple English plural normalization."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [
        word[:-3] + "y"
        if len(word) > 4 and word.endswith("ies")
        else word[:-1]
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is"))
        else word
        for word in words
        if word not in _STOP
    ]


def sentences(text: str) -> list[str]:
    """Split plain text at basic sentence and line boundaries."""
    return [part for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part]


def instruction_like(text: str) -> bool:
    """Flag common instruction override and forced-output patterns."""
    return _INSTRUCTION.search(text) is not None

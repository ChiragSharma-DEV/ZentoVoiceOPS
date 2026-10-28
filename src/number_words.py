"""Spoken English digit-word conversion for voice input.

Converts runs of spoken digit words ("six nine five two oh eight six seven")
into contiguous digits ("69520867"). Only the order-ID and verification-value
extractors call this, and only when the turn's modality is voice.
"""

from __future__ import annotations

import re

DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|o)"
_RUN_RE = re.compile(rf"\b{_WORD}(?:[\s-]+{_WORD})*\b", re.IGNORECASE)
_WORD_RE = re.compile(rf"\b{_WORD}\b", re.IGNORECASE)


def normalise_digit_words(text: str) -> str:
    """Replace each run of spoken digit words with its contiguous digits."""
    def _replace(match: re.Match[str]) -> str:
        return "".join(
            DIGIT_WORDS[word.lower()] for word in _WORD_RE.findall(match.group(0))
        )

    return _RUN_RE.sub(_replace, text)

"""ASR text normalizer for VoiceOps 2026.

Normalizes spelled-out digits inside SKU-like and order-ID-like tokens into digits.
Shared function used by both retrieval and the voice layer.
"""

from __future__ import annotations
import re

DIGIT_WORDS = {
    "zero": "0", "oh": "0", "0": "0",
    "one": "1", "1": "1",
    "two": "2", "to": "2", "too": "2", "2": "2",
    "three": "3", "3": "3",
    "four": "4", "for": "4", "4": "4",
    "five": "5", "5": "5",
    "six": "6", "6": "6",
    "seven": "7", "7": "7",
    "eight": "8", "ate": "8", "8": "8",
    "nine": "9", "9": "9",
}

DIGIT_PAT = r"(?:zero|oh|one|two|to|too|three|four|for|five|six|seven|eight|ate|nine|\d)"

# Regex candidate pattern: 3 letters, separator, then 4 spelled-out or numeric digits (no trailing separator)
SKU_CANDIDATE_PATTERN = re.compile(
    rf"\b([a-zA-Z]{{3}})[- ]+({DIGIT_PAT}(?:[- ]+{DIGIT_PAT}){{3}})\b",
    re.IGNORECASE
)

# Regex candidate pattern: 8 spelled-out or numeric digits (no trailing separator)
ORDER_ID_CANDIDATE_PATTERN = re.compile(
    rf"\b({DIGIT_PAT}(?:[- ]+{DIGIT_PAT}){{7}})\b",
    re.IGNORECASE
)

NON_SKU_PREFIXES = {"one", "two", "six", "ten", "the", "for", "and", "can", "out", "how", "why", "who", "all", "any"}
KNOWN_CATALOG_PREFIXES = {"ELE", "FSH", "BTY", "HOM", "BKS", "INR", "GRO", "DGT"}


def normalise_asr_text(text: str) -> str:
    """Convert spelled-out digits inside SKU-like and order-ID-like tokens into digits.

    Change a token only when the result matches the SKU pattern (3 letters, a hyphen, 4 digits)
    or is an 8-digit order ID.
    """
    if not text:
        return ""

    def replace_sku(match: re.Match) -> str:
        prefix = match.group(1).upper()
        if prefix.lower() in NON_SKU_PREFIXES:
            return match.group(0)

        raw_match = match.group(0)
        # If no hyphen was present, require prefix to be a known catalog prefix
        if "-" not in raw_match[:5] and prefix not in KNOWN_CATALOG_PREFIXES:
            return raw_match

        raw_digits = re.findall(r"[a-zA-Z]+|\d", match.group(2).lower())
        digit_chars = [DIGIT_WORDS[w] for w in raw_digits if w in DIGIT_WORDS]
        if len(digit_chars) == 4:
            digits_str = "".join(digit_chars)
            candidate = f"{prefix}-{digits_str}"
            if re.fullmatch(r"[A-Z]{3}-\d{4}", candidate):
                return candidate
        return match.group(0)

    def replace_order_id(match: re.Match) -> str:
        raw_digits = re.findall(r"[a-zA-Z]+|\d", match.group(1).lower())
        digit_chars = [DIGIT_WORDS[w] for w in raw_digits if w in DIGIT_WORDS]
        if len(digit_chars) == 8:
            candidate = "".join(digit_chars)
            if re.fullmatch(r"\d{8}", candidate):
                return candidate
        return match.group(0)

    # 1. Replace SKU candidates
    normalised = SKU_CANDIDATE_PATTERN.sub(replace_sku, text)
    # 2. Replace Order ID candidates
    normalised = ORDER_ID_CANDIDATE_PATTERN.sub(replace_order_id, normalised)

    return normalised

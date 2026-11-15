"""Cheap, deterministic quality filters for text documents."""

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Optional


def normalize_text(text: str) -> str:
    """NFKC normalization plus outer whitespace trim."""
    return unicodedata.normalize("NFKC", text).strip()


@dataclass
class FilterConfig:
    min_chars: int = 50
    max_chars: int = 200_000
    min_alpha_ratio: float = 0.5
    max_symbol_ratio: float = 0.3
    max_repeated_line_ratio: float = 0.3


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if char.isalpha()) / len(text)


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for char in text if not char.isalnum() and not char.isspace()) / len(text)


def _repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0
    return max(Counter(lines).values()) / len(lines)


def check(text: str, config: FilterConfig) -> Optional[str]:
    """Return a rejection reason, or None when the document passes."""
    if not text:
        return "empty"
    if len(text) < config.min_chars:
        return "too_short"
    if len(text) > config.max_chars:
        return "too_long"
    if _alpha_ratio(text) < config.min_alpha_ratio:
        return "low_alpha"
    if _symbol_ratio(text) > config.max_symbol_ratio:
        return "high_symbol"
    if _repeated_line_ratio(text) > config.max_repeated_line_ratio:
        return "repeated_lines"
    return None

"""Lexical quality scoring baseline.

This is a deterministic, model-free baseline: token diversity, 4-gram
repetition, and normalized token entropy. It is useful as a cheap pre-filter or
as a proxy metric, not as a substitute for a trained quality classifier.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+", re.UNICODE)
DEFAULT_WEIGHTS = {"unique": 0.4, "repeat": 0.3, "entropy": 0.3}


@dataclass(frozen=True)
class QualityStats:
    tokens: int
    unique_ratio: float
    entropy: float
    repeat_4gram_ratio: float
    avg_word_len: float

    def as_dict(self):
        return {
            "tokens": self.tokens,
            "unique_ratio": self.unique_ratio,
            "entropy": self.entropy,
            "repeat_4gram_ratio": self.repeat_4gram_ratio,
            "avg_word_len": self.avg_word_len,
        }


def quality_stats(text: str) -> QualityStats:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return QualityStats(0, 0.0, 0.0, 0.0, 0.0)

    counts = Counter(words)
    total = len(words)
    unique_ratio = len(counts) / total
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    entropy_norm = entropy / math.log2(len(counts)) if len(counts) > 1 else 0.0

    if total >= 4:
        grams = Counter(tuple(words[i : i + 4]) for i in range(total - 3))
        repeat_ratio = 1.0 - len(grams) / sum(grams.values())
    else:
        repeat_ratio = 0.0

    avg_word_len = sum(len(word) for word in words) / total
    return QualityStats(total, unique_ratio, entropy_norm, repeat_ratio, avg_word_len)


def quality_score(stats: QualityStats, weights=None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    return (
        weights["unique"] * stats.unique_ratio
        + weights["repeat"] * (1.0 - stats.repeat_4gram_ratio)
        + weights["entropy"] * stats.entropy
    )


class QualityScorer:
    """Score text with the lexical baseline and optionally keep it."""

    def __init__(self, min_score: float = 0.65, weights=None):
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        self.min_score = min_score
        self.weights = weights or DEFAULT_WEIGHTS

    def score(self, text: str) -> float:
        return quality_score(quality_stats(text), self.weights)

    def keep(self, text: str) -> bool:
        return self.score(text) >= self.min_score

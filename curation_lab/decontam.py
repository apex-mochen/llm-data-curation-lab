"""N-gram decontamination against a reference evaluation set."""

import hashlib
from typing import Iterable, Iterator, List

from .hashing import tokenize


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def ngram_hashes(text: str, n: int, mode: str = "word") -> Iterator[int]:
    tokens = tokenize(text, mode)
    if len(tokens) < n:
        return
    if mode == "word":
        for i in range(len(tokens) - n + 1):
            yield _hash64(" ".join(tokens[i : i + n]))
    else:
        for i in range(len(tokens) - n + 1):
            yield _hash64("".join(tokens[i : i + n]))


class NGramDecontaminator:
    """Exact n-gram overlap against a (usually small) reference set."""

    def __init__(
        self,
        reference_texts: Iterable[str],
        n: int = 13,
        threshold: float = 0.5,
        mode: str = "word",
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.n = n
        self.threshold = threshold
        self.mode = mode
        self.reference_hashes = set()
        for text in reference_texts:
            self.reference_hashes.update(ngram_hashes(text, n, mode))
        if not self.reference_hashes:
            raise ValueError("reference set produced no n-grams; all documents are shorter than n")

    @property
    def index_size(self) -> int:
        return len(self.reference_hashes)

    def match_ratio(self, text: str) -> float:
        hashes: List[int] = list(ngram_hashes(text, self.n, self.mode))
        if not hashes:
            return 0.0
        return sum(1 for value in hashes if value in self.reference_hashes) / len(hashes)

    def is_contaminated(self, text: str) -> bool:
        return self.match_ratio(text) >= self.threshold

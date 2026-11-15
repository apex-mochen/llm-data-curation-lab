"""64-bit SimHash for near-duplicate detection.

SimHash is a second near-duplicate algorithm alongside MinHash: each shingle
hash votes on every bit, and the sign of the vote becomes the fingerprint bit.
Two fingerprints are near duplicates when their Hamming distance is small.
"""

from typing import Iterable

import numpy as np

from .hashing import shingle_hashes

_BITS = 64
_BIT_INDEX = np.arange(_BITS, dtype=np.uint64)


class SimHash:
    def __init__(self, ngram: int = 5, mode: str = "word") -> None:
        if ngram <= 0:
            raise ValueError("ngram must be positive")
        self.ngram = ngram
        self.mode = mode

    def fingerprint(self, text: str) -> np.uint64:
        votes = np.zeros(_BITS, dtype=np.int64)
        for shingle_hash in shingle_hashes(text, self.ngram, self.mode):
            bits = ((np.uint64(shingle_hash) >> _BIT_INDEX) & np.uint64(1)).astype(np.int64)
            votes += bits * 2 - 1
        value = 0
        for bit, vote in enumerate(votes):
            if vote > 0:
                value |= 1 << bit
        return np.uint64(value)

    def hamming(self, a: np.uint64, b: np.uint64) -> int:
        return int(a ^ b).bit_count()

    def similarity(self, a: np.uint64, b: np.uint64) -> float:
        return 1.0 - self.hamming(a, b) / _BITS


def simhash_pairs(texts: Iterable[str], ngram: int = 5, mode: str = "word"):
    hasher = SimHash(ngram=ngram, mode=mode)
    return [hasher.fingerprint(text) for text in texts]

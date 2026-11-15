"""Deduplication primitives: exact hashing and MinHash-LSH near dedup."""

from typing import List, Set

import numpy as np

from .hashing import exact_hash
from .minhash import LSH, MinHash


class ExactDedup:
    """Keep the first occurrence of an exact text."""

    def __init__(self) -> None:
        self._seen: Set[bytes] = set()

    @property
    def size(self) -> int:
        return len(self._seen)

    def check_and_add(self, text: str) -> bool:
        """Return True when ``text`` was already seen."""
        digest = exact_hash(text)
        if digest in self._seen:
            return True
        self._seen.add(digest)
        return False


class MinHashLSHDedup:
    """Keep the first document of each near-duplicate cluster.

    ``bands`` and ``rows`` control the LSH S-curve; the default 16 x 8 on 128
    permutations has an approximate threshold of 0.66.
    """

    def __init__(
        self,
        num_perm: int = 128,
        bands: int = 16,
        rows: int = 8,
        ngram: int = 5,
        mode: str = "word",
        seed: int = 42,
    ) -> None:
        if bands * rows > num_perm:
            raise ValueError("bands * rows cannot exceed num_perm")
        self.minhash = MinHash(num_perm=num_perm, seed=seed, ngram=ngram, mode=mode)
        self.lsh = LSH(num_perm=num_perm, bands=bands, rows=rows)
        self._buckets: List[Set[bytes]] = self.lsh.buckets()
        self.num_docs = 0

    def signature(self, text: str) -> np.ndarray:
        return self.minhash.signature(text)

    def check_and_add(self, text: str) -> bool:
        """Return True when ``text`` collides with an already kept document."""
        keys = list(self.lsh.band_keys(self.signature(text)))
        if any(key in bucket for key, bucket in zip(keys, self._buckets)):
            return True
        for key, bucket in zip(keys, self._buckets):
            bucket.add(key)
        self.num_docs += 1
        return False

    @property
    def bucket_count(self) -> int:
        return sum(len(bucket) for bucket in self._buckets)

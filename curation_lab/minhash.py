"""MinHash signatures and LSH banding.

The implementation is deliberately small and dependency-light (numpy only) so
the algorithm can be tested and benchmarked independently from a full data
pipeline.
"""

from typing import Iterator, List, Sequence

import numpy as np

from .hashing import shingle_hashes

_PRIME = np.uint64((1 << 61) - 1)
_MAX_UINT64 = np.uint64((1 << 64) - 1)
_BLOCK = 4096


class MinHash:
    """MinHash signature with universal hashing over 32-bit shingle hashes."""

    def __init__(self, num_perm: int = 128, seed: int = 42, ngram: int = 5, mode: str = "word"):
        if num_perm <= 0:
            raise ValueError("num_perm must be positive")
        self.num_perm = num_perm
        self.ngram = ngram
        self.mode = mode
        rng = np.random.default_rng(seed)
        # Keep a and b below 2**31 so a*h + b fits in uint64 for a 32-bit h.
        self.a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)

    def _fold(self, block: Sequence[int], signature: np.ndarray) -> None:
        hashes = np.asarray(block, dtype=np.uint64)
        values = (self.a[:, None] * hashes[None, :] + self.b[:, None]) % _PRIME
        np.minimum(signature, values.min(axis=1), out=signature)

    def signature(self, text: str) -> np.ndarray:
        signature = np.full(self.num_perm, _MAX_UINT64, dtype=np.uint64)
        block: List[int] = []
        for shingle_hash in shingle_hashes(text, self.ngram, self.mode):
            block.append(shingle_hash)
            if len(block) == _BLOCK:
                self._fold(block, signature)
                block.clear()
        if block:
            self._fold(block, signature)
        return signature


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    if sig_a.shape != sig_b.shape:
        raise ValueError("signatures must have the same length")
    return float(np.mean(sig_a == sig_b))


class LSH:
    """Band the signature; two documents collide if any band matches."""

    def __init__(self, num_perm: int, bands: int, rows: int):
        if bands <= 0 or rows <= 0:
            raise ValueError("bands and rows must be positive")
        if bands * rows > num_perm:
            raise ValueError("bands * rows cannot exceed num_perm")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows

    def band_keys(self, signature: np.ndarray) -> Iterator[bytes]:
        for band in range(self.bands):
            start = band * self.rows
            yield signature[start : start + self.rows].tobytes()

    def buckets(self) -> List[set]:
        return [set() for _ in range(self.bands)]


def lsh_threshold(bands: int, rows: int) -> float:
    """Approximate Jaccard threshold of the S-curve."""
    return (1.0 / bands) ** (1.0 / rows)

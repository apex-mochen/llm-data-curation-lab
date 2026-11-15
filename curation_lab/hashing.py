"""Stable hashing helpers for the curation lab."""

import hashlib
import zlib
from typing import Iterable, Iterator, List


def exact_hash(text: str) -> bytes:
    """128-bit stable hash for exact duplicate detection."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def tokenize(text: str, mode: str) -> List[str]:
    if mode == "word":
        return text.split()
    if mode == "char":
        return list(text)
    raise ValueError(f"unknown tokenization mode: {mode!r}")


def shingle_hashes(text: str, n: int, mode: str = "word") -> Iterator[int]:
    """Yield a stable 32-bit hash for every n-gram shingle.

    CRC32 is used here because MinHash only needs a uniform-ish 32-bit value;
    exact duplicate detection uses ``exact_hash`` instead.
    """
    tokens = tokenize(text, mode)
    if len(tokens) < n:
        return
    if mode == "word":
        for i in range(len(tokens) - n + 1):
            yield zlib.crc32(" ".join(tokens[i : i + n]).encode("utf-8"))
    else:
        for i in range(len(tokens) - n + 1):
            yield zlib.crc32("".join(tokens[i : i + n]).encode("utf-8"))


def ngram_hashes(text: str, n: int, mode: str = "word") -> Iterable[int]:
    return shingle_hashes(text, n, mode)

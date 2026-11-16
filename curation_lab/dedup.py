"""Deduplication primitives: exact hashing and MinHash-LSH near dedup."""

import sqlite3
from typing import List, Optional, Set

import numpy as np

from .hashing import exact_hash
from .minhash import LSH, MinHash


class ExactDedup:
    """Keep the first occurrence of an exact text.

    With ``sqlite_path`` set, digests are stored in SQLite instead of an
    in-memory set, bounding memory for larger-than-RAM corpora at the cost of
    disk I/O.
    """

    def __init__(self, sqlite_path: Optional[str] = None) -> None:
        self._sqlite_path = sqlite_path
        self._conn: Optional[sqlite3.Connection] = None
        self._seen: Set[bytes] = set()
        self._pending = 0
        if sqlite_path is not None:
            self._conn = sqlite3.connect(sqlite_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS seen (digest BLOB PRIMARY KEY)")

    @property
    def size(self) -> int:
        if self._conn is not None:
            return int(self._conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0])
        return len(self._seen)

    def check_and_add(self, text: str) -> bool:
        """Return True when ``text`` was already seen."""
        digest = exact_hash(text)
        if self._conn is not None:
            cursor = self._conn.execute("INSERT OR IGNORE INTO seen(digest) VALUES (?)", (digest,))
            if cursor.rowcount:
                self._pending += 1
                # Batch commits so per-record inserts stay cheap; close()
                # flushes the tail.
                if self._pending >= 1000:
                    self._conn.commit()
                    self._pending = 0
                return False
            return True
        if digest in self._seen:
            return True
        self._seen.add(digest)
        return False

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None


class MinHashLSHDedup:
    """Keep the first document of each near-duplicate cluster.

    ``bands`` and ``rows`` control the LSH S-curve; the default 16 x 8 on 128
    permutations has an approximate threshold of 0.66.

    With ``transitive=True`` (default), band keys from duplicate documents are
    also recorded. A later document that only matches an already-dropped
    document is still removed, so the result is the connected components of the
    band graph. With ``transitive=False``, documents are compared only against
    kept documents (first-match), which can keep a chain member.
    """

    def __init__(
        self,
        num_perm: int = 128,
        bands: int = 16,
        rows: int = 8,
        ngram: int = 5,
        mode: str = "word",
        seed: int = 42,
        transitive: bool = True,
    ) -> None:
        if bands * rows > num_perm:
            raise ValueError("bands * rows cannot exceed num_perm")
        self.minhash = MinHash(num_perm=num_perm, seed=seed, ngram=ngram, mode=mode)
        self.lsh = LSH(num_perm=num_perm, bands=bands, rows=rows)
        self._buckets: List[Set[bytes]] = self.lsh.buckets()
        self.transitive = transitive
        self.num_docs = 0
        self.num_seen = 0

    def signature(self, text: str) -> np.ndarray:
        return self.minhash.signature(text)

    def check_and_add(self, text: str) -> bool:
        """Return True when ``text`` belongs to an already seen cluster."""
        keys = list(self.lsh.band_keys(self.signature(text)))
        duplicate = any(key in bucket for key, bucket in zip(keys, self._buckets))
        if duplicate and not self.transitive:
            return True
        # Record keys for duplicates too when clustering transitively, so a
        # later document can chain through an already-dropped document.
        for key, bucket in zip(keys, self._buckets):
            bucket.add(key)
        self.num_seen += 1
        if duplicate:
            return True
        self.num_docs += 1
        return False

    @property
    def bucket_count(self) -> int:
        return sum(len(bucket) for bucket in self._buckets)

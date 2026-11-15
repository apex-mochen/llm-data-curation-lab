"""Small, testable building blocks for LLM data curation."""

from .dedup import ExactDedup, MinHashLSHDedup
from .hashing import exact_hash, shingle_hashes
from .minhash import LSH, MinHash, jaccard_estimate, lsh_threshold

__all__ = [
    "ExactDedup",
    "MinHashLSHDedup",
    "LSH",
    "MinHash",
    "exact_hash",
    "jaccard_estimate",
    "lsh_threshold",
    "shingle_hashes",
]

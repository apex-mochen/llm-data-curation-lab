"""Small, testable building blocks for LLM data curation."""

from .dedup import ExactDedup, MinHashLSHDedup
from .hashing import exact_hash, shingle_hashes
from .minhash import LSH, MinHash, jaccard_estimate, lsh_threshold
from .mixing import MixSource, allocate_quotas, mix_jsonl, reservoir_sample_jsonl
from .simhash import SimHash, simhash_pairs

__all__ = [
    "ExactDedup",
    "MinHashLSHDedup",
    "LSH",
    "MinHash",
    "MixSource",
    "SimHash",
    "allocate_quotas",
    "exact_hash",
    "jaccard_estimate",
    "lsh_threshold",
    "mix_jsonl",
    "reservoir_sample_jsonl",
    "shingle_hashes",
    "simhash_pairs",
]

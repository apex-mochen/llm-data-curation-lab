"""Small, testable building blocks for LLM data curation."""

from .dedup import ExactDedup, MinHashLSHDedup
from .hashing import exact_hash, shingle_hashes
from .lsh_params import collision_probability, optimal_lsh_params
from .minhash import LSH, MinHash, jaccard_estimate, lsh_threshold
from .mixing import (
    MixSource,
    allocate_quotas,
    count_jsonl,
    mix_jsonl,
    mix_jsonl_temperature,
    reservoir_sample_jsonl,
    temperature_quotas,
)
from .simhash import SimHash, simhash_pairs

__all__ = [
    "ExactDedup",
    "MinHashLSHDedup",
    "LSH",
    "MinHash",
    "MixSource",
    "SimHash",
    "allocate_quotas",
    "collision_probability",
    "count_jsonl",
    "exact_hash",
    "jaccard_estimate",
    "lsh_threshold",
    "mix_jsonl",
    "mix_jsonl_temperature",
    "optimal_lsh_params",
    "reservoir_sample_jsonl",
    "shingle_hashes",
    "simhash_pairs",
]

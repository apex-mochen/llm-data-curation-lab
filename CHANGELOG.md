# Changelog

All notable changes to `llm-data-curation-lab` are documented here.

## v0.2.0 — 2026-11-16

### Added

- Quality-weighted temperature mixing:
  `p_i ∝ count_i**alpha * quality_i**beta`.
- CLI support: `--quality` (one per `--input`) and `--quality-beta`.
- Quota validation tests for length mismatch and non-positive quality.
- CLI end-to-end test for quality-weighted mixing.
- Benchmark comparing `beta=0` (temperature only) with `beta=1` on a
  duplicate-heavy source.

### Changed

- 54 unit tests (was 48 in v0.1.0).
- README and DESIGN document the quality-weighted mixing formula, CLI and
  benchmark caveats.

### Benchmark

10/90 sources, source B quality 0.25, total 400: `beta=0` quotas `[333, 67]`,
B copy docs 35.3, unique texts 387; `beta=1` quotas `[381, 19]`, B copy docs
8.0, unique texts 396. The quality score is a synthetic proxy, not a
downstream training-quality result.

## v0.1.0 — 2026-11-15

Initial public release.

### Algorithms

- Streaming JSONL quality filters: length, alpha ratio, symbol ratio,
  repeated-line ratio.
- Exact deduplication with BLAKE2b-128.
- MinHash-LSH near deduplication: 128 permutations, 16 × 8 bands, transitive
  clustering, optional threshold-driven bands/rows via `optimal_lsh_params`.
- SimHash: 64-bit fingerprints, Hamming distance, and pigeonhole-banded
  pipeline dedup (`--near-algo simhash`).
- 13-gram overwrite-style decontamination against an evaluation corpus.
- Weighted data mixing with proportional quotas and per-source reservoir
  sampling; DoReMi-style temperature mixing (`p_i ∝ count_i ** alpha`).

### Benchmarks

- 20k-document synthetic corpus: MinHash-LSH and SimHash pipeline comparison.
- WikiText-2 5k and full 14,324-document pipeline runs.
- 1,000-pair MinHash vs SimHash quality evaluation (true 5-gram Jaccard ground
  truth).
- LSH parameter selection and banding quality on synthetic and real pairs.
- Decontamination control and n-gram/threshold sweep on WikiText-2 + GSM8K.
- Mixing quality with a duplicate-heavy source.

### Tests and CI

- 48 unit tests.
- GitHub Actions CI on Python 3.10, 3.11 and 3.12.
- flake8, black and isort clean.

### Known limitations

- Benchmarks use synthetic/controlled corpora plus WikiText-2 samples; they are
  not downstream LLM training-quality results.
- MinHash-LSH is the default near-dedup algorithm; SimHash is a lower-memory
  alternative and its Hamming threshold is corpus-sensitive.
- Mixing is a batch stage and is not streaming.
- Decontamination evaluates n-gram overlap, not semantic contamination.

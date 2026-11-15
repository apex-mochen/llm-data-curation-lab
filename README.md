# LLM Data Curation Lab

A small, reproducible data-engineering project for LLM pretraining/SFT data:

**quality filters → exact dedup → MinHash-LSH near dedup → n-gram decontamination → stats**

It is intentionally dependency-light (Python + NumPy) so every algorithm is
readable, testable, and benchmarkable. The goal is to demonstrate the data
algorithms and the streaming engineering around them, not to replace a
production framework.

## Why this project exists

Most "data engineering" portfolio projects stop at `pandas.read_csv` + a few
filters. LLM data work has three harder properties:

1. **Scale**: documents are long-tail; a single 10 MB document can OOM a worker.
2. **Algorithms**: exact dedup is easy, but near dedup (MinHash + LSH) and
   benchmark decontamination (n-gram overlap) have real precision/recall and
   memory trade-offs.
3. **Measurability**: throughput, peak RSS, dedup rate, and contamination
   removal must be reported with reproducible commands.

This project implements those pieces directly and measures them.

## Layout

```
curation_lab/
  hashing.py    stable 32-bit shingle hashes + 128-bit exact hashes
  minhash.py    MinHash signatures (universal hashing) + LSH banding
  dedup.py      ExactDedup + MinHashLSHDedup
  filters.py    length / alpha / symbol / repeated-line filters
  decontam.py   n-gram overlap against an evaluation set
  pipeline.py   streaming JSONL pipeline + stats
  cli.py        `curation-lab run ...`
tests/          21 unit tests (algorithms + end-to-end pipeline)
bench/          synthetic corpus generator + benchmark scripts
```

## Usage

```bash
python -m curation_lab.cli run \
  --input corpus.jsonl \
  --output kept.jsonl \
  --eval eval.jsonl \
  --stats stats.json
```

Input/output are JSONL with a `text` field (configurable in
`PipelineConfig`). The pipeline is streaming: it never loads the corpus into
memory, but the exact-hash set and LSH band buckets do grow with the number of
kept documents.

## Algorithms

- **Exact dedup**: BLAKE2b-128 of the NFKC-normalized text; first occurrence wins.
- **MinHash**: 128 permutations; each shingle hash `h` (CRC32) is mapped by
  `(a*h + b) mod (2^61-1)` with `a, b < 2^31`, so the product fits in
  `uint64`. Shingles are folded in 4,096-hash blocks; the signature is
  `O(num_perm)` memory regardless of document length.
- **LSH**: 16 bands × 8 rows (approximate Jaccard threshold 0.66). A document
  is a near duplicate if any band bucket already contains its band key.
- **Decontamination**: BLAKE2b-64 hashes of 13-grams from the reference set;
  a document is dropped when the fraction of its n-grams present in the
  reference set is at least the threshold (default 0.5).

## Tests

```bash
python -m pytest tests -q
```

`21 passed` locally. The suite covers MinHash estimate accuracy vs brute-force
Jaccard, deterministic signatures, block-boundary equivalence, exact/LSH dedup,
filter boundaries, decontamination ratio, and the full pipeline stats/output.

## Benchmark

Synthetic corpus: 20,000 documents × 100 whitespace tokens, 50% unique, 25%
exact duplicates, 15% one-token near duplicates, 10% benchmark copies.
Three fresh processes; identical corpus; `bench/bench_pipeline.py`.

| Input | Exact dups | Near dups | Contaminated | Output | Docs/s | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 6,950 | 2,962 | 50 | 10,038 | 7,506–7,751 | 68.8 MiB |

Standalone MinHash-LSH benchmark (`bench/bench_dedup.py`): 10k docs at
~12.7k docs/s (72 MiB), 50k docs at ~12.9k docs/s (176 MiB). The end-to-end
pipeline is slower because it also parses JSON, runs filters, decontaminates,
and writes output.

Raw files: `bench/pipeline_results.jsonl`, `bench/results.jsonl`,
`bench/pipeline.err`, `bench/bench.err`.

## What this does not prove

- Synthetic corpus only; no real FineWeb/C4 sample yet, no downstream model
  evaluation, no comparison with Data-Juicer or datatrove on the same data.
- Single process only; no Ray/multiprocessing, checkpointing, or sharding.
- Dedup state is in memory; memory grows with the number of unique/kept
  documents (inherent to this design, and documented).
- No tokenizer-based token statistics yet.

## Roadmap

1. Run on a real public corpus (FineWeb / Wikipedia sample) and compare with
   Data-Juicer / datatrove on the same subset.
2. Add checkpoint/resume and shard-level parallelism; measure the scaling curve.
3. Add a token-count stage and a data-mixing sampler.
4. Publish charts and a short design note in the README.

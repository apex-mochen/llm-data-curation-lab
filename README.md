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
docs/DESIGN.md  memory model, algorithm details, limitations
```

Design trade-offs (why 32-bit shingle hashes, why 61-bit modulo, why the
dedup index is the only growing state) are documented in
[docs/DESIGN.md](docs/DESIGN.md).

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

`23 passed` locally. The suite covers MinHash estimate accuracy vs brute-force
Jaccard, deterministic signatures, block-boundary equivalence, exact/LSH dedup,
transitive vs first-match chain handling, filter boundaries, decontamination
ratio, and the full pipeline stats/output.

## Benchmark

Synthetic corpus: 20,000 documents × 100 whitespace tokens, 50% unique, 25%
exact duplicates, 15% one-token near duplicates, 10% benchmark copies.
Three fresh processes; identical corpus; `bench/bench_pipeline.py`.

```bash
python bench/make_corpus.py --docs 20000 --tokens 100 --out-dir bench/data
NEAR_DEDUP=1 python bench/bench_pipeline.py   # exact + near
NEAR_DEDUP=0 python bench/bench_pipeline.py   # exact only
```

| Variant | Exact dups | Near dups | Output | Docs/s | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact only (`NEAR_DEDUP=0`) | 6,950 | 0 | 13,000 | 11,518–11,842 | 40.7 MiB |
| exact + MinHash-LSH | 6,950 | 2,962 | 10,038 | 7,420–7,682 | 69.3–69.6 MiB |

The near-dedup stage removes a further 2,962 documents (22.8% of the exact-only
output) at a cost of about +55% wall time and +29 MiB peak RSS. That is the
trade-off a curation pipeline has to make explicitly; both variants are
measured on the same corpus.

Standalone MinHash-LSH benchmark (`bench/bench_dedup.py`): 10k docs at
~12.7k docs/s (72 MiB), 50k docs at ~12.9k docs/s (176 MiB). The end-to-end
pipeline is slower because it also parses JSON, runs filters, decontaminates,
and writes output.

Raw files: `bench/pipeline_results.jsonl`, `bench/pipeline_exact_only.jsonl`,
`bench/results.jsonl`, `bench/pipeline.err`, `bench/bench.err`.

## Real-text check (local OSS corpus)

To exercise the pipeline on real text without network access, the corpus below
is built from the public Data-Juicer repository's `.md` / `.txt` / `.py` files:

```bash
python bench/make_repo_corpus.py \
  --root /path/to/data-juicer \
  --out-dir bench/real_corpus \
  --exts .md .txt .py
DATA_DIR=bench/real_corpus EVAL=none NEAR_DEDUP=1 python bench/bench_pipeline.py
```

| Docs | Chars | Filtered | Exact dups | Near dups | Output | Docs/s | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,177 | 12.0 M | 72 low-alpha + 2 too-long | 0 | 99 | 1,004 | 670–697 | 57–59 MiB |

This is a convenience sample of one public repository, not a public benchmark
and not a claim about web-scale data. It does show that the pipeline handles
real Markdown/Python text (including Chinese docs and code punctuation), and
that near-dedup finds 99 similar documents where exact hashing finds none.
Raw file: `bench/real_pipeline_results.jsonl`.

## Cross-check against Data-Juicer

To check that the MinHash-LSH implementation finds the same near-duplicate
population as a mature implementation, both were run on the same 13,000
exact-deduplicated documents with 128 permutations, 16 bands × 8 rows and
5-word shingles:

```bash
LIMIT=13000 VARIANT=ours python bench/compare_dj_minhash.py
LIMIT=13000 VARIANT=dj   python bench/compare_dj_minhash.py
```

| Implementation | Removed | Docs/s | Peak RSS |
| --- | ---: | ---: | ---: |
| lab `MinHashLSHDedup` | 2,962 | 12,451 | 79.7 MiB |
| Data-Juicer `document_minhash_deduplicator` | 3,000 | 2,046 | 302.8 MiB |

The removal counts agree within 1.3%. The 38-document difference is consistent
with hash/LSH randomness: the lab hashes shingles with CRC32 and Data-Juicer
uses SHA-1, so borderline pairs can fall on different sides of the band
collision. It is not evidence of a semantic gap. The lab also supports
transitive clustering (`transitive=True`, the default); on this corpus it gives
the same 2,962 removals as first-match because there are no chain cases, but it
is covered by a dedicated chain test.
Throughput and RSS are **not** directly comparable: the Data-Juicer run also
builds an Arrow dataset and runs its `process()` clustering stage. The useful
signal is the near-identical duplicate population, not a speed ranking.
Raw file: `bench/compare_results.jsonl`.

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

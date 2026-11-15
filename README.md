# LLM Data Curation Lab

[![ci](https://github.com/apex-mochen/llm-data-curation-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/apex-mochen/llm-data-curation-lab/actions/workflows/ci.yml)

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
  lsh_params.py  choose LSH bands/rows from a target Jaccard threshold
  dedup.py      ExactDedup + MinHashLSHDedup
  filters.py    length / alpha / symbol / repeated-line filters
  decontam.py   n-gram overlap against an evaluation set
  mixing.py     weighted quotas + per-source reservoir sampling
  simhash.py    64-bit SimHash fingerprints + Hamming distance
  pipeline.py   streaming JSONL pipeline + stats
  cli.py        `curation-lab run ...` / `curation-lab mix ...`
tests/          42 unit tests (algorithms + end-to-end pipeline)
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

## Data mixing

```bash
python -m curation_lab.cli mix \
  --input code.jsonl:0.5 \
  --input web.jsonl:0.3 \
  --input math.jsonl:0.2 \
  --output mixed.jsonl \
  --total 100000
```

Quotas are allocated proportionally with deterministic remainder handling; each
source contributes a uniform reservoir sample of its quota, seeded per source.
Sources shorter than their quota underfill (the remainder is not redistributed
in v1). Memory is O(total output documents) because the reservoir is held
before writing.

Temperature mixing (`p_i ∝ count_i ** alpha`, DoReMi-style) is available
without explicit weights:

```bash
python -m curation_lab.cli mix \
  --input small.jsonl:1.0 \
  --input large.jsonl:1.0 \
  --output mixed.jsonl --total 10 --temperature 0.5
```

`alpha=0` is uniform, `alpha=1` is proportional to source size, and
`0 < alpha < 1` upweights smaller sources. For a 10/90 source split and a
target of 10 documents: `alpha=1` gives `[1, 9]`, `alpha=0.5` gives `[3, 7]`,
and `alpha=0` gives `[5, 5]`.

## Algorithms

- **Exact dedup**: BLAKE2b-128 of the NFKC-normalized text; first occurrence wins.
- **MinHash**: 128 permutations; each shingle hash `h` (CRC32) is mapped by
  `(a*h + b) mod (2^61-1)` with `a, b < 2^31`, so the product fits in
  `uint64`. Shingles are folded in 4,096-hash blocks; the signature is
  `O(num_perm)` memory regardless of document length.
- **LSH**: 16 bands × 8 rows (approximate Jaccard threshold 0.66). A document
  is a near duplicate if any band bucket already contains its band key.
  `transitive=True` (default) also records keys from dropped documents, so the
  result is the connected components of the band graph; `transitive=False`
  compares only against kept documents (first-match).
- **Decontamination**: BLAKE2b-64 hashes of 13-grams from the reference set;
  a document is dropped when the fraction of its n-grams present in the
  reference set is at least the threshold (default 0.5).
- **SimHash**: each 5-gram shingle hash votes on all 64 fingerprint bits; the
  sign of the vote sets the bit. Two documents are near duplicates when their
  Hamming distance is below a threshold. Used as a second near-duplicate
  algorithm and as a cross-check on MinHash.

## LSH parameter selection

`curation_lab/lsh_params.py` chooses bands/rows from a target Jaccard
threshold. It minimizes the average area of the S-curve's false-positive
region below the target and false-negative region above it, subject to
`bands * rows <= num_perm`.

```bash
python -m curation_lab.cli lsh-params --threshold 0.7 --num-perm 128
```

| Target Jaccard | Bands | Rows | S-curve threshold | Weighted error |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 25 | 5 | 0.525 | 0.0218 |
| 0.7 | 12 | 10 | 0.780 | 0.0158 |
| 0.9 | 3 | 42 | 0.974 | 0.0033 |

The fixed 16 × 8 configuration used by the pipeline has an S-curve threshold of
0.707. The selected parameters can differ because the objective is the
integrated error, not the point where the curve crosses 0.5. The integration is
a 200-step numeric approximation.

## MinHash vs SimHash (quality)

`bench/bench_dedup_quality.py` builds 1,000 labeled pairs (200 base documents ×
exact / 1-word / 5-word / 10-word changes / unrelated). Ground truth is exact
5-gram Jaccard on shingle sets; a pair is a duplicate when Jaccard ≥ 0.7.

| Algorithm | Threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| MinHash (128 perms) | Jaccard 0.7 | 1.000 | 0.939 | 0.968 |
| SimHash (64 bit) | Hamming 3 | 1.000 | 0.658 | 0.793 |
| SimHash (64 bit) | Hamming 5 | 0.998 | 0.784 | 0.878 |
| SimHash (64 bit) | Hamming 11 (best F1) | 0.987 | 0.986 | 0.987 |

800 of the 1,000 pairs are duplicates by true Jaccard. MinHash's mean absolute
Jaccard error is 0.047. SimHash's mean Hamming distance is 2.93 for duplicates
and 16.2 for non-duplicates, so the threshold choice is the main trade-off:
strict thresholds give perfect precision but lose recall.

This is a synthetic, single-configuration comparison (100-token documents,
5-grams); it is not a claim that one algorithm is universally better.

## Tests

```bash
python -m pytest tests -q
```

`42 passed` locally. The suite covers MinHash estimate accuracy vs brute-force
Jaccard, deterministic signatures, block-boundary equivalence, exact/LSH dedup,
transitive vs first-match chain handling, SimHash fingerprints/Hamming, LSH
parameter selection, filter boundaries, decontamination ratio, weighted and
temperature quota allocation + reservoir mixing, and the full pipeline
stats/output.

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

## Public corpus (WikiText-2)

A real public corpus check, using the raw WikiText-2 training split
(`Salesforce/wikitext`, `wikitext-2-raw-v1`) via the Hugging Face `datasets`
loader:

```bash
python bench/make_hf_corpus.py \
  --dataset Salesforce/wikitext --config wikitext-2-raw-v1 \
  --split train --limit 40000 --min-chars 200 --out-dir bench/public_corpus_full
DATA_DIR=bench/public_corpus_full EVAL=none NEAR_DEDUP=1 python bench/bench_pipeline.py
```

| Corpus | Docs | Chars | Filtered | Exact dups | Near dups | Output | Docs/s | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k sample | 5,000 | 3.71 M | 4 low-alpha | 0 | 17 | 4,979 | 3,279–3,967 | 65.0–65.3 MiB |
| full train (≥200 chars) | 14,324 | 10.4 M | 6 low-alpha | 0 | 71 | 14,247 | 3,169–4,086 | 79.2–80.2 MiB |

The 14,324 documents are all non-empty rows of the 36,718-row train split with
at least 200 characters. This is a public-corpus engineering check, not a
downstream-quality claim. It confirms that the pipeline runs on real Wikipedia
text and that near-dedup finds repeated/similar documents where exact hashing
finds none (17/5k and 71/14.3k). Raw files:
`bench/public_pipeline_results.jsonl`, `bench/public_full_pipeline_results.jsonl`.
The corpus is downloaded at run time and is not committed.

## Decontamination control (WikiText-2 + GSM8K)

A positive/negative control for the decontamination stage, using real public
data: 2,000 clean WikiText-2 documents as the corpus, 300 GSM8K test questions
as the reference set, and 100 exact + 100 one-word-changed GSM8K questions
injected into the corpus.

```bash
python bench/bench_decontam_control.py \
  --wikitext-corpus bench/public_corpus/corpus.jsonl \
  --out-dir bench/decontam_control --n-eval 300 --n-inject 100 --n-clean 2000
```

| Clean docs | Eval questions | Injected exact | Injected near | Index 13-grams | Precision | Recall | Max clean ratio | Min near ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 300 | 100 | 100 | 10,146 | 1.0 | 1.0 | 0.0 | 0.857 |

The control shows the intended behavior: no clean WikiText document reaches the
0.5 threshold, every exact copy has ratio 1.0, and every one-word-changed copy
stays above 0.857. This is a controlled synthetic injection on real text, not a
measurement of how contaminated a real training corpus is. Raw file:
`bench/decontam_control/result.json`.

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

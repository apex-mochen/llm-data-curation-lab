# Case study: building a small LLM data-curation lab

This is the narrative version of `llm-data-curation-lab`: what problem it solves,
which algorithms were chosen, what the measurements showed, and what the
project does **not** prove.

## Problem

LLM pretraining/SFT data pipelines need to remove junk and duplicates before
tokenization. Three questions drove the project:

1. Can exact and near-duplicate removal be implemented with understandable,
   testable algorithms rather than a black-box dependency?
2. Which near-duplicate algorithm is the right default: MinHash-LSH or SimHash?
3. How do decontamination and data mixing change the output, and how do we
   measure that without training a model?

The project is deliberately small enough to run locally and inspect end to end.

## Pipeline

`JSONL → quality filters → exact dedup → near dedup → decontamination → mixing`

- Quality filters: length, alpha ratio, symbol ratio, repeated-line ratio.
- Exact dedup: BLAKE2b-128 over NFKC-normalized text.
- Near dedup: MinHash-LSH (default) or SimHash with pigeonhole banding.
- Decontamination: 13-gram overlap against an evaluation corpus.
- Mixing: proportional quotas, temperature mixing, quality-weighted mixing.

The CLI exposes the same pieces (`run`, `mix`, `lsh-params`).

## Decision 1: MinHash without materializing shingle sets

The first implementation built a set of all shingles per document and then
materialized a full hash array. For a 1M-token document this is unnecessary:
MinHash only needs the minimum under each permutation, and duplicate shingles
cannot make the minimum smaller.

Streaming the shingles in bounded blocks reduced peak RSS from 345 MiB to
268 MiB (−22.4%) and kept the signature byte-for-byte identical. The differential
test compares the old set-based implementation with the streaming one.

## Decision 2: LSH parameters from the target threshold

The pipeline's fixed 16 × 8 banding has an S-curve crossing at 0.707. A target
threshold of 0.5 or 0.9 needs different bands/rows. `optimal_lsh_params`
integrates the false-positive area below the target and the false-negative area
above it, subject to `bands * rows <= num_perm`.

For `num_perm=128`, the selected configurations were 25 × 5 for 0.5, 12 × 10
for 0.7, and 3 × 42 for 0.9. The point is not that one configuration is
universally best; it is that the band choice should follow the target and be
validated on data.

## Decision 3: MinHash vs SimHash

On 1,000 labeled pairs (800 true duplicates by exact 5-gram Jaccard ≥ 0.7):

| Algorithm | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| MinHash @0.7 | 1.000 | 0.939 | 0.968 |
| SimHash H=11 (best F1) | 0.987 | 0.986 | 0.987 |
| SimHash H=3 | 1.000 | 0.658 | 0.793 |

SimHash can win on F1 at a tuned threshold, but its Hamming threshold is much
more corpus-sensitive. On real WikiText-2 (14,324 docs), MinHash removed 71
documents at 4,476.6 docs/s and 78.9 MiB; SimHash H=3 removed 125 at
725.8 docs/s and 47.7 MiB; H=8 removed 13,280 and left only 1,038.

Conclusion: MinHash stays the default. SimHash is a lower-memory alternative
with a threshold that must be retuned per corpus. This is exactly the kind of
"synthetic result does not generalize" finding the project is meant to expose.

## Decision 4: Decontamination is a threshold problem

The control set was 2,000 clean WikiText-2 documents, 300 GSM8K evaluation
questions, 100 exact injections and 100 one-word-changed injections.

At 13-gram / threshold 0.5: precision 1.0, recall 1.0, clean max overlap 0.0,
nearest near-copy ratio 0.857.

The sweep matters: at threshold 0.9, four near copies with ratio 0.857 are
missed. A single "0.5 is reasonable" claim would hide that boundary.

## Decision 5: Mixing is not deduplication

With 1,000 clean documents in source A and 200 documents in source B (100 exact
copies of A), temperature mixing at α=1/0.5/0 spent 35/66/100 quota on known
copies and produced 387/380/377 unique texts.

Quality-weighted mixing (`p_i ∝ count_i^α * quality_i^β`) with B quality 0.25
changed β=0 quotas [333,67] / 35.3 copies / 387 unique to β=1 quotas [381,19] /
8.0 copies / 396 unique. The quality score is a synthetic proxy; the experiment
only shows the quota mechanics, not a downstream training gain.

## Decision 6: exact dedup on disk when memory is the constraint

The default exact dedup keeps 128-bit digests in a Python set. For
larger-than-RAM corpora, `--exact-db` stores them in SQLite instead, keeping the
same first-occurrence semantics.

At 1M unique rows: memory backend 1.03 s / 132.2 MiB, SQLite backend
79.5 s / 42.8 MiB. That is ~77× slower for ~3× less memory. SQLite is the
memory-constrained option, not a throughput option.

## What failed and was not shipped

- Batching the whole minhash column in `DocumentMinhashDeduplicator.process`
  looked 19% faster at 100k rows but was 31% slower at 300k. It was dropped
  instead of being turned into a PR.
- SimHash H=8 on real text over-deduplicated 93% of the corpus; it is documented
  as a warning, not a recommended setting.
- A decontamination operator prototype duplicated upstream PR #1184, so it was
  not published.
- Ray band-sharded LSH (200k signatures, 16 bands, 20k duplicate groups) matched
  clusters but did not scale: 0.83× / 0.97× / 1.14× / 1.04× at 1/2/4/8 workers.
  The central Python union-find over ~2.88M edges dominated; this is recorded as
  a negative result, not a distributed-clustering success.

## Limits

- Benchmarks use synthetic/controlled corpora plus WikiText-2 samples, not
  downstream LLM training quality.
- "Duplicate" is defined by exact 5-gram Jaccard ≥ 0.7; there is no human
  annotation.
- Mixing is a batch stage; the pipeline is not a distributed system.
- Data-juicer: 17 PRs merged; #1188/#1189 are still under review.

## What I would do next

1. Distributed union-find (BTS) or aggressive edge deduplication; naive band
   sharding was tested and did not scale (see the failure section).
2. Add a held-out downstream proxy (filter pass rate, token distribution) for
   mixing instead of relying only on duplicate counts.
3. Persist dedup state for checkpoint/resume, and measure the cost of restart.

The value of the project is not a production-scale claim; it is a reproducible
record of algorithm choices, measurements, and the places where the obvious
choice was wrong.

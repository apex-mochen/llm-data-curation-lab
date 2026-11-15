# Design notes

## Goals

- Make LLM data curation **algorithms** explicit and testable: exact dedup,
  MinHash + LSH, n-gram decontamination.
- Keep the **streaming boundary** honest: the corpus is never loaded as a whole;
  only the dedup index grows with the number of kept documents.
- Make every performance number reproducible with a command and a raw file.

## Non-goals

- Replacing Data-Juicer / datatrove / NeMo Curator.
- Distributed execution, GPU tokenization, or model-based quality scoring.
- A production SLA; this is a lab with explicit limitations.

## Pipeline

```
JSONL reader
  → NFKC normalize + trim
  → quality filters (length, alpha ratio, symbol ratio, repeated lines)
  → exact dedup (BLAKE2b-128)
  → MinHash-LSH near dedup (128 perms, 16×8 bands)
  → n-gram decontamination (optional reference JSONL)
  → JSONL writer + stats
```

Each document is processed once. The reader and writer are streaming; the
filter stage allocates only per-document objects.

## Memory model

| State | Grows with | Bound |
| --- | --- | --- |
| exact hash set | number of kept documents | 16 B + set overhead per doc |
| LSH band buckets | number of kept documents (first-match) or all documents (transitive) × bands | 8 B × rows + bytes overhead per band key |
| MinHash signature | `num_perm` | O(1) per document |
| shingle folding block | block size | 4,096 hashes |
| decontam index | reference n-grams | 8 B per unique n-gram |

The project deliberately does **not** claim constant memory: dedup state is
inherently proportional to the number of unique documents. What is bounded is
the per-document transient allocation, which is what causes long-tail OOMs.

## MinHash details

- Shingle hashing: CRC32 over the n-gram string (32-bit, fast, stable).
- Permutations: `(a*h + b) mod (2^61-1)` with `a, b < 2^31`, so `a*h + b` fits
  in `uint64`; using 61-bit coefficients would overflow before the modulo.
- Signatures are folded in 4,096-hash blocks; the minimum operation is
  associative, so a single block and many blocks produce the same signature
  (covered by a test).
- Duplicate shingles are harmless: MinHash takes a minimum over a multiset.

## LSH details

- 16 bands × 8 rows on 128 permutations; approximate threshold
  `(1/16)^(1/8) ≈ 0.66`.
- `transitive=True` (default) records band keys for duplicate documents too, so
  a later document that only matches an already-dropped document is still
  removed (connected components). `transitive=False` compares only against kept
  documents (first-match), which can keep a chain member.
- The first document of a cluster is kept (stable order).
- Band keys are raw signature slices (`bytes`), not Python tuples, to keep the
  bucket memory predictable.

## LSH parameter selection

- `optimal_lsh_params(threshold, num_perm)` minimizes the average of the
  false-positive area below the target and the false-negative area above it,
  with `bands * rows <= num_perm`.
- The integration is a 200-step numeric approximation; the selected S-curve
  crossing can be above the target because the objective is integrated error,
  not a single point.
- The pipeline keeps a fixed 16 × 8 default; the helper is available for recipes
  that want a different threshold.

## SimHash details

- 64 fingerprint bits; each 5-gram shingle hash votes +1/-1 on every bit, and
  the sign of the vote sets the bit.
- Near-duplicate decision is a Hamming-distance threshold. The benchmark in the
  README shows the trade-off: Hamming 3 gives precision 1.0 / recall 0.66,
  while Hamming 11 gives the best F1 (0.987) with some false positives.
- SimHash is implemented as a second algorithm and a cross-check on MinHash,
  not as the pipeline's default deduplicator.

## Decontamination details

- Reference n-grams are hashed with BLAKE2b-64 (not CRC32) because the
  reference set is small and false positives are more costly there.
- A document is dropped when `matched / total >= threshold`.
- Documents shorter than `n` have ratio 0 and are kept.

## Data mixing details

- Quotas are proportional to source weights; the integer remainder is assigned
  by largest fractional part, with source order as the deterministic tie-breaker.
- Temperature mode uses `p_i ∝ count_i ** alpha`: `alpha=0` uniform, `alpha=1`
  proportional, `0 < alpha < 1` upweights small sources. Source counts are
  streamed before sampling.
- Each source contributes a uniform reservoir sample of its quota; the reservoir
  is seeded per source, so the mix is reproducible.
- A source shorter than its quota underfills; the remainder is not
  redistributed in v1.
- Memory is O(total output documents) because each source's reservoir is held
  before writing. This is a batch mixing stage, not a streaming one.

## Benchmark methodology

- One document per fresh process; RSS sampled every 2 ms with `psutil`.
- Three runs per point; medians reported.
- Synthetic corpus: fixed seed, 20k docs × 100 tokens, 50% unique / 25% exact /
  15% one-token near / 10% benchmark copies.
- Real-text check: 1,177 `.md`/`.txt`/`.py` files from a public repository,
  used as a convenience sample, not as a public benchmark.
- Results and limitations are in the top-level README; raw files live under
  `bench/`.

## Known limitations / next steps

1. No real public web corpus yet (network unavailable in the build
   environment); the loader is JSONL-first.
2. Single process; no shard-level parallelism or checkpoint/resume.
3. Dedup index is in memory; a disk-backed index would be needed for very
   large corpora.
4. No tokenizer-level statistics or data-mixing sampler yet.

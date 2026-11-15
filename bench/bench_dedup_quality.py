"""Precision/recall of MinHash and SimHash on labeled near-duplicate pairs.

Ground truth is the exact Jaccard similarity of 5-gram shingle sets. A pair is
labeled duplicate when its true Jaccard is >= 0.7. MinHash is evaluated at the
same 0.7 threshold; SimHash is swept over Hamming thresholds and the best F1 is
reported alongside fixed thresholds.
"""

import json
import os
import random

import numpy as np

from curation_lab.hashing import shingle_hashes
from curation_lab.minhash import MinHash
from curation_lab.simhash import SimHash

N_BASE = 200
TOKENS = 100
VOCAB = 5000
NGRAM = 5
JACCARD_THRESHOLD = 0.7
SEED = 20261115


def make_doc(rng, prefix):
    return " ".join(f"{prefix}{rng.randrange(VOCAB)}" for _ in range(TOKENS))


def true_jaccard(a, b):
    sa = set(shingle_hashes(a, NGRAM, "word"))
    sb = set(shingle_hashes(b, NGRAM, "word"))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def metrics(labels, predictions):
    tp = sum(1 for y, p in zip(labels, predictions) if p and y)
    fp = sum(1 for y, p in zip(labels, predictions) if p and not y)
    fn = sum(1 for y, p in zip(labels, predictions) if not p and y)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def main():
    rng = random.Random(SEED)
    bases = [make_doc(rng, "w") for _ in range(N_BASE)]
    other = [make_doc(rng, "x") for _ in range(N_BASE)]

    pairs = []
    for base, unrelated in zip(bases, other):
        tokens = base.split()
        variants = {
            "exact": base,
            "near1": " ".join(["changed"] + tokens[1:]),
            "near5": " ".join(["changed"] * 5 + tokens[5:]),
            "near10": " ".join(["changed"] * 10 + tokens[10:]),
            "unrelated": unrelated,
        }
        for kind, variant in variants.items():
            pairs.append((base, variant, kind))

    # Cache signatures/fingerprints by text.
    minhash = MinHash(num_perm=128, seed=42, ngram=NGRAM, mode="word")
    simhash = SimHash(ngram=NGRAM, mode="word")
    minhash_cache = {}
    simhash_cache = {}

    def get_minhash(text):
        if text not in minhash_cache:
            minhash_cache[text] = minhash.signature(text)
        return minhash_cache[text]

    def get_simhash(text):
        if text not in simhash_cache:
            simhash_cache[text] = simhash.fingerprint(text)
        return simhash_cache[text]

    labels = []
    minhash_predictions = []
    simhash_distances = []
    jaccard_errors = []
    true_values = []

    for a, b, _kind in pairs:
        true = true_jaccard(a, b)
        estimate = float(np.mean(get_minhash(a) == get_minhash(b)))
        distance = simhash.hamming(get_simhash(a), get_simhash(b))
        labels.append(true >= JACCARD_THRESHOLD)
        minhash_predictions.append(estimate >= JACCARD_THRESHOLD)
        simhash_distances.append(distance)
        jaccard_errors.append(abs(estimate - true))
        true_values.append(true)

    simhash_sweep = []
    for threshold in range(1, 17):
        result = metrics(labels, [d <= threshold for d in simhash_distances])
        result["threshold"] = threshold
        simhash_sweep.append(result)
    best = max(simhash_sweep, key=lambda item: (item["f1"], -item["threshold"]))

    dup_distances = [d for d, y in zip(simhash_distances, labels) if y]
    clean_distances = [d for d, y in zip(simhash_distances, labels) if not y]

    result = {
        "config": {
            "base_docs": N_BASE,
            "pairs": len(pairs),
            "tokens_per_doc": TOKENS,
            "ngram": NGRAM,
            "jaccard_threshold": JACCARD_THRESHOLD,
            "minhash_permutations": 128,
            "simhash_bits": 64,
        },
        "duplicate_pairs_by_true_jaccard": sum(labels),
        "minhash": {
            **metrics(labels, minhash_predictions),
            "mean_abs_jaccard_error": round(float(np.mean(jaccard_errors)), 6),
        },
        "simhash": {
            "best_threshold": best["threshold"],
            "best": best,
            "at_3": metrics(labels, [d <= 3 for d in simhash_distances]),
            "at_5": metrics(labels, [d <= 5 for d in simhash_distances]),
            "mean_hamming_duplicates": round(float(np.mean(dup_distances)), 3) if dup_distances else None,
            "mean_hamming_nonduplicates": round(float(np.mean(clean_distances)), 3) if clean_distances else None,
        },
    }
    print(json.dumps(result, indent=2))
    os.makedirs("bench", exist_ok=True)
    with open("bench/dedup_quality_results.json", "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2)


if __name__ == "__main__":
    main()

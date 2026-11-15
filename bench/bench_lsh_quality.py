"""Precision/recall of the actual LSH banding decision on labeled pairs.

Ground truth: exact 5-gram Jaccard >= 0.7. A pair is predicted duplicate when
any band key matches. The same 1,000 pairs as the MinHash/SimHash benchmark are
used, so the numbers are directly comparable.
"""

import json
import os
import random

import numpy as np

from curation_lab.hashing import shingle_hashes
from curation_lab.minhash import LSH, MinHash

N_BASE = 200
TOKENS = 100
VOCAB = 5000
NGRAM = 5
THRESHOLD = 0.7
SEED = 20261115
CONFIGS = [(8, 16), (12, 10), (16, 8), (25, 5), (3, 42)]


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
        pairs.append((base, base))
        pairs.append((base, " ".join(["changed"] + tokens[1:])))
        pairs.append((base, " ".join(["changed"] * 5 + tokens[5:])))
        pairs.append((base, " ".join(["changed"] * 10 + tokens[10:])))
        pairs.append((base, unrelated))

    minhash = MinHash(num_perm=128, seed=42, ngram=NGRAM, mode="word")
    signature_cache = {}

    def signature(text):
        if text not in signature_cache:
            signature_cache[text] = minhash.signature(text)
        return signature_cache[text]

    labels = []
    signatures = []
    for a, b in pairs:
        labels.append(true_jaccard(a, b) >= THRESHOLD)
        signatures.append((signature(a), signature(b)))

    results = []
    for bands, rows in CONFIGS:
        lsh = LSH(num_perm=128, bands=bands, rows=rows)
        predictions = []
        for sig_a, sig_b in signatures:
            keys_a = list(lsh.band_keys(sig_a))
            keys_b = list(lsh.band_keys(sig_b))
            predictions.append(any(a == b for a, b in zip(keys_a, keys_b)))
        result = metrics(labels, predictions)
        result["bands"] = bands
        result["rows"] = rows
        result["s_curve_threshold"] = round((1.0 / bands) ** (1.0 / rows), 6)
        results.append(result)

    output = {
        "pairs": len(pairs),
        "duplicates_by_true_jaccard": sum(labels),
        "threshold": THRESHOLD,
        "configs": results,
    }
    print(json.dumps(output, indent=2))
    os.makedirs("bench", exist_ok=True)
    with open("bench/lsh_quality_results.json", "w", encoding="utf-8") as fout:
        json.dump(output, fout, indent=2)


if __name__ == "__main__":
    main()

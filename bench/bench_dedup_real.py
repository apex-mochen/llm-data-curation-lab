"""LSH dedup precision/recall on real WikiText-2 documents.

Builds labeled pairs from real documents (exact, one-word change, five-word
change, unrelated) and evaluates the actual any-band-match decision. Ground
truth is exact 5-gram Jaccard >= 0.7.
"""

import json
import os

from curation_lab.hashing import shingle_hashes
from curation_lab.minhash import LSH, MinHash

CORPUS = "bench/public_corpus/corpus.jsonl"
N_BASE = 200
NGRAM = 5
THRESHOLD = 0.7
CONFIGS = [(8, 16), (12, 10), (16, 8), (25, 5)]


def load_docs(path, limit):
    docs = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            text = json.loads(line).get("text", "")
            if len(text.split()) >= 50:
                docs.append(text)
            if len(docs) >= limit:
                break
    return docs


def true_jaccard(a, b):
    sa = set(shingle_hashes(a, NGRAM, "word"))
    sb = set(shingle_hashes(b, NGRAM, "word"))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def mutate(text, changes):
    tokens = text.split()
    if len(tokens) < changes + 1:
        return text
    for i in range(changes):
        tokens[i] = "changed"
    return " ".join(tokens)


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
    docs = load_docs(CORPUS, N_BASE * 2)
    bases = docs[:N_BASE]
    unrelated = docs[N_BASE : N_BASE * 2]

    pairs = []
    for base, other in zip(bases, unrelated):
        pairs.append((base, base))
        pairs.append((base, mutate(base, 1)))
        pairs.append((base, mutate(base, 5)))
        pairs.append((base, other))

    minhash = MinHash(num_perm=128, seed=42, ngram=NGRAM, mode="word")
    signature_cache = {}

    def signature(text):
        if text not in signature_cache:
            signature_cache[text] = minhash.signature(text)
        return signature_cache[text]

    labels = [true_jaccard(a, b) >= THRESHOLD for a, b in pairs]
    signatures = [(signature(a), signature(b)) for a, b in pairs]

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
        "corpus": CORPUS,
        "documents": len(docs),
        "pairs": len(pairs),
        "duplicates_by_true_jaccard": sum(labels),
        "threshold": THRESHOLD,
        "configs": results,
    }
    print(json.dumps(output, indent=2))
    os.makedirs("bench", exist_ok=True)
    with open("bench/dedup_real_results.json", "w", encoding="utf-8") as fout:
        json.dump(output, fout, indent=2)


if __name__ == "__main__":
    main()

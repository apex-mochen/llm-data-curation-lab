"""Synthetic benchmark for exact + MinHash-LSH dedup."""

import json
import os
import random
import threading
import time

import psutil

from curation_lab.dedup import ExactDedup, MinHashLSHDedup

N_DOCS = int(os.environ.get("N_DOCS", "10000"))
TOKENS = int(os.environ.get("TOKENS", "100"))
REPEAT = int(os.environ.get("REPEAT", "0"))


def build_corpus(n_docs, tokens, seed=20261115):
    rng = random.Random(seed)
    base = [" ".join(f"w{rng.randrange(2000)}" for _ in range(tokens)) for _ in range(max(1, n_docs // 10))]
    docs = []
    for i in range(n_docs):
        if i % 10 == 0:
            docs.append(base[i % len(base)])
        elif i % 10 == 1:
            parts = base[i % len(base)].split()
            parts[0] = "changed"
            docs.append(" ".join(parts))
        else:
            docs.append(" ".join(f"w{rng.randrange(2000)}" for _ in range(tokens)))
    return docs


def peak_rss_and_time(fn):
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.002)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    dups = fn()
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=1)
    return peak[0], elapsed, dups


def main():
    docs = build_corpus(N_DOCS, TOKENS)

    exact = ExactDedup()
    exact_peak, exact_seconds, exact_dups = peak_rss_and_time(
        lambda: sum(1 for doc in docs if exact.check_and_add(doc))
    )

    near = MinHashLSHDedup(num_perm=128, bands=16, rows=8, ngram=5, mode="word")
    near_peak, near_seconds, near_dups = peak_rss_and_time(lambda: sum(1 for doc in docs if near.check_and_add(doc)))

    out = {
        "n_docs": N_DOCS,
        "tokens_per_doc": TOKENS,
        "repeat": REPEAT,
        "exact_dups": exact_dups,
        "near_dups": near_dups,
        "exact_seconds": round(exact_seconds, 4),
        "near_seconds": round(near_seconds, 4),
        "near_docs_per_second": round(N_DOCS / near_seconds, 1),
        "exact_peak_rss_mb": round(exact_peak / 1024 / 1024, 2),
        "near_peak_rss_mb": round(near_peak / 1024 / 1024, 2),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()

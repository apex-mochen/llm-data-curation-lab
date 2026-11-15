"""Cross-check the lab's MinHash-LSH stage against Data-Juicer's operator.

Both variants read the same exact-deduplicated JSONL and use 128 permutations,
16 bands x 8 rows, 5-word shingles. This is a cross-check, not a leaderboard:
the clustering implementations differ.
"""

import json
import os
import threading
import time

import psutil

from curation_lab.dedup import MinHashLSHDedup

VARIANT = os.environ.get("VARIANT", "ours")
LIMIT = int(os.environ.get("LIMIT", "500"))
INPUT = os.environ.get("INPUT", "bench/data/out.jsonl")


def load_texts():
    texts = []
    with open(INPUT, encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                texts.append(json.loads(line)["text"])
            if len(texts) >= LIMIT:
                break
    return texts


def run_ours(texts):
    dedup = MinHashLSHDedup(num_perm=128, bands=16, rows=8, ngram=5, mode="word")
    removed = 0
    for text in texts:
        if dedup.check_and_add(text):
            removed += 1
    return removed


def run_datajuicer(texts):
    from data_juicer.core.data import NestedDataset
    from data_juicer.ops.deduplicator.document_minhash_deduplicator import (
        DocumentMinhashDeduplicator,
    )

    dataset = NestedDataset.from_list([{"text": text} for text in texts])
    op = DocumentMinhashDeduplicator(
        tokenization="space",
        window_size=5,
        num_permutations=128,
        num_bands=16,
        num_rows_per_band=8,
    )
    dataset = dataset.map(op.compute_hash, batch_size=500)
    processed, _ = op.process(dataset)
    return len(texts) - len(processed)


def main():
    texts = load_texts()
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
    removed = run_ours(texts) if VARIANT == "ours" else run_datajuicer(texts)
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=1)

    print(
        json.dumps(
            {
                "variant": VARIANT,
                "docs": len(texts),
                "removed": removed,
                "seconds": round(elapsed, 4),
                "docs_per_second": round(len(texts) / elapsed, 1),
                "peak_rss_mb": round(peak[0] / 1024 / 1024, 2),
            }
        )
    )


if __name__ == "__main__":
    main()

"""Compare in-memory and SQLite exact-dedup backends."""

import json
import os
import threading
import time

import psutil

from curation_lab.dedup import ExactDedup

N = int(os.environ.get("N", "100000"))
BACKEND = os.environ.get("BACKEND", "memory")
PATH = os.path.join("bench", "exact_db_test.sqlite")


def main():
    if os.path.exists(PATH):
        os.remove(PATH)
    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.005)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    dedup = ExactDedup(sqlite_path=PATH if BACKEND == "sqlite" else None)
    started = time.perf_counter()
    for i in range(N):
        dedup.check_and_add(f"exact document {i} with a stable payload")
    elapsed = time.perf_counter() - started
    dedup.close()
    stop.set()
    thread.join(timeout=1)

    print(
        json.dumps(
            {
                "backend": BACKEND,
                "n": N,
                "elapsed": round(elapsed, 3),
                "docs_per_second": round(N / elapsed, 1) if elapsed else 0,
                "peak_rss_mb": round(peak[0] / 1024 / 1024, 1),
            }
        )
    )


if __name__ == "__main__":
    main()

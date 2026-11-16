"""Ray-parallel MinHash signature throughput benchmark.

Documents are generated inside each worker (no text transfer). Each task
returns a digest of its signatures; the driver hashes the ordered shard digests
so all worker counts can be compared.
"""

import hashlib
import os
import time

import numpy as np
import ray

from curation_lab.minhash import MinHash

N = int(os.environ.get("N", "1000000"))
TOKENS = int(os.environ.get("TOKENS", "20"))
NUM_PERM = int(os.environ.get("NUM_PERM", "128"))
WORKERS = int(os.environ.get("WORKERS", "4"))
TASKS = int(os.environ.get("TASKS", "16"))


def make_text(index):
    return " ".join(f"w{(index * 131 + j) % 5000}" for j in range(TOKENS))


@ray.remote
def signature_shard(start, count, num_perm, tokens):
    minhash = MinHash(num_perm=num_perm, seed=42, ngram=5, mode="word")
    digest = hashlib.sha256()
    for index in range(start, start + count):
        signature = minhash.signature(make_text(index))
        digest.update(np.asarray(signature, dtype=np.uint64).tobytes())
    return count, digest.hexdigest()


def local_signature_digest(n, num_perm, tokens, tasks):
    minhash = MinHash(num_perm=num_perm, seed=42, ngram=5, mode="word")
    shard_size = (n + tasks - 1) // tasks
    combined = hashlib.sha256()
    for start in range(0, n, shard_size):
        digest = hashlib.sha256()
        for index in range(start, min(start + shard_size, n)):
            signature = minhash.signature(make_text(index))
            digest.update(np.asarray(signature, dtype=np.uint64).tobytes())
        combined.update(digest.hexdigest().encode("ascii"))
    return combined.hexdigest()


def main():
    started = time.perf_counter()
    local_digest = local_signature_digest(N, NUM_PERM, TOKENS, TASKS)
    local_seconds = time.perf_counter() - started

    ray.init(num_cpus=WORKERS, include_dashboard=False, logging_level="ERROR")
    shard_size = (N + TASKS - 1) // TASKS
    refs = [
        signature_shard.remote(start, min(shard_size, N - start), NUM_PERM, TOKENS)
        for start in range(0, N, shard_size)
    ]
    started = time.perf_counter()
    results = ray.get(refs)
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    for _, shard_digest in results:
        digest.update(shard_digest.encode("ascii"))
    ray.shutdown()

    print(
        {
            "n": N,
            "tokens": TOKENS,
            "num_perm": NUM_PERM,
            "workers": WORKERS,
            "tasks": len(refs),
            "local_seconds": round(local_seconds, 3),
            "ray_seconds": round(elapsed, 3),
            "speedup_vs_local": round(local_seconds / elapsed, 2) if elapsed else None,
            "local_digest": local_digest,
            "ray_digest": digest.hexdigest(),
            "digest_match": digest.hexdigest() == local_digest,
        }
    )


if __name__ == "__main__":
    main()

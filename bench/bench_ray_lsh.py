"""Ray band-sharded LSH candidate generation + global union-find.

Signatures are generated once with controlled duplicate groups; each Ray task
owns a band range and returns candidate pairs. The driver applies one global
union-find, so the cluster count must match the single-process baseline.
"""

import os
import time

import numpy as np
import ray

N = int(os.environ.get("N", "200000"))
GROUPS = int(os.environ.get("GROUPS", "20000"))
COPIES = int(os.environ.get("COPIES", "10"))
NUM_PERM = int(os.environ.get("NUM_PERM", "128"))
BANDS = int(os.environ.get("BANDS", "16"))
ROWS = int(os.environ.get("ROWS", "8"))
WORKERS = int(os.environ.get("WORKERS", "4"))
SEED = 42


class UnionFind:

    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def clusters(self):
        roots = {self.find(i) for i in range(len(self.parent))}
        return len(roots)


def make_signatures():
    rng = np.random.RandomState(SEED)
    bases = rng.randint(0, np.iinfo(np.uint64).max, size=(GROUPS, NUM_PERM), dtype=np.uint64)
    signatures = np.repeat(bases, COPIES, axis=0)
    return signatures[:N]


def band_edges(signatures, band_start, band_end):
    pairs = []
    for band in range(band_start, band_end):
        start = band * ROWS
        buckets = {}
        for doc in range(signatures.shape[0]):
            key = signatures[doc, start : start + ROWS].tobytes()
            first = buckets.get(key)
            if first is None:
                buckets[key] = doc
            else:
                pairs.append((first, doc))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def local_clusters(signatures):
    uf = UnionFind(len(signatures))
    for band in range(BANDS):
        edges = band_edges(signatures, band, band + 1)
        for a, b in edges:
            uf.union(int(a), int(b))
    return uf.clusters()


@ray.remote
def remote_band_edges(signatures, band_start, band_end):
    return band_edges(signatures, band_start, band_end)


def main():
    signatures = make_signatures()
    started = time.perf_counter()
    expected = local_clusters(signatures)
    local_seconds = time.perf_counter() - started

    ray.init(num_cpus=WORKERS, include_dashboard=False, logging_level="ERROR")
    ranges = np.linspace(0, BANDS, WORKERS + 1).astype(int)
    signatures_ref = ray.put(signatures)
    started = time.perf_counter()
    refs = [
        remote_band_edges.remote(signatures_ref, int(ranges[i]), int(ranges[i + 1]))
        for i in range(WORKERS)
    ]
    edge_arrays = ray.get(refs)
    ray.shutdown()

    uf = UnionFind(len(signatures))
    for edges in edge_arrays:
        for a, b in edges:
            uf.union(int(a), int(b))
    elapsed = time.perf_counter() - started
    got = uf.clusters()
    print(
        {
            "n": len(signatures),
            "groups": GROUPS,
            "copies": COPIES,
            "bands": BANDS,
            "rows": ROWS,
            "workers": WORKERS,
            "local_seconds": round(local_seconds, 3),
            "ray_seconds": round(elapsed, 3),
            "speedup_vs_local": round(local_seconds / elapsed, 2) if elapsed else None,
            "local_clusters": expected,
            "ray_clusters": got,
            "cluster_match": expected == got,
        }
    )


if __name__ == "__main__":
    main()

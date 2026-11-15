"""Choose LSH bands/rows from a target Jaccard threshold.

The S-curve collision probability for bands ``b`` and rows ``r`` is
``1 - (1 - s**r)**b``. We pick the (b, r) pair that minimizes the average of
the false-positive area below the target and the false-negative area above it,
with ``b * r <= num_perm``.
"""

from typing import Dict


def collision_probability(similarity: float, bands: int, rows: int) -> float:
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be in [0, 1]")
    return 1.0 - (1.0 - similarity**rows) ** bands


def optimal_lsh_params(threshold: float, num_perm: int, steps: int = 200) -> Dict:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if num_perm <= 0:
        raise ValueError("num_perm must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    best = None
    for bands in range(1, num_perm + 1):
        rows = num_perm // bands
        if rows < 1:
            continue
        false_positive = 0.0
        false_negative = 0.0
        for i in range(steps + 1):
            similarity = i / steps
            probability = collision_probability(similarity, bands, rows)
            if similarity < threshold:
                false_positive += probability
            else:
                false_negative += 1.0 - probability
        false_positive *= threshold / steps
        false_negative *= (1.0 - threshold) / steps
        score = 0.5 * false_positive + 0.5 * false_negative
        if best is None or score < best["score"]:
            best = {
                "bands": bands,
                "rows": rows,
                "score": score,
                "s_curve_threshold": (1.0 / bands) ** (1.0 / rows),
            }

    if best is None:
        raise ValueError("no valid bands/rows pair")
    best["num_perm"] = num_perm
    best["target_threshold"] = threshold
    best["score"] = round(best["score"], 8)
    best["s_curve_threshold"] = round(best["s_curve_threshold"], 6)
    return best

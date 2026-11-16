"""Synthetic separation check for the lexical quality baseline."""

import json
import random

from curation_lab.quality import QualityScorer, quality_score, quality_stats

rng = random.Random(20261116)
scorer = QualityScorer()
vocab = [f"w{i}" for i in range(500)]


def clean_text():
    return " ".join(rng.choice(vocab) for _ in range(30))


def repeated_text():
    return " ".join(["the"] * 30)


def low_diversity_text():
    return " ".join(rng.choice(["alpha", "beta", "gamma"]) for _ in range(30))


def summarize(name, maker, n=1000):
    scores = [quality_score(quality_stats(maker())) for _ in range(n)]
    kept = sum(score >= scorer.min_score for score in scores)
    return {
        "name": name,
        "n": n,
        "mean_score": round(sum(scores) / n, 4),
        "keep_rate": round(kept / n, 4),
    }


def main():
    results = [
        summarize("clean_random", clean_text),
        summarize("repeated_token", repeated_text),
        summarize("low_diversity", low_diversity_text),
    ]
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

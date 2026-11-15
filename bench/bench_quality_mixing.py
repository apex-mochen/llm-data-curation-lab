"""Compare temperature mixing with and without a quality weight.

Source A: 1,000 unique documents, quality 1.0.
Source B: 200 documents (100 copies of A, 100 unique), quality 0.25.
beta=0 is plain temperature mixing; beta=1 applies p_i ∝ count_i * quality_i.
"""

import json
import os

from bench_mixing_quality import read_output, write_sources

from curation_lab.mixing import MixSource, mix_jsonl_quality

TOTAL = 400
ALPHA = 1.0
BETA_VALUES = [0.0, 1.0]
SEEDS = [1, 2, 3]


def main():
    a_path, b_path = write_sources()
    sources = [MixSource(a_path, 1.0, 1.0), MixSource(b_path, 1.0, 0.25)]
    results = []
    for beta in BETA_VALUES:
        per_seed = []
        for seed in SEEDS:
            output = os.path.join("bench", "mixing_eval", f"quality_beta_{beta}_seed_{seed}.jsonl")
            stats = mix_jsonl_quality(sources, output, TOTAL, alpha=ALPHA, beta=beta, seed=seed)
            row = read_output(output)
            row["quotas"] = stats["quotas"]
            per_seed.append(row)
        averaged = {
            key: round(sum(item[key] for item in per_seed) / len(per_seed), 2)
            for key in ("a_docs", "b_docs", "b_copy_docs", "duplicate_texts", "unique_texts", "output")
        }
        averaged["quotas"] = per_seed[0]["quotas"]
        averaged["beta"] = beta
        results.append(averaged)

    output = {
        "config": {
            "source_a_quality": 1.0,
            "source_b_quality": 0.25,
            "alpha": ALPHA,
            "total": TOTAL,
            "seeds": SEEDS,
        },
        "results": results,
    }
    print(json.dumps(output, indent=2))
    with open("bench/quality_mixing_results.json", "w", encoding="utf-8") as fout:
        json.dump(output, fout, indent=2)


if __name__ == "__main__":
    main()

"""Downstream proxy for mixing: how many mixed documents pass quality filters.

Source A: 1,000 clean documents.
Source B: 200 documents: 100 copies of A + 100 short/symbol-heavy noisy docs.
We compare temperature mixing (beta=0) with quality-weighted mixing (beta=1)
using filter pass rate, unique texts and duplicate texts as proxies.
"""

import json
import os

from curation_lab.filters import FilterConfig, check, normalize_text
from curation_lab.mixing import MixSource, mix_jsonl_quality

OUT_DIR = "bench/mixing_eval"
TOTAL = 400
ALPHA = 1.0
BETA_VALUES = [0.0, 1.0]
SEEDS = [1, 2, 3]


def write_sources():
    os.makedirs(OUT_DIR, exist_ok=True)
    a_path = os.path.join(OUT_DIR, "downstream_a.jsonl")
    b_path = os.path.join(OUT_DIR, "downstream_b.jsonl")
    a_texts = []
    with open(a_path, "w", encoding="utf-8") as fout:
        for i in range(1000):
            text = f"alpha document {i} " + " ".join(f"word{i}alpha{j}beta" for j in range(80))
            a_texts.append(text)
            fout.write(json.dumps({"id": f"A{i}", "text": text}, ensure_ascii=False) + "\n")
    with open(b_path, "w", encoding="utf-8") as fout:
        for i in range(100):
            fout.write(json.dumps({"id": f"B{i}", "text": a_texts[i]}, ensure_ascii=False) + "\n")
        for i in range(100):
            fout.write(json.dumps({"id": f"B{100 + i}", "text": "!!! ??? 123"}, ensure_ascii=False) + "\n")
    return a_path, b_path


def evaluate(path):
    filters = FilterConfig(min_chars=50)
    texts = []
    passed = 0
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            text = normalize_text(json.loads(line)["text"])
            texts.append(text)
            if check(text, filters) is None:
                passed += 1
    return {
        "output": len(texts),
        "passed_filters": passed,
        "pass_rate": round(passed / len(texts), 4) if texts else 0.0,
        "unique_texts": len(set(texts)),
        "duplicate_texts": len(texts) - len(set(texts)),
    }


def main():
    a_path, b_path = write_sources()
    sources = [MixSource(a_path, 1.0, 1.0), MixSource(b_path, 1.0, 0.25)]
    results = []
    for beta in BETA_VALUES:
        per_seed = []
        for seed in SEEDS:
            output = os.path.join(OUT_DIR, f"downstream_beta_{beta}_seed_{seed}.jsonl")
            stats = mix_jsonl_quality(sources, output, TOTAL, alpha=ALPHA, beta=beta, seed=seed)
            row = evaluate(output)
            row["quotas"] = stats["quotas"]
            per_seed.append(row)
        averaged = {
            key: round(sum(item[key] for item in per_seed) / len(per_seed), 2)
            for key in ("output", "passed_filters", "pass_rate", "unique_texts", "duplicate_texts")
        }
        averaged["quotas"] = per_seed[0]["quotas"]
        averaged["beta"] = beta
        results.append(averaged)

    payload = {
        "config": {
            "source_a": 1000,
            "source_b_copies": 100,
            "source_b_noisy": 100,
            "source_b_quality": 0.25,
            "total": TOTAL,
            "min_chars": 50,
            "seeds": SEEDS,
        },
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    with open("bench/mixing_downstream_results.json", "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)


if __name__ == "__main__":
    main()

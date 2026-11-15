"""Compare proportional and temperature mixing when a small source is noisy.

Source A has 1,000 unique documents. Source B has 200 documents: 100 are exact
copies of A documents and 100 are unique. We mix 400 documents at alpha=1
(proportional), alpha=0.5 and alpha=0, then measure how much of the output is
duplicate text and how many unique documents survive.
"""

import json
import os
import random

from curation_lab.mixing import MixSource, mix_jsonl_temperature

OUT_DIR = "bench/mixing_eval"
A_DOCS = 1000
B_COPIES = 100
B_UNIQUE = 100
TOTAL = 400
ALPHAS = [1.0, 0.5, 0.0]
SEEDS = [1, 2, 3]


def write_sources():
    os.makedirs(OUT_DIR, exist_ok=True)
    a_path = os.path.join(OUT_DIR, "source_a.jsonl")
    b_path = os.path.join(OUT_DIR, "source_b.jsonl")
    a_texts = []
    with open(a_path, "w", encoding="utf-8") as fout:
        for i in range(A_DOCS):
            text = f"alpha domain document {i} " + " ".join(f"token{i}_{j}" for j in range(80))
            a_texts.append(text)
            fout.write(json.dumps({"id": f"A{i}", "text": text}, ensure_ascii=False) + "\n")
    with open(b_path, "w", encoding="utf-8") as fout:
        for i in range(B_COPIES):
            fout.write(json.dumps({"id": f"B{i}", "text": a_texts[i]}, ensure_ascii=False) + "\n")
        for i in range(B_UNIQUE):
            text = f"beta domain unique document {i} " + " ".join(f"btoken{i}_{j}" for j in range(80))
            fout.write(json.dumps({"id": f"B{B_COPIES + i}", "text": text}, ensure_ascii=False) + "\n")
    return a_path, b_path


def read_output(path):
    texts = []
    a_docs = 0
    b_docs = 0
    b_copy_docs = 0
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            texts.append(record["text"])
            if record["id"].startswith("A"):
                a_docs += 1
            else:
                b_docs += 1
                suffix = record["id"][1:]
                if suffix.isdigit() and int(suffix) < B_COPIES:
                    b_copy_docs += 1
    unique = len(set(texts))
    return {
        "a_docs": a_docs,
        "b_docs": b_docs,
        "b_copy_docs": b_copy_docs,
        "duplicate_texts": len(texts) - unique,
        "unique_texts": unique,
        "output": len(texts),
    }


def main():
    a_path, b_path = write_sources()
    sources = [MixSource(a_path, 1.0), MixSource(b_path, 1.0)]
    results = []
    for alpha in ALPHAS:
        per_seed = []
        for seed in SEEDS:
            output = os.path.join(OUT_DIR, f"mixed_alpha_{alpha}_seed_{seed}.jsonl")
            mix_jsonl_temperature(sources, output, TOTAL, alpha, seed=seed)
            per_seed.append(read_output(output))
        averaged = {
            key: round(sum(item[key] for item in per_seed) / len(per_seed), 2)
            for key in ("a_docs", "b_docs", "b_copy_docs", "duplicate_texts", "unique_texts", "output")
        }
        averaged["alpha"] = alpha
        results.append(averaged)

    output = {
        "config": {
            "source_a_unique": A_DOCS,
            "source_b_copies": B_COPIES,
            "source_b_unique": B_UNIQUE,
            "total": TOTAL,
            "seeds": SEEDS,
        },
        "results": results,
    }
    print(json.dumps(output, indent=2))
    with open("bench/mixing_quality_results.json", "w", encoding="utf-8") as fout:
        json.dump(output, fout, indent=2)


if __name__ == "__main__":
    main()

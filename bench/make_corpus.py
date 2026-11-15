"""Generate a reproducible synthetic corpus for the pipeline benchmark."""

import argparse
import json
import random
import string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=20000)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rng = random.Random(20261115)
    out_dir = args.out_dir
    import os

    os.makedirs(out_dir, exist_ok=True)
    vocab = ["".join(rng.choice(string.ascii_lowercase) for _ in range(7)) for _ in range(5000)]

    def make_doc(prefix):
        return " ".join(rng.choice(vocab) for _ in range(args.tokens))

    evals = [make_doc("eval") for _ in range(50)]
    with open(f"{out_dir}/eval.jsonl", "w", encoding="utf-8") as fout:
        for i, text in enumerate(evals):
            fout.write(json.dumps({"id": f"eval-{i}", "text": text}) + "\n")

    unique = [make_doc("w") for _ in range(args.docs)]
    records = []
    for i in range(args.docs):
        bucket = i % 20
        if bucket < 10:  # 50% unique
            text = unique[i]
        elif bucket < 15:  # 25% exact duplicate of a unique doc
            text = unique[i - 10]
        elif bucket < 18:  # 15% near duplicate of an already emitted doc
            parts = unique[i - 10].split()
            parts[0] = "changed"
            text = " ".join(parts)
        else:  # 10% contaminated
            text = evals[(i // 20) % len(evals)]
        records.append({"id": i, "text": text})

    with open(f"{out_dir}/corpus.jsonl", "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"docs": args.docs, "tokens": args.tokens, "dir": out_dir}))


if __name__ == "__main__":
    main()

"""Fetch a small public text corpus and write it as JSONL."""

import argparse
import json
import os

import datasets
from datasets import load_dataset

# The local test environment has an import-only torch stub; disabling torch
# detection keeps datasets on the pure-python iterable path.
datasets.config.TORCH_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = load_dataset(args.dataset, args.config, split=args.split)

    written = 0
    total_chars = 0
    with open(f"{args.out_dir}/corpus.jsonl", "w", encoding="utf-8") as fout:
        for index, row in enumerate(dataset):
            if written >= args.limit:
                break
            text = row.get(args.text_key, "")
            if not isinstance(text, str) or len(text.strip()) < args.min_chars:
                continue
            text = text.strip()
            fout.write(json.dumps({"id": f"{args.dataset}-{index}", "text": text}, ensure_ascii=False) + "\n")
            written += 1
            total_chars += len(text)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "config": args.config,
                "split": args.split,
                "docs": written,
                "chars": total_chars,
                "out": f"{args.out_dir}/corpus.jsonl",
            }
        )
    )


if __name__ == "__main__":
    main()

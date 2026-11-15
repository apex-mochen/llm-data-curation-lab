"""Positive/negative control for n-gram decontamination on public data.

Clean corpus: WikiText-2 documents. Reference: GSM8K test questions. We inject
exact and one-word-changed copies of a subset of the questions, then measure the
decontaminator's precision/recall at n=13, threshold=0.5.
"""

import argparse
import json
import os

import datasets
from datasets import load_dataset

from curation_lab.decontam import NGramDecontaminator

datasets.config.TORCH_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikitext-corpus", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--n-inject", type=int, default=100)
    parser.add_argument("--n-clean", type=int, default=2000)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    eval_texts = [row["question"] for row in gsm][: args.n_eval]

    clean = []
    with open(args.wikitext_corpus, encoding="utf-8") as fin:
        for line in fin:
            if len(clean) >= args.n_clean:
                break
            if line.strip():
                clean.append(json.loads(line)["text"])

    n_inject = min(args.n_inject, len(eval_texts) // 2)
    exact_texts = eval_texts[:n_inject]
    near_texts = []
    for text in eval_texts[n_inject : 2 * n_inject]:
        parts = text.split()
        parts[0] = "changed"
        near_texts.append(" ".join(parts))

    corpus = []
    for i, text in enumerate(clean):
        corpus.append({"id": f"clean-{i}", "label": "clean", "text": text})
    for i, text in enumerate(exact_texts):
        corpus.append({"id": f"exact-{i}", "label": "exact", "text": text})
    for i, text in enumerate(near_texts):
        corpus.append({"id": f"near-{i}", "label": "near", "text": text})

    decontam = NGramDecontaminator(eval_texts, n=args.ngram, threshold=args.threshold)
    tp = fp = fn = 0
    ratios = {"clean": [], "exact": [], "near": []}
    for record in corpus:
        ratio = decontam.match_ratio(record["text"])
        ratios[record["label"]].append(ratio)
        predicted = ratio >= args.threshold
        actual = record["label"] != "clean"
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    result = {
        "clean_docs": len(clean),
        "eval_questions": len(eval_texts),
        "injected_exact": len(exact_texts),
        "injected_near": len(near_texts),
        "index_ngrams": decontam.index_size,
        "ngram": args.ngram,
        "threshold": args.threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "max_clean_ratio": round(max(ratios["clean"]) if ratios["clean"] else 0.0, 6),
        "min_exact_ratio": round(min(ratios["exact"]) if ratios["exact"] else 0.0, 6),
        "min_near_ratio": round(min(ratios["near"]) if ratios["near"] else 0.0, 6),
    }
    print(json.dumps(result, indent=2))
    with open(os.path.join(args.out_dir, "result.json"), "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=2)


if __name__ == "__main__":
    main()
